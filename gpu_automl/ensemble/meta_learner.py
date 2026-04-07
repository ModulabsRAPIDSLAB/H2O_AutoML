"""Meta Learner for Stacked Ensemble.

Implements non-negative GLM meta learner following H2O's approach.
Uses post-hoc clipping (primary) or scipy NNLS (fallback).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import cudf
import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


class NNStrategy(Enum):
    """Non-negative weight strategy for meta learner."""

    CLIP = "clip"  # Post-hoc clipping + renormalization (GPU-native)
    NNLS = "nnls"  # scipy NNLS on CPU (exact constraint)


class MetaLearner:
    """Non-negative GLM Meta Learner for Stacked Ensemble.

    Trains a GLM on Level-One Data (OOF predictions from base models)
    with non-negative weight constraints and optional L1 regularization
    for sparse ensemble selection.

    Parameters
    ----------
    task : str
        "classification" or "regression".
    strategy : NNStrategy
        Non-negative weight strategy.
    alpha : float
        L1 regularization strength for sparse selection.
    """

    def __init__(
        self,
        task: str = "classification",
        strategy: NNStrategy = NNStrategy.CLIP,
        alpha: float = 0.01,
        seed: int = 42,
    ):
        self.task = task
        self.strategy = strategy
        self.alpha = alpha
        self.seed = seed
        self._weights: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._model = None
        self._is_fitted = False

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._weights

    def fit(self, level_one_data: np.ndarray, y: np.ndarray) -> "MetaLearner":
        """Fit meta learner on Level-One Data.

        Parameters
        ----------
        level_one_data : np.ndarray
            Shape (n_samples, n_base_models). OOF predictions from base models.
        y : np.ndarray
            True target values.
        """
        if self.strategy == NNStrategy.CLIP:
            self._fit_clip(level_one_data, y)
        elif self.strategy == NNStrategy.NNLS:
            self._fit_nnls(level_one_data, y)

        self._is_fitted = True
        logger.info(
            f"MetaLearner fitted (strategy={self.strategy.value}). "
            f"Weights: {self._weights}"
        )
        return self

    def _fit_clip(self, X: np.ndarray, y: np.ndarray) -> None:
        """Strategy 1: Train cuML GLM then clip negative weights."""
        X_gpu = cudf.DataFrame(X)
        y_gpu = cudf.Series(y)

        if self.task == "classification":
            from cuml.linear_model import LogisticRegression

            model = LogisticRegression(
                C=1.0 / max(self.alpha, 1e-8),
                penalty="l1",
                max_iter=1000,
            )
        else:
            from cuml.linear_model import Ridge

            model = Ridge(alpha=self.alpha)

        model.fit(X_gpu, y_gpu)

        # Extract coefficients — handle cudf Series, cupy array, or numpy
        coef = model.coef_
        if hasattr(coef, "to_numpy"):  # cudf Series
            coef = coef.to_numpy()
        elif hasattr(coef, "get"):  # cupy array
            coef = coef.get()
        coef = np.asarray(coef, dtype=np.float64).flatten()

        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            if hasattr(intercept, "to_numpy"):
                intercept = intercept.to_numpy()
            elif hasattr(intercept, "get"):
                intercept = intercept.get()
            self._intercept = float(np.asarray(intercept, dtype=np.float64).flatten()[0])

        # Post-hoc clipping: set negative weights to 0
        coef = np.maximum(coef, 0.0)

        # Renormalize to sum to 1 (for interpretability)
        # Also zero out intercept since we're doing weighted average of probabilities
        total = coef.sum()
        if total > 0:
            coef = coef / total
        else:
            coef = np.ones_like(coef) / len(coef)
            logger.warning("All weights were negative. Using equal weights.")

        self._weights = coef
        self._intercept = 0.0  # intercept is invalid after clipping+renorm
        self._model = model

    def _fit_nnls(self, X: np.ndarray, y: np.ndarray) -> None:
        """Strategy 3: scipy NNLS on CPU for exact non-negative constraint."""
        from scipy.optimize import nnls

        # Level-One Data is small (N x L), so CPU transfer is negligible
        weights, residual = nnls(X, y)

        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(X.shape[1]) / X.shape[1]
            logger.warning("NNLS returned all-zero weights. Using equal weights.")

        self._weights = weights
        self._intercept = 0.0

    def predict(self, level_one_data: np.ndarray) -> np.ndarray:
        """Predict using weighted combination of base model predictions."""
        if not self._is_fitted:
            raise RuntimeError("MetaLearner is not fitted.")

        # Weighted average: predictions = X @ weights
        preds = level_one_data @ self._weights + self._intercept
        return preds

    def predict_proba(self, level_one_data: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        preds = self.predict(level_one_data)
        # Clip to valid probability range
        return np.clip(preds, 0.0, 1.0)
