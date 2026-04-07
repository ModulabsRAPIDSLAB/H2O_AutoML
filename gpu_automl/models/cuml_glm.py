"""cuML GLM (Generalized Linear Model) GPU wrapper.

Phase 0 Technical Spike Decision — Non-negative Meta Learner:
  cuML LogisticRegression does NOT natively support non-negative weight
  constraints. We implement the following strategy (PRD Section 5.0):

  Primary (Rank 1): Post-hoc clipping + renormalization
    - Train cuML GLM normally on GPU
    - Clip negative coefficients to 0
    - Renormalize remaining weights to sum to 1
    - Pros: GPU-native, simple, fast
    - Cons: Not mathematically identical to constrained optimization

  Fallback (Rank 3): scipy.optimize.nnls on CPU
    - Transfer Level-One Data to CPU (small: N x L matrix)
    - Run scipy NNLS for exact non-negative least squares
    - Pros: Mathematically correct
    - Cons: CPU transfer (negligible for small L1 data)

  The MetaLearner class in ensemble/meta_learner.py implements both
  strategies and selects based on configuration.
"""

from __future__ import annotations

from typing import Any, Optional

import cudf
import numpy as np

from gpu_automl.models.base import BaseModel, ModelConfig


class CuMLGLM(BaseModel):
    """cuML Logistic/Linear Regression wrapper.

    For classification: LogisticRegression
    For regression: LinearRegression (Ridge)
    """

    def __init__(
        self,
        config: ModelConfig,
        seed: int = 42,
        task: str = "classification",
    ):
        super().__init__(config, seed)
        self.task = task

    def _build_model(self) -> Any:
        params = self.config.params.copy()

        if self.task == "classification":
            from cuml.linear_model import LogisticRegression

            base_params = {
                "max_iter": params.pop("max_iter", 1000),
                "C": params.pop("C", 1.0),
                "penalty": params.pop("penalty", "l2"),
            }
            base_params.update(params)
            return LogisticRegression(**base_params)
        else:
            from cuml.linear_model import LinearRegression

            base_params = {
                "fit_intercept": params.pop("fit_intercept", True),
                "algorithm": params.pop("algorithm", "eig"),
            }
            base_params.update(params)
            return LinearRegression(**base_params)

    def _fit_impl(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
        X_val: Optional[cudf.DataFrame] = None,
        y_val: Optional[cudf.Series] = None,
    ) -> None:
        self._model.fit(X, y)

    def _predict_impl(self, X: cudf.DataFrame) -> np.ndarray:
        preds = self._model.predict(X)
        if hasattr(preds, "values"):
            return preds.to_numpy()
        if hasattr(preds, "get"):
            return preds.get()
        return np.asarray(preds)

    def _predict_proba_impl(self, X: cudf.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification tasks.")
        preds = self._model.predict_proba(X)
        if hasattr(preds, "values"):
            return preds.to_numpy()
        if hasattr(preds, "get"):
            return preds.get()
        return np.asarray(preds)
