"""Stacked Ensemble implementation following H2O's two-type approach.

Two ensemble types (from H2O AutoML design):
  1. All Models Ensemble: uses all base models
  2. Best of Family Ensemble: uses best model from each algorithm family
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cudf
import numpy as np

from gpu_automl.ensemble.meta_learner import MetaLearner, NNStrategy

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result of ensemble creation."""

    ensemble_id: str
    ensemble_type: str  # "all_models" or "best_of_family"
    base_model_ids: list[str]
    weights: np.ndarray
    metric_value: float


class StackedEnsemble:
    """H2O-style Stacked Ensemble using OOF predictions.

    Creates two types of ensembles:
    1. All Models: combines all base models
    2. Best of Family: combines best model per algorithm family

    Uses a non-negative GLM meta learner with L1 regularization.
    """

    def __init__(
        self,
        task: str = "classification",
        nn_strategy: NNStrategy = NNStrategy.CLIP,
        alpha: float = 0.01,
        seed: int = 42,
    ):
        self.task = task
        self.nn_strategy = nn_strategy
        self.alpha = alpha
        self.seed = seed
        self._all_models_meta: Optional[MetaLearner] = None
        self._bof_meta: Optional[MetaLearner] = None
        self._all_models_ids: list[str] = []
        self._bof_model_ids: list[str] = []
        self._all_fold_models: dict[str, list[Any]] = {}

    def build_all_models_ensemble(
        self,
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        fold_models: dict[str, list[Any]],
        metric_fn: Any,
    ) -> EnsembleResult:
        """Build ensemble using all base models.

        Parameters
        ----------
        oof_predictions : dict[str, np.ndarray]
            Model ID -> OOF predictions array.
        y : np.ndarray
            True target values.
        fold_models : dict[str, list[Any]]
            Model ID -> list of fold-trained models.
        metric_fn : callable
            Metric function.

        Returns
        -------
        EnsembleResult
        """
        model_ids = sorted(oof_predictions.keys())
        if len(model_ids) < 2:
            logger.warning("Need at least 2 models for stacking. Skipping.")
            return EnsembleResult(
                ensemble_id="SE_AllModels",
                ensemble_type="all_models",
                base_model_ids=model_ids,
                weights=np.array([1.0]),
                metric_value=0.0,
            )

        level_one = self._build_level_one_data(oof_predictions, model_ids)

        meta = MetaLearner(
            task=self.task,
            strategy=self.nn_strategy,
            alpha=self.alpha,
            seed=self.seed,
        )
        meta.fit(level_one, y)

        # Evaluate ensemble OOF performance
        ensemble_oof = meta.predict(level_one)
        if self.task == "classification":
            ensemble_oof = np.clip(ensemble_oof, 0, 1)
        metric_value = metric_fn(y, ensemble_oof)

        self._all_models_meta = meta
        self._all_models_ids = model_ids
        self._all_fold_models = fold_models

        logger.info(
            f"All Models Ensemble built: {len(model_ids)} models, "
            f"metric={metric_value:.6f}"
        )

        return EnsembleResult(
            ensemble_id="SE_AllModels",
            ensemble_type="all_models",
            base_model_ids=model_ids,
            weights=meta.weights,
            metric_value=metric_value,
        )

    def build_best_of_family_ensemble(
        self,
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        model_families: dict[str, str],
        model_metrics: dict[str, float],
        fold_models: dict[str, list[Any]],
        metric_fn: Any,
        higher_is_better: bool = True,
    ) -> EnsembleResult:
        """Build ensemble using best model from each algorithm family.

        Parameters
        ----------
        model_families : dict[str, str]
            Model ID -> family name (e.g., "xgboost", "rf", "glm").
        model_metrics : dict[str, float]
            Model ID -> CV metric value.
        higher_is_better : bool
            Whether higher metric is better.
        """
        # Select best model per family
        family_best: dict[str, tuple[str, float]] = {}
        for mid, family in model_families.items():
            metric = model_metrics.get(mid, 0.0)
            if family not in family_best:
                family_best[family] = (mid, metric)
            else:
                _, best_metric = family_best[family]
                if (higher_is_better and metric > best_metric) or (
                    not higher_is_better and metric < best_metric
                ):
                    family_best[family] = (mid, metric)

        selected_ids = [mid for mid, _ in family_best.values()]
        if len(selected_ids) < 2:
            logger.warning("Need >= 2 families for BoF ensemble. Skipping.")
            return EnsembleResult(
                ensemble_id="SE_BestOfFamily",
                ensemble_type="best_of_family",
                base_model_ids=selected_ids,
                weights=np.array([1.0]),
                metric_value=0.0,
            )

        level_one = self._build_level_one_data(oof_predictions, selected_ids)

        meta = MetaLearner(
            task=self.task,
            strategy=self.nn_strategy,
            alpha=self.alpha,
            seed=self.seed,
        )
        meta.fit(level_one, y)

        ensemble_oof = meta.predict(level_one)
        if self.task == "classification":
            ensemble_oof = np.clip(ensemble_oof, 0, 1)
        metric_value = metric_fn(y, ensemble_oof)

        self._bof_meta = meta
        self._bof_model_ids = selected_ids

        logger.info(
            f"Best of Family Ensemble built: {len(selected_ids)} families, "
            f"metric={metric_value:.6f}"
        )

        return EnsembleResult(
            ensemble_id="SE_BestOfFamily",
            ensemble_type="best_of_family",
            base_model_ids=selected_ids,
            weights=meta.weights,
            metric_value=metric_value,
        )

    def predict(
        self,
        X: cudf.DataFrame,
        ensemble_type: str = "all_models",
    ) -> np.ndarray:
        """Predict using the stacked ensemble.

        Averages predictions from all fold models for each base model,
        then applies meta learner weights.
        """
        if ensemble_type == "all_models":
            meta = self._all_models_meta
            model_ids = self._all_models_ids
        else:
            meta = self._bof_meta
            model_ids = self._bof_model_ids

        if meta is None:
            raise RuntimeError(f"Ensemble '{ensemble_type}' is not built.")

        # Get base model predictions (average across folds)
        base_preds = []
        for mid in model_ids:
            fold_models = self._all_fold_models[mid]
            fold_preds = []
            for fm in fold_models:
                if self.task == "classification":
                    p = fm.predict_proba(X)
                    if p.ndim == 2:
                        p = p[:, 1]
                else:
                    p = fm.predict(X)
                fold_preds.append(p)
            avg_pred = np.mean(fold_preds, axis=0)
            base_preds.append(avg_pred)

        level_one = np.column_stack(base_preds)
        return meta.predict(level_one)

    def _build_level_one_data(
        self, oof_predictions: dict[str, np.ndarray], model_ids: list[str]
    ) -> np.ndarray:
        """Stack OOF predictions into Level-One Data matrix."""
        columns = []
        for mid in model_ids:
            oof = oof_predictions[mid]
            if oof.ndim == 2:
                # Multiclass: use class 1 probability for binary, all for multi
                if oof.shape[1] == 2:
                    columns.append(oof[:, 1])
                else:
                    for c in range(oof.shape[1]):
                        columns.append(oof[:, c])
            else:
                columns.append(oof)
        return np.column_stack(columns)
