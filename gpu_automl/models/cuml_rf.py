"""cuML Random Forest GPU model wrapper.

Note (Phase 0 Technical Spike):
  cuML RF has a GPU max_depth limit (typically 16 for GPU execution).
  Strategy: use reasonable defaults within GPU limits. If deeper trees
  are needed, log a warning and cap at max supported depth.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cudf
import numpy as np

from gpu_automl.models.base import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

# cuML GPU RF max_depth limit
_CUML_RF_MAX_DEPTH_GPU = 16


class CuMLRandomForest(BaseModel):
    """cuML RandomForest wrapper for GPU-native training.

    Uses cuml.ensemble.RandomForestClassifier/Regressor.
    Handles the GPU max_depth limitation by capping and warning.

    Early stopping alternative: since cuML RF doesn't support early stopping
    directly, we use n_estimators grid search as the HPO strategy (Phase 4).
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
        if self.task == "classification":
            from cuml.ensemble import RandomForestClassifier

            model_cls = RandomForestClassifier
        else:
            from cuml.ensemble import RandomForestRegressor

            model_cls = RandomForestRegressor

        params = self.config.params.copy()

        # Enforce GPU max_depth limit
        if "max_depth" in params and params["max_depth"] > _CUML_RF_MAX_DEPTH_GPU:
            logger.warning(
                f"cuML RF max_depth {params['max_depth']} exceeds GPU limit "
                f"{_CUML_RF_MAX_DEPTH_GPU}. Capping to {_CUML_RF_MAX_DEPTH_GPU}."
            )
            params["max_depth"] = _CUML_RF_MAX_DEPTH_GPU

        base_params = {
            "n_estimators": params.pop("n_estimators", 100),
            "max_depth": params.pop("max_depth", 12),
            "max_features": params.pop("max_features", "sqrt"),
            "random_state": self.seed,
        }
        base_params.update(params)

        return model_cls(**base_params)

    def _fit_impl(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
        X_val: Optional[cudf.DataFrame] = None,
        y_val: Optional[cudf.Series] = None,
    ) -> None:
        # cuML RF doesn't support eval_set / early stopping
        self._model.fit(X, y)

    def _predict_impl(self, X: cudf.DataFrame) -> np.ndarray:
        preds = self._model.predict(X)
        if hasattr(preds, "values"):  # cudf Series
            return preds.to_numpy()
        if hasattr(preds, "get"):  # cupy
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
