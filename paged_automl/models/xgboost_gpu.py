"""XGBoost GPU model wrapper with early stopping support."""

from __future__ import annotations

from typing import Any, Optional

import cudf
import numpy as np
import xgboost as xgb

from paged_automl.models.base import BaseModel, ModelConfig


class XGBoostGPU(BaseModel):
    """XGBoost with GPU histogram (tree_method='hist', device='cuda').

    Supports early stopping via validation set.
    """

    def __init__(
        self,
        config: ModelConfig,
        seed: int = 42,
        task: str = "classification",
        early_stopping_rounds: int = 10,
    ):
        super().__init__(config, seed)
        self.task = task
        self.early_stopping_rounds = early_stopping_rounds

    def _build_model(self) -> Any:
        base_params = {
            "tree_method": "hist",
            "device": "cuda",
            "random_state": self.seed,
            "verbosity": 0,
            "n_jobs": 1,
        }

        if self.task == "classification":
            base_params["eval_metric"] = "logloss"
        else:
            base_params["eval_metric"] = "rmse"

        # Merge user params (override defaults)
        params = {**base_params, **self.config.params}

        if self.task == "classification":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    def _fit_impl(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
        X_val: Optional[cudf.DataFrame] = None,
        y_val: Optional[cudf.Series] = None,
    ) -> None:
        fit_kwargs: dict[str, Any] = {}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        # XGBoost sklearn API uses callbacks for early stopping in recent versions
        if self.early_stopping_rounds and X_val is not None:
            self._model.set_params(
                early_stopping_rounds=self.early_stopping_rounds,
            )

        self._model.fit(X, y, **fit_kwargs)

    def _predict_impl(self, X: cudf.DataFrame) -> np.ndarray:
        preds = self._model.predict(X)
        if hasattr(preds, "get"):  # cupy array
            return preds.get()
        return np.asarray(preds)

    def _predict_proba_impl(self, X: cudf.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification tasks.")
        preds = self._model.predict_proba(X)
        if hasattr(preds, "get"):
            return preds.get()
        return np.asarray(preds)
