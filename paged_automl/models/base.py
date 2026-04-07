"""Base model interface for all GPU AutoML models."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cudf
import numpy as np


@dataclass
class TrainResult:
    """Training result containing model metadata."""

    model_id: str
    algorithm: str
    params: dict[str, Any]
    metric_name: str
    metric_value: float
    training_time_secs: float
    peak_vram_gb: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for a model instance."""

    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    family: str = ""  # e.g., "xgboost", "rf", "glm"


class BaseModel(ABC):
    """Abstract base class for all GPU models.

    All model wrappers must implement fit() and predict().
    Provides common functionality: timing, VRAM tracking, seed management.
    """

    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self._model: Any = None
        self._is_fitted = False
        self._training_time: float = 0.0
        self._peak_vram_gb: float = 0.0

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def algorithm(self) -> str:
        return self.config.algorithm

    @property
    def family(self) -> str:
        return self.config.family

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def _build_model(self) -> Any:
        """Build and return the underlying model object."""

    @abstractmethod
    def _fit_impl(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
        X_val: Optional[cudf.DataFrame] = None,
        y_val: Optional[cudf.Series] = None,
    ) -> None:
        """Internal fit implementation."""

    @abstractmethod
    def _predict_impl(self, X: cudf.DataFrame) -> np.ndarray:
        """Internal predict implementation. Returns numpy array."""

    @abstractmethod
    def _predict_proba_impl(self, X: cudf.DataFrame) -> np.ndarray:
        """Internal predict_proba implementation. Returns numpy array."""

    def fit(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
        X_val: Optional[cudf.DataFrame] = None,
        y_val: Optional[cudf.Series] = None,
        profiler: Any = None,
    ) -> TrainResult:
        """Train the model with optional validation set and memory profiling."""
        start = time.perf_counter()
        vram_before = 0.0
        if profiler is not None:
            vram_before = profiler.get_current_vram_gb()

        self._model = self._build_model()
        self._fit_impl(X, y, X_val, y_val)
        self._is_fitted = True

        self._training_time = time.perf_counter() - start
        if profiler is not None:
            self._peak_vram_gb = profiler.get_peak_vram_gb_since(vram_before)

        return TrainResult(
            model_id=self.model_id,
            algorithm=self.algorithm,
            params=self.config.params,
            metric_name="",
            metric_value=0.0,
            training_time_secs=self._training_time,
            peak_vram_gb=self._peak_vram_gb,
        )

    def predict(self, X: cudf.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.model_id} is not fitted.")
        return self._predict_impl(X)

    def predict_proba(self, X: cudf.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.model_id} is not fitted.")
        return self._predict_proba_impl(X)

    def get_params(self) -> dict[str, Any]:
        return self.config.params.copy()
