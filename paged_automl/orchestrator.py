"""Training orchestrator implementing H2O AutoML's training strategy.

H2O Training Order:
  Phase A — Baseline: one default model per algorithm
  Phase B — Diversity: pre-specified hyperparameter grids
  Phase C — Random Search: random HP sampling until time budget exhausted

Time control: max_runtime_secs enforced across all phases.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import cudf
import numpy as np

from paged_automl.data.cv import CrossValidator
from paged_automl.ensemble.stacking import StackedEnsemble
from paged_automl.hpo.presets import get_presets
from paged_automl.hpo.random_search import RandomSearch
from paged_automl.memory.estimator import VRAMEstimator
from paged_automl.memory.profiler import MemoryProfiler
from paged_automl.models.base import BaseModel, ModelConfig
from paged_automl.models.cuml_glm import CuMLGLM
from paged_automl.models.cuml_rf import CuMLRandomForest
from paged_automl.models.xgboost_gpu import XGBoostGPU
from paged_automl.reporting.leaderboard import Leaderboard, LeaderboardEntry
from paged_automl.scheduler import (
    MemoryAwareScheduler,
    SchedulingMode,
    TrainTask,
)

logger = logging.getLogger(__name__)

# Algorithm registry
_ALGORITHM_REGISTRY: dict[str, tuple[type, str]] = {
    "xgboost": (XGBoostGPU, "xgboost"),
    "rf": (CuMLRandomForest, "rf"),
    "glm": (CuMLGLM, "glm"),
}


def _default_metric_fn(task: str):
    """Return default metric function based on task type."""
    if task == "classification":
        from sklearn.metrics import roc_auc_score

        return roc_auc_score, "auc", True
    else:
        from sklearn.metrics import mean_squared_error

        def rmse(y_true, y_pred):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        return rmse, "rmse", False


class Orchestrator:
    """Training pipeline orchestrator.

    Manages the H2O-style training order, CV, ensembling, and time control.

    Parameters
    ----------
    task : str
        "classification" or "regression".
    max_runtime_secs : int
        Time budget in seconds.
    max_models : int
        Maximum number of base models.
    nfolds : int
        Number of CV folds.
    seed : int
        Random seed.
    memory_aware : bool
        Enable Memory-Aware Scheduling.
    algorithms : list[str], optional
        Algorithms to use. Default: all available.
    training_strategy : str
        "baseline", "diversity", or "full" (baseline + diversity + random).
    """

    def __init__(
        self,
        task: str = "classification",
        max_runtime_secs: int = 300,
        max_models: int = 20,
        nfolds: int = 5,
        seed: int = 42,
        memory_aware: bool = True,
        algorithms: Optional[list[str]] = None,
        training_strategy: str = "full",
        profiler: Optional[MemoryProfiler] = None,
        dask_client: Optional[Any] = None,
    ):
        self.task = task
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.nfolds = nfolds
        self.seed = seed
        self.memory_aware = memory_aware
        self.algorithms = algorithms or list(_ALGORITHM_REGISTRY.keys())
        self.training_strategy = training_strategy
        self.profiler = profiler
        self.dask_client = dask_client

        self._metric_fn, self._metric_name, self._higher_is_better = (
            _default_metric_fn(task)
        )
        self._cv = CrossValidator(nfolds=nfolds, seed=seed)
        self._leaderboard = Leaderboard(
            metric_name=self._metric_name,
            higher_is_better=self._higher_is_better,
        )
        self._stacker = StackedEnsemble(task=task, seed=seed)

        # Memory-Aware Scheduler
        self._estimator = VRAMEstimator()
        scheduling_mode = (
            SchedulingMode.AWARE if memory_aware else SchedulingMode.NAIVE
        )
        self._scheduler = MemoryAwareScheduler(
            profiler=profiler or MemoryProfiler(),
            estimator=self._estimator,
            mode=scheduling_mode,
        )

        # State
        self._oof_predictions: dict[str, np.ndarray] = {}
        self._fold_models: dict[str, list[Any]] = {}
        self._model_families: dict[str, str] = {}
        self._model_metrics: dict[str, float] = {}
        self._model_counter = 0
        self._start_time: float = 0.0
        self._n_rows: int = 0
        self._n_features: int = 0

    @property
    def leaderboard(self) -> Leaderboard:
        return self._leaderboard

    def _time_remaining(self) -> float:
        if self.max_runtime_secs <= 0:
            return float("inf")
        return max(0, self.max_runtime_secs - (time.perf_counter() - self._start_time))

    def _is_time_up(self) -> bool:
        return self._time_remaining() <= 0

    def _budget_exceeded(self) -> bool:
        return self._model_counter >= self.max_models or self._is_time_up()

    def _next_model_id(self, algorithm: str) -> str:
        self._model_counter += 1
        return f"{algorithm}_{self._model_counter}"

    def run(
        self,
        X: cudf.DataFrame,
        y: cudf.Series,
    ) -> Leaderboard:
        """Execute the full training pipeline.

        Returns the final leaderboard with all models and ensembles.
        """
        self._start_time = time.perf_counter()
        self._n_rows, self._n_features = X.shape
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else y.values.get()

        logger.info(
            f"Orchestrator starting: task={self.task}, "
            f"data=({self._n_rows} x {self._n_features}), "
            f"budget={self.max_runtime_secs}s, "
            f"max_models={self.max_models}, "
            f"memory_aware={self.memory_aware}"
        )

        # Phase A: Baseline models
        if self.profiler:
            self.profiler.begin_stage("baseline")
        self._run_baseline_phase(X, y, y_np)
        if self.profiler:
            self.profiler.end_stage()

        # Phase B: Diversity models
        if not self._budget_exceeded() and self.training_strategy in (
            "diversity",
            "full",
        ):
            if self.profiler:
                self.profiler.begin_stage("diversity")
            self._run_diversity_phase(X, y, y_np)
            if self.profiler:
                self.profiler.end_stage()

        # Phase C: Random Search
        if not self._budget_exceeded() and self.training_strategy == "full":
            if self.profiler:
                self.profiler.begin_stage("random_search")
            self._run_random_search_phase(X, y, y_np)
            if self.profiler:
                self.profiler.end_stage()

        # Stacked Ensembles
        if len(self._oof_predictions) >= 2:
            if self.profiler:
                self.profiler.begin_stage("stacking")
            self._build_ensembles(y_np)
            if self.profiler:
                self.profiler.end_stage()

        elapsed = time.perf_counter() - self._start_time
        logger.info(
            f"Orchestrator done: {len(self._leaderboard)} models, "
            f"{elapsed:.1f}s elapsed"
        )
        return self._leaderboard

    def _run_baseline_phase(
        self, X: cudf.DataFrame, y: cudf.Series, y_np: np.ndarray
    ) -> None:
        """Phase A: Train one default model per algorithm."""
        logger.info("=== Phase A: Baseline ===")
        for algo in self.algorithms:
            if self._budget_exceeded():
                break
            configs = get_presets(algo, self.task, level="baseline")
            for params in configs:
                if self._budget_exceeded():
                    break
                self._train_model(algo, params, X, y, y_np)

    def _run_diversity_phase(
        self, X: cudf.DataFrame, y: cudf.Series, y_np: np.ndarray
    ) -> None:
        """Phase B: Train models with pre-specified diverse hyperparameters."""
        logger.info("=== Phase B: Diversity ===")
        for algo in self.algorithms:
            if self._budget_exceeded():
                break
            configs = get_presets(algo, self.task, level="diversity")
            for params in configs:
                if self._budget_exceeded():
                    break
                self._train_model(algo, params, X, y, y_np)

    def _run_random_search_phase(
        self, X: cudf.DataFrame, y: cudf.Series, y_np: np.ndarray
    ) -> None:
        """Phase C: Random Search until time/model budget exhausted."""
        logger.info("=== Phase C: Random Search ===")
        rs = RandomSearch(seed=self.seed)

        while not self._budget_exceeded():
            # Cycle through algorithms
            for algo in self.algorithms:
                if self._budget_exceeded():
                    break
                configs = rs.sample(algo, n_configs=1)
                for params in configs:
                    if self._budget_exceeded():
                        break
                    self._train_model(algo, params, X, y, y_np)

    def _train_model(
        self,
        algorithm: str,
        params: dict,
        X: cudf.DataFrame,
        y: cudf.Series,
        y_np: np.ndarray,
    ) -> Optional[str]:
        """Train a single model with CV and collect OOF predictions.

        Uses Memory-Aware Scheduler to check VRAM before training.
        In AWARE mode, skips the model if estimated VRAM exceeds available.
        In NAIVE mode, trains immediately without VRAM checks.
        """
        model_id = self._next_model_id(algorithm)
        model_cls, family = _ALGORITHM_REGISTRY[algorithm]

        # Memory-Aware: estimate VRAM and check availability
        if self.memory_aware and self.profiler and self.profiler._initialized:
            estimate = self._estimator.estimate(
                algorithm=algorithm,
                n_rows=self._n_rows,
                n_features=self._n_features,
                params=params,
            )
            free_vram = self.profiler.get_free_vram_gb()
            if free_vram < estimate.estimated_gb:
                logger.warning(
                    f"Skipping {model_id}: estimated VRAM {estimate.estimated_gb:.2f} GB "
                    f"> free {free_vram:.2f} GB"
                )
                # Revert model counter so we don't waste IDs
                self._model_counter -= 1
                return None
            logger.info(
                f"VRAM check passed for {model_id}: "
                f"estimated {estimate.estimated_gb:.2f} GB, "
                f"free {free_vram:.2f} GB"
            )

        config = ModelConfig(
            algorithm=algorithm,
            params=params,
            model_id=model_id,
            family=family,
        )

        def model_factory():
            return model_cls(config=config, seed=self.seed, task=self.task)

        logger.info(f"Training {model_id} ({algorithm})...")

        try:
            fold_models, oof, mean_metric = self._cv.cross_validate(
                model_factory=model_factory,
                X=X,
                y=y,
                task=self.task,
                metric_fn=self._metric_fn,
                profiler=self.profiler,
                streaming_oof=(self.memory_aware),
            )
        except Exception as e:
            logger.error(f"Model {model_id} failed: {e}")
            return None

        # Record actual VRAM for estimator refinement
        peak_vram = max(
            (fm._peak_vram_gb for fm in fold_models), default=0.0
        )
        if peak_vram > 0:
            self._estimator.record_actual(
                algorithm=algorithm,
                n_rows=self._n_rows,
                n_features=self._n_features,
                params=params,
                actual_vram_gb=peak_vram,
            )

        # Store results
        self._oof_predictions[model_id] = oof
        self._fold_models[model_id] = fold_models
        self._model_families[model_id] = family
        self._model_metrics[model_id] = mean_metric

        # Update leaderboard
        training_time = sum(
            fm._training_time for fm in fold_models
        ) / len(fold_models)

        self._leaderboard.add(
            LeaderboardEntry(
                model_id=model_id,
                algorithm=algorithm,
                family=family,
                metric_name=self._metric_name,
                metric_value=mean_metric,
                training_time_secs=training_time,
                peak_vram_gb=peak_vram,
                params=params,
            )
        )

        logger.info(f"  {model_id}: {self._metric_name}={mean_metric:.6f}")
        return model_id

    def _build_ensembles(self, y_np: np.ndarray) -> None:
        """Build stacked ensembles (All Models + Best of Family)."""
        logger.info("=== Building Stacked Ensembles ===")

        # All Models Ensemble
        try:
            result_all = self._stacker.build_all_models_ensemble(
                oof_predictions=self._oof_predictions,
                y=y_np,
                fold_models=self._fold_models,
                metric_fn=self._metric_fn,
            )
            self._leaderboard.add(
                LeaderboardEntry(
                    model_id=result_all.ensemble_id,
                    algorithm="StackedEnsemble",
                    family="ensemble",
                    metric_name=self._metric_name,
                    metric_value=result_all.metric_value,
                    training_time_secs=0.0,
                    is_ensemble=True,
                )
            )
            logger.info(
                f"  All Models Ensemble: "
                f"{self._metric_name}={result_all.metric_value:.6f}"
            )
        except Exception as e:
            logger.error(f"All Models Ensemble failed: {e}")

        # Best of Family Ensemble
        try:
            result_bof = self._stacker.build_best_of_family_ensemble(
                oof_predictions=self._oof_predictions,
                y=y_np,
                model_families=self._model_families,
                model_metrics=self._model_metrics,
                fold_models=self._fold_models,
                metric_fn=self._metric_fn,
                higher_is_better=self._higher_is_better,
            )
            self._leaderboard.add(
                LeaderboardEntry(
                    model_id=result_bof.ensemble_id,
                    algorithm="StackedEnsemble",
                    family="ensemble",
                    metric_name=self._metric_name,
                    metric_value=result_bof.metric_value,
                    training_time_secs=0.0,
                    is_ensemble=True,
                )
            )
            logger.info(
                f"  Best of Family Ensemble: "
                f"{self._metric_name}={result_bof.metric_value:.6f}"
            )
        except Exception as e:
            logger.error(f"Best of Family Ensemble failed: {e}")

    def get_stacker(self) -> StackedEnsemble:
        return self._stacker

    def get_oof_predictions(self) -> dict[str, np.ndarray]:
        return dict(self._oof_predictions)

    def get_fold_models(self) -> dict[str, list[Any]]:
        return dict(self._fold_models)
