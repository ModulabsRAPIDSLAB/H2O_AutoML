"""GPUAutoML — Main entry point (sklearn-compatible API).

Usage:
    from gpu_automl import GPUAutoML

    automl = GPUAutoML(max_runtime_secs=300, memory_aware=True)
    automl.fit(X_train, y_train)
    lb = automl.leaderboard()
    preds = automl.predict(X_test)
    profile = automl.memory_profile()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import cudf
import numpy as np
import pandas as pd

from gpu_automl.data.loader import load_data
from gpu_automl.data.preprocessor import Preprocessor
from gpu_automl.memory.pool import PoolStrategy, RMMPoolManager
from gpu_automl.memory.profiler import MemoryProfiler
from gpu_automl.orchestrator import Orchestrator
from gpu_automl.reporting.leaderboard import Leaderboard
from gpu_automl.reporting.memory_report import MemoryReport

logger = logging.getLogger(__name__)


class GPUAutoML:
    """Memory-Aware GPU AutoML with H2O-style Stacked Ensemble.

    Parameters
    ----------
    max_runtime_secs : int
        Maximum runtime in seconds. 0 = no limit.
    max_models : int
        Maximum number of base models to train.
    nfolds : int
        Number of cross-validation folds.
    seed : int
        Random seed for reproducibility.
    memory_aware : bool
        Enable Memory-Aware Scheduling (our core contribution).
    memory_profile : bool
        Enable VRAM profiling (pynvml-based).
    pool_strategy : str
        rmm pool strategy: "none", "fixed", "managed", "adaptive".
    algorithms : list[str], optional
        Algorithms to include. Default: ["xgboost", "rf", "glm"].
    training_strategy : str
        "baseline" (fast), "diversity" (moderate), "full" (baseline + diversity + random).
    preprocess : bool
        Auto-preprocess data (missing values, categorical encoding).
    task : str, optional
        "classification", "regression", or None (auto-detect).
    use_dask : bool
        Enable Dask-CUDA multi-GPU cluster. Auto-detected based on GPU count.
    """

    def __init__(
        self,
        max_runtime_secs: int = 300,
        max_models: int = 20,
        nfolds: int = 5,
        seed: int = 42,
        memory_aware: bool = True,
        memory_profile: bool = True,
        pool_strategy: str = "adaptive",
        algorithms: Optional[list[str]] = None,
        training_strategy: str = "full",
        preprocess: bool = True,
        task: Optional[str] = None,
        use_dask: bool = False,
    ):
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.nfolds = nfolds
        self.seed = seed
        self.memory_aware = memory_aware
        self.memory_profile = memory_profile
        self.pool_strategy = pool_strategy
        self.algorithms = algorithms or ["xgboost", "rf", "glm"]
        self.training_strategy = training_strategy
        self.preprocess = preprocess
        self._task = task
        self.use_dask = use_dask

        self._profiler: Optional[MemoryProfiler] = None
        self._pool_manager: Optional[RMMPoolManager] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._preprocessor: Optional[Preprocessor] = None
        self._leaderboard: Optional[Leaderboard] = None
        self._dask_cluster: Any = None
        self._dask_client: Any = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[cudf.DataFrame, pd.DataFrame, str, Path],
        y: Union[cudf.Series, pd.Series, str, None] = None,
        target_col: Optional[str] = None,
    ) -> "GPUAutoML":
        """Train the AutoML pipeline.

        Parameters
        ----------
        X : cudf.DataFrame, pd.DataFrame, str, or Path
            Training features. If str/Path, loads from file.
        y : cudf.Series, pd.Series, str, or None
            Target variable. If None, uses target_col to split from X.
        target_col : str, optional
            Column name for target (when X contains both features and target).
        """
        # Step 1: Load data if needed
        X, y = self._prepare_data(X, y, target_col)

        # Step 2: Auto-detect task
        task = self._detect_task(y)
        logger.info(f"Task: {task}")

        # Step 3: Initialize GPU resources
        self._init_gpu_resources()

        # Step 4: Preprocess
        if self.preprocess:
            if self._profiler:
                self._profiler.begin_stage("preprocessing")
            self._preprocessor = Preprocessor()
            X = self._preprocessor.fit_transform(X)
            if self._profiler:
                self._profiler.end_stage()

        # Step 5: Run orchestrator
        self._orchestrator = Orchestrator(
            task=task,
            max_runtime_secs=self.max_runtime_secs,
            max_models=self.max_models,
            nfolds=self.nfolds,
            seed=self.seed,
            memory_aware=self.memory_aware,
            algorithms=self.algorithms,
            training_strategy=self.training_strategy,
            profiler=self._profiler,
            dask_client=self._dask_client,
        )

        self._leaderboard = self._orchestrator.run(X, y)
        self._is_fitted = True

        logger.info(f"AutoML complete. {len(self._leaderboard)} models trained.")
        return self

    def shutdown(self) -> None:
        """Clean up GPU resources (Dask cluster, profiler, rmm pool)."""
        if self._dask_client is not None:
            self._dask_client.close()
            self._dask_client = None
        if self._dask_cluster is not None:
            self._dask_cluster.close()
            self._dask_cluster = None
        if self._profiler is not None:
            self._profiler.shutdown()
        if self._pool_manager is not None:
            self._pool_manager.shutdown()

    def predict(
        self,
        X: Union[cudf.DataFrame, pd.DataFrame],
        model_id: Optional[str] = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : cudf.DataFrame or pd.DataFrame
            Test features.
        model_id : str, optional
            Specific model to use. Default: leaderboard #1 model.
        """
        if not self._is_fitted:
            raise RuntimeError("AutoML is not fitted. Call fit() first.")

        X = self._ensure_cudf(X)
        if self._preprocessor is not None:
            X = self._preprocessor.transform(X)

        if model_id is None:
            # Use best model (leaderboard #1)
            best = self._leaderboard.best
            if best is None:
                raise RuntimeError("No models in leaderboard.")
            model_id = best.model_id

        # Check if it's an ensemble
        if model_id.startswith("SE_"):
            stacker = self._orchestrator.get_stacker()
            ensemble_type = (
                "all_models" if model_id == "SE_AllModels" else "best_of_family"
            )
            return stacker.predict(X, ensemble_type=ensemble_type)

        # Single model: average fold predictions
        fold_models = self._orchestrator.get_fold_models().get(model_id)
        if fold_models is None:
            raise ValueError(f"Model '{model_id}' not found.")

        preds_list = []
        for fm in fold_models:
            if self._task == "classification":
                p = fm.predict_proba(X)
                if p.ndim == 2:
                    p = p[:, 1]
            else:
                p = fm.predict(X)
            preds_list.append(p)

        return np.mean(preds_list, axis=0)

    def leaderboard(self) -> pd.DataFrame:
        """Get the model leaderboard as a pandas DataFrame."""
        if self._leaderboard is None:
            return pd.DataFrame()
        return self._leaderboard.to_dataframe()

    def get_memory_report(self) -> MemoryReport:
        """Get the memory profiling report."""
        return MemoryReport(
            profiler=self._profiler,
            leaderboard=self._leaderboard,
        )

    def _prepare_data(
        self,
        X: Union[cudf.DataFrame, pd.DataFrame, str, Path],
        y: Union[cudf.Series, pd.Series, str, None],
        target_col: Optional[str],
    ) -> tuple[cudf.DataFrame, cudf.Series]:
        """Convert inputs to cuDF and split if needed."""
        if isinstance(X, (str, Path)):
            if target_col:
                X, y = load_data(X, target_col=target_col)
            else:
                X = load_data(X)

        X = self._ensure_cudf(X)

        if target_col and y is None:
            y = X[target_col]
            X = X.drop(columns=[target_col])

        if y is None:
            raise ValueError("Target y must be provided (via y= or target_col=).")

        if isinstance(y, pd.Series):
            y = cudf.Series(y)

        return X, y

    def _detect_task(self, y: cudf.Series) -> str:
        """Auto-detect classification vs regression."""
        if self._task is not None:
            return self._task

        n_unique = int(y.nunique())
        if n_unique <= 20 or y.dtype in ("object", "str", "category", "bool"):
            self._task = "classification"
        else:
            self._task = "regression"

        return self._task

    def _init_gpu_resources(self) -> None:
        """Initialize memory profiler, rmm pool, and optionally Dask-CUDA cluster."""
        if self.memory_profile:
            self._profiler = MemoryProfiler(enable_rmm_logging=False)
            self._profiler.initialize()

        strategy = PoolStrategy(self.pool_strategy)
        if strategy != PoolStrategy.NONE:
            self._pool_manager = RMMPoolManager(strategy=strategy)
            self._pool_manager.initialize()

        # Dask-CUDA cluster for multi-GPU or parallel fold/model execution
        if self.use_dask:
            try:
                from dask_cuda import LocalCUDACluster
                from dask.distributed import Client

                self._dask_cluster = LocalCUDACluster(
                    rmm_pool_size=None,  # let rmm pool manager handle it
                    rmm_managed_memory=False,
                )
                self._dask_client = Client(self._dask_cluster)
                n_workers = len(self._dask_client.scheduler_info()["workers"])
                logger.info(
                    f"Dask-CUDA cluster initialized: {n_workers} GPU worker(s)"
                )
            except Exception as e:
                logger.warning(f"Dask-CUDA cluster init failed: {e}. Using single GPU.")
                self._dask_client = None
                self._dask_cluster = None

    @staticmethod
    def _ensure_cudf(
        df: Union[cudf.DataFrame, pd.DataFrame],
    ) -> cudf.DataFrame:
        if isinstance(df, pd.DataFrame):
            return cudf.DataFrame.from_pandas(df)
        return df

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_models = len(self._leaderboard) if self._leaderboard else 0
        return (
            f"GPUAutoML(task={self._task}, status={status}, "
            f"models={n_models}, memory_aware={self.memory_aware})"
        )
