"""K-fold cross-validation and OOF prediction collection on GPU."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cudf
import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result from a single fold training."""

    fold_idx: int
    train_idx: cp.ndarray
    val_idx: cp.ndarray
    oof_predictions: np.ndarray  # predictions on validation fold
    metric_value: float
    peak_vram_gb: float = 0.0


class CrossValidator:
    """GPU-native K-fold cross-validation with OOF collection.

    Performs stratified K-fold splitting on GPU using cuDF/CuPy,
    trains models on each fold, and collects Out-of-Fold predictions
    for stacking (Level-One Data).
    """

    def __init__(self, nfolds: int = 5, seed: int = 42):
        self.nfolds = nfolds
        self.seed = seed

    def create_folds(
        self, y: cudf.Series, stratified: bool = True
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        """Create K-fold indices on GPU.

        Parameters
        ----------
        y : cudf.Series
            Target variable for stratification.
        stratified : bool
            Whether to stratify by target value.

        Returns
        -------
        list of (train_idx, val_idx) CuPy arrays
        """
        n = len(y)
        rng = np.random.RandomState(self.seed)

        if stratified and y.nunique() <= 50:
            return self._stratified_split(y, n, rng)
        return self._simple_split(n, rng)

    def _simple_split(
        self, n: int, rng: np.random.RandomState
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        indices = rng.permutation(n)
        fold_sizes = np.full(self.nfolds, n // self.nfolds, dtype=int)
        fold_sizes[: n % self.nfolds] += 1

        folds = []
        current = 0
        for size in fold_sizes:
            val_idx = indices[current : current + size]
            train_idx = np.concatenate([indices[:current], indices[current + size :]])
            folds.append((cp.asarray(train_idx), cp.asarray(val_idx)))
            current += size
        return folds

    def _stratified_split(
        self, y: cudf.Series, n: int, rng: np.random.RandomState
    ) -> list[tuple[cp.ndarray, cp.ndarray]]:
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else cp.asnumpy(y.values)
        classes = np.unique(y_np)

        fold_indices: list[list[int]] = [[] for _ in range(self.nfolds)]

        for cls in classes:
            cls_idx = np.where(y_np == cls)[0]
            rng.shuffle(cls_idx)
            fold_sizes = np.full(self.nfolds, len(cls_idx) // self.nfolds, dtype=int)
            fold_sizes[: len(cls_idx) % self.nfolds] += 1

            current = 0
            for fold_i, size in enumerate(fold_sizes):
                fold_indices[fold_i].extend(cls_idx[current : current + size].tolist())
                current += size

        folds = []
        all_indices = np.arange(n)
        for fold_i in range(self.nfolds):
            val_idx = np.array(fold_indices[fold_i])
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_indices[train_mask]
            folds.append((cp.asarray(train_idx), cp.asarray(val_idx)))

        return folds

    def cross_validate(
        self,
        model_factory: Any,
        X: cudf.DataFrame,
        y: cudf.Series,
        task: str,
        metric_fn: Any,
        profiler: Any = None,
        streaming_oof: bool = False,
    ) -> tuple[list[Any], np.ndarray, float]:
        """Run K-fold CV and collect OOF predictions.

        Parameters
        ----------
        model_factory : callable
            Function that returns a fresh BaseModel instance for each fold.
        X : cudf.DataFrame
            Feature matrix.
        y : cudf.Series
            Target vector.
        task : str
            "classification" or "regression".
        metric_fn : callable
            Metric function(y_true, y_pred) -> float.
        profiler : MemoryProfiler, optional
            For VRAM tracking.
        streaming_oof : bool
            If True, move OOF predictions to host immediately to save VRAM.

        Returns
        -------
        (fold_models, oof_predictions, mean_metric)
        """
        folds = self.create_folds(y, stratified=(task == "classification"))
        n = len(y)
        is_classification = task == "classification"

        # Determine OOF shape
        n_classes = int(y.nunique()) if is_classification else 0
        if is_classification and n_classes > 2:
            oof = np.zeros((n, n_classes), dtype=np.float32)
        else:
            oof = np.zeros(n, dtype=np.float32)

        fold_models = []
        fold_metrics = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"  Fold {fold_i + 1}/{self.nfolds}")

            # Index into GPU dataframes
            train_idx_np = cp.asnumpy(train_idx)
            val_idx_np = cp.asnumpy(val_idx)

            X_train = X.iloc[train_idx_np]
            y_train = y.iloc[train_idx_np]
            X_val = X.iloc[val_idx_np]
            y_val = y.iloc[val_idx_np]

            model = model_factory()
            model.fit(X_train, y_train, X_val, y_val, profiler=profiler)

            # Collect OOF predictions
            if is_classification and n_classes > 2:
                preds = model.predict_proba(X_val)
                metric_preds = preds
            elif is_classification:
                preds = model.predict_proba(X_val)
                if preds.ndim == 2:
                    preds = preds[:, 1]
                metric_preds = preds
            else:
                preds = model.predict(X_val)
                metric_preds = preds

            # Streaming OOF: ensure predictions are on host (numpy) immediately
            # to free GPU memory from intermediate cupy/cudf prediction buffers.
            if streaming_oof:
                if hasattr(preds, "get"):  # cupy array on GPU
                    preds = preds.get()
                preds = np.asarray(preds, dtype=np.float32)
                metric_preds = preds

            oof[val_idx_np] = preds

            y_val_np = y_val.to_numpy() if hasattr(y_val, "to_numpy") else cp.asnumpy(y_val.values)
            fold_metric = metric_fn(y_val_np, metric_preds)
            fold_metrics.append(fold_metric)
            fold_models.append(model)

            # Streaming OOF: explicitly free fold data from GPU
            if streaming_oof:
                del X_train, y_train, X_val, y_val
                cp.get_default_memory_pool().free_all_blocks()

            logger.info(f"    Fold {fold_i + 1} metric: {fold_metric:.6f}")

        mean_metric = float(np.mean(fold_metrics))
        logger.info(f"  Mean CV metric: {mean_metric:.6f}")

        return fold_models, oof, mean_metric
