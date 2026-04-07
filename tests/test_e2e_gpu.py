"""End-to-end GPU AutoML integration test.

Generates synthetic data on GPU and runs the full pipeline:
  Data load -> Preprocess -> CV -> Base models -> Stacked Ensemble -> Predict

Usage:
    python -m tests.test_e2e_gpu
    python -m tests.test_e2e_gpu --rows 10000 --features 20 --models 5 --time 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cudf
import cupy as cp
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_e2e")


def generate_synthetic_data(
    n_rows: int = 10000,
    n_features: int = 20,
    task: str = "classification",
    seed: int = 42,
) -> tuple[cudf.DataFrame, cudf.Series, cudf.DataFrame, cudf.Series]:
    """Generate synthetic tabular data directly on GPU."""
    rng = cp.random.RandomState(seed)

    X_train_gpu = rng.randn(n_rows, n_features).astype(cp.float32)
    X_test_gpu = rng.randn(n_rows // 5, n_features).astype(cp.float32)

    # Create feature names
    col_names = [f"f{i}" for i in range(n_features)]

    X_train = cudf.DataFrame(X_train_gpu, columns=col_names)
    X_test = cudf.DataFrame(X_test_gpu, columns=col_names)

    if task == "classification":
        # Binary classification: logistic function of weighted sum
        weights = rng.randn(n_features).astype(cp.float32)
        logits_train = X_train_gpu @ weights
        probs_train = 1.0 / (1.0 + cp.exp(-logits_train))
        y_train = cudf.Series((probs_train > 0.5).astype(cp.int32))

        logits_test = X_test_gpu @ weights
        probs_test = 1.0 / (1.0 + cp.exp(-logits_test))
        y_test = cudf.Series((probs_test > 0.5).astype(cp.int32))
    else:
        weights = rng.randn(n_features).astype(cp.float32)
        y_train = cudf.Series(
            (X_train_gpu @ weights + rng.randn(n_rows).astype(cp.float32) * 0.1)
        )
        y_test = cudf.Series(
            (X_test_gpu @ weights + rng.randn(n_rows // 5).astype(cp.float32) * 0.1)
        )

    return X_train, y_train, X_test, y_test


def test_full_pipeline(
    n_rows: int = 10000,
    n_features: int = 20,
    max_models: int = 5,
    max_runtime_secs: int = 120,
    memory_aware: bool = True,
):
    """Run the full GPU AutoML pipeline as an integration test."""
    from gpu_automl import GPUAutoML

    print("=" * 60)
    print("GPU AutoML E2E Integration Test")
    print("=" * 60)

    # Step 1: Generate data
    print(f"\n[1/5] Generating synthetic data ({n_rows} x {n_features})...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_rows=n_rows, n_features=n_features
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target distribution: {y_train.value_counts().to_pandas().to_dict()}")

    # Step 2: Initialize AutoML
    print(f"\n[2/5] Initializing GPUAutoML (memory_aware={memory_aware})...")
    automl = GPUAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        nfolds=5,
        seed=42,
        memory_aware=memory_aware,
        memory_profile=True,
        pool_strategy="adaptive",
        training_strategy="full",
        preprocess=False,  # synthetic data is clean
    )
    print(f"  {automl}")

    # Step 3: Fit
    print(f"\n[3/5] Training (budget: {max_runtime_secs}s, max_models: {max_models})...")
    t0 = time.perf_counter()
    automl.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Step 4: Leaderboard
    print("\n[4/5] Leaderboard:")
    lb = automl.leaderboard()
    print(lb.to_string(index=False))

    # Step 5: Predict
    print("\n[5/5] Predictions:")
    preds = automl.predict(X_test)
    print(f"  Prediction shape: {preds.shape}")
    print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Mean prediction: {preds.mean():.4f}")

    # Memory profile
    profile = automl.get_memory_report()
    print(f"\n{profile}")

    # Validation checks
    print("\n" + "=" * 60)
    print("Validation Checks:")
    errors = []

    # Check leaderboard has models
    if len(lb) == 0:
        errors.append("Leaderboard is empty!")
    else:
        print(f"  [OK] Leaderboard has {len(lb)} models")

    # Check ensembles were created
    ensemble_count = lb["is_ensemble"].sum() if "is_ensemble" in lb.columns else 0
    if ensemble_count >= 1:
        print(f"  [OK] {ensemble_count} ensemble(s) created")
    else:
        errors.append("No ensembles created")

    # Check predictions shape
    if preds.shape[0] == X_test.shape[0]:
        print(f"  [OK] Prediction count matches test data ({preds.shape[0]})")
    else:
        errors.append(f"Prediction count mismatch: {preds.shape[0]} vs {X_test.shape[0]}")

    # Check predictions are valid probabilities for classification
    if preds.min() >= -0.1 and preds.max() <= 1.1:
        print("  [OK] Predictions in valid range")
    else:
        errors.append(f"Predictions out of range: [{preds.min()}, {preds.max()}]")

    # Check peak VRAM is recorded
    if "peak_vram_gb" in lb.columns and lb["peak_vram_gb"].sum() > 0:
        print(f"  [OK] VRAM profiling active (total peak: {lb['peak_vram_gb'].max():.3f} GB)")
    else:
        print("  [WARN] VRAM profiling shows 0 (may need pynvml)")

    if errors:
        print(f"\n  FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ALL CHECKS PASSED")

    # Cleanup
    automl.shutdown()
    return True


def test_memory_naive_vs_aware(
    n_rows: int = 50000,
    n_features: int = 20,
    max_models: int = 5,
):
    """Compare Memory-Naive vs Memory-Aware scheduling."""
    from gpu_automl import GPUAutoML

    print("\n" + "=" * 60)
    print("Memory-Naive vs Memory-Aware Comparison")
    print("=" * 60)

    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_rows=n_rows, n_features=n_features
    )

    results = {}
    for mode_name, memory_aware in [("Naive", False), ("Aware", True)]:
        print(f"\n--- {mode_name} Mode ---")
        automl = GPUAutoML(
            max_runtime_secs=120,
            max_models=max_models,
            memory_aware=memory_aware,
            memory_profile=True,
            training_strategy="baseline",
            preprocess=False,
        )

        t0 = time.perf_counter()
        automl.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        lb = automl.leaderboard()
        results[mode_name] = {
            "time": elapsed,
            "models": len(lb),
            "best_auc": lb["auc"].iloc[0] if len(lb) > 0 else 0,
        }
        print(f"  Time: {elapsed:.1f}s, Models: {len(lb)}")
        if len(lb) > 0:
            print(f"  Best AUC: {lb['auc'].iloc[0]:.6f}")

        automl.shutdown()

    print("\n--- Comparison ---")
    for mode, r in results.items():
        print(f"  {mode}: {r['time']:.1f}s, {r['models']} models, AUC={r['best_auc']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU AutoML E2E Test")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--compare", action="store_true", help="Run Naive vs Aware comparison")
    args = parser.parse_args()

    success = test_full_pipeline(
        n_rows=args.rows,
        n_features=args.features,
        max_models=args.models,
        max_runtime_secs=args.time,
    )

    if args.compare:
        test_memory_naive_vs_aware(
            n_rows=args.rows,
            n_features=args.features,
            max_models=args.models,
        )

    sys.exit(0 if success else 1)
