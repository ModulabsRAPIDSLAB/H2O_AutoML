"""VRAM usage estimator for Memory-Aware Scheduling.

Estimates expected VRAM consumption per model based on:
  - Number of rows and features in the dataset
  - Model algorithm and hyperparameters
  - Historical profiling data (Phase 2 data-driven regression)

Accuracy target: actual VRAM +/- 20% (FR-061).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VRAMEstimate:
    """Estimated VRAM usage for a model training task."""

    algorithm: str
    estimated_gb: float
    confidence: float  # 0.0 ~ 1.0
    breakdown: dict[str, float]  # component -> GB


# Heuristic coefficients (to be refined with Phase 2 profiling data)
# Format: base_gb + per_row_per_feature * n_rows * n_features
_HEURISTIC_COEFFICIENTS = {
    "xgboost": {
        "base_gb": 0.3,
        "bytes_per_element": 8,  # float64 hist bins
        "tree_overhead_factor": 1.5,
    },
    "rf": {
        "base_gb": 0.2,
        "bytes_per_element": 4,  # float32
        "tree_overhead_factor": 2.0,  # RF stores more per tree
    },
    "glm": {
        "base_gb": 0.1,
        "bytes_per_element": 8,
        "tree_overhead_factor": 1.0,  # no trees
    },
}


class VRAMEstimator:
    """Estimates VRAM usage for model training tasks.

    Phase 2 approach: heuristic formulas based on data size and algorithm.
    Phase 3 refinement: regression model trained on profiling data.

    Parameters
    ----------
    safety_margin : float
        Multiplier for safety (e.g., 1.2 = 20% extra margin).
    """

    def __init__(self, safety_margin: float = 1.2):
        self.safety_margin = safety_margin
        self._profile_data: list[dict] = []
        self._regression_model = None

    def estimate(
        self,
        algorithm: str,
        n_rows: int,
        n_features: int,
        params: Optional[dict] = None,
    ) -> VRAMEstimate:
        """Estimate VRAM usage for a training task.

        Parameters
        ----------
        algorithm : str
            "xgboost", "rf", or "glm".
        n_rows : int
            Number of training rows.
        n_features : int
            Number of features.
        params : dict, optional
            Hyperparameters (affect estimate for some algorithms).
        """
        if self._regression_model is not None:
            return self._estimate_regression(algorithm, n_rows, n_features, params)

        return self._estimate_heuristic(algorithm, n_rows, n_features, params)

    def _estimate_heuristic(
        self,
        algorithm: str,
        n_rows: int,
        n_features: int,
        params: Optional[dict] = None,
    ) -> VRAMEstimate:
        """Heuristic estimation based on data dimensions."""
        params = params or {}
        coefs = _HEURISTIC_COEFFICIENTS.get(algorithm, _HEURISTIC_COEFFICIENTS["glm"])

        # Data matrix size
        data_gb = (n_rows * n_features * coefs["bytes_per_element"]) / (1024**3)

        # Training overhead (internal data structures, gradients, etc.)
        train_overhead = data_gb * coefs["tree_overhead_factor"]

        # Algorithm-specific adjustments
        if algorithm == "xgboost":
            n_estimators = params.get("n_estimators", 100)
            max_depth = params.get("max_depth", 6)
            # XGBoost keeps histograms in memory
            hist_gb = (n_rows * n_features * 4) / (1024**3)  # float32 hists
            train_overhead += hist_gb * min(max_depth / 6, 2.0)
        elif algorithm == "rf":
            n_estimators = params.get("n_estimators", 100)
            # RF builds trees in parallel on GPU
            train_overhead *= min(n_estimators / 100, 3.0)

        total = (coefs["base_gb"] + data_gb + train_overhead) * self.safety_margin

        return VRAMEstimate(
            algorithm=algorithm,
            estimated_gb=total,
            confidence=0.6,  # heuristic = lower confidence
            breakdown={
                "base": coefs["base_gb"],
                "data": data_gb,
                "training_overhead": train_overhead,
                "safety_margin": total - (coefs["base_gb"] + data_gb + train_overhead),
            },
        )

    def _estimate_regression(
        self,
        algorithm: str,
        n_rows: int,
        n_features: int,
        params: Optional[dict] = None,
    ) -> VRAMEstimate:
        """Data-driven estimation using regression model (Phase 3)."""
        # Feature vector: [n_rows, n_features, algo_onehot, key_params]
        algo_map = {"xgboost": 0, "rf": 1, "glm": 2}
        algo_idx = algo_map.get(algorithm, 2)
        algo_onehot = [0, 0, 0]
        algo_onehot[algo_idx] = 1

        params = params or {}
        features = np.array(
            [
                n_rows,
                n_features,
                *algo_onehot,
                params.get("n_estimators", 100),
                params.get("max_depth", 6),
            ],
            dtype=np.float64,
        ).reshape(1, -1)

        estimated = float(self._regression_model.predict(features)[0])
        estimated = max(estimated, 0.1) * self.safety_margin

        return VRAMEstimate(
            algorithm=algorithm,
            estimated_gb=estimated,
            confidence=0.85,  # regression = higher confidence
            breakdown={"regression_estimate": estimated / self.safety_margin},
        )

    def record_actual(
        self,
        algorithm: str,
        n_rows: int,
        n_features: int,
        params: dict,
        actual_vram_gb: float,
    ) -> None:
        """Record actual VRAM usage for future regression model training."""
        self._profile_data.append(
            {
                "algorithm": algorithm,
                "n_rows": n_rows,
                "n_features": n_features,
                "params": params,
                "actual_vram_gb": actual_vram_gb,
            }
        )

    def fit_regression(self) -> None:
        """Train regression model on collected profiling data (Phase 3)."""
        if len(self._profile_data) < 10:
            logger.warning(
                f"Only {len(self._profile_data)} profile samples. "
                "Need >= 10 for regression. Using heuristics."
            )
            return

        from sklearn.ensemble import GradientBoostingRegressor

        algo_map = {"xgboost": 0, "rf": 1, "glm": 2}
        X_list, y_list = [], []

        for d in self._profile_data:
            algo_idx = algo_map.get(d["algorithm"], 2)
            algo_onehot = [0, 0, 0]
            algo_onehot[algo_idx] = 1
            features = [
                d["n_rows"],
                d["n_features"],
                *algo_onehot,
                d["params"].get("n_estimators", 100),
                d["params"].get("max_depth", 6),
            ]
            X_list.append(features)
            y_list.append(d["actual_vram_gb"])

        X = np.array(X_list)
        y = np.array(y_list)

        self._regression_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, random_state=42
        )
        self._regression_model.fit(X, y)
        logger.info(
            f"VRAM regression model trained on {len(self._profile_data)} samples."
        )
