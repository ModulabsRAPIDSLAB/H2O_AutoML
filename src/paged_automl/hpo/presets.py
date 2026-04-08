"""Pre-specified hyperparameter sets following H2O AutoML strategy.

H2O's approach:
  1. Baseline: default hyperparameters (fast, reasonable baseline)
  2. Diversity: pre-specified grids for each algorithm
  3. Random Search: random sampling from search space

This module provides the pre-specified grids for step 1 and 2.
"""

from __future__ import annotations

from typing import Any


def get_presets(
    algorithm: str,
    task: str = "classification",
    level: str = "baseline",
) -> list[dict[str, Any]]:
    """Get pre-specified hyperparameter configurations.

    Parameters
    ----------
    algorithm : str
        "xgboost", "rf", or "glm".
    task : str
        "classification" or "regression".
    level : str
        "baseline" (1 config), "diversity" (3-5 configs), or "all".

    Returns
    -------
    List of parameter dicts.
    """
    presets = _PRESETS.get(algorithm, {})
    baseline = presets.get("baseline", [{}])
    diversity = presets.get("diversity", [])

    if level == "baseline":
        return baseline
    elif level == "diversity":
        return diversity
    else:  # "all"
        return baseline + diversity


_PRESETS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "xgboost": {
        "baseline": [
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }
        ],
        "diversity": [
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 3,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.02,
                "subsample": 0.9,
                "colsample_bytree": 0.6,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
            {
                "n_estimators": 200,
                "max_depth": 10,
                "learning_rate": 0.1,
                "subsample": 0.6,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 0.1,
            },
        ],
    },
    "rf": {
        "baseline": [
            {
                "n_estimators": 100,
                "max_depth": 12,
                "max_features": "sqrt",
            }
        ],
        "diversity": [
            {
                "n_estimators": 200,
                "max_depth": 8,
                "max_features": 0.5,
            },
            {
                "n_estimators": 50,
                "max_depth": 16,
                "max_features": "sqrt",
            },
            {
                "n_estimators": 300,
                "max_depth": 10,
                "max_features": 0.3,
            },
        ],
    },
    "glm": {
        "baseline": [
            {
                "C": 1.0,
                "penalty": "l2",
                "max_iter": 1000,
            }
        ],
        "diversity": [
            {
                "C": 0.1,
                "penalty": "l1",
                "max_iter": 1000,
            },
            {
                "C": 10.0,
                "penalty": "l2",
                "max_iter": 2000,
            },
        ],
    },
}
