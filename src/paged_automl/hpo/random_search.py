"""Random Search hyperparameter optimization.

Implements Bergstra & Bengio (2012) random search strategy,
which is more efficient than grid search for most hyperparameter
optimization problems.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Search spaces per algorithm
_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": ("int_log", 50, 1000),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_uniform", 0.005, 0.3),
        "subsample": ("uniform", 0.5, 1.0),
        "colsample_bytree": ("uniform", 0.3, 1.0),
        "min_child_weight": ("int", 1, 10),
        "reg_alpha": ("log_uniform", 1e-4, 10.0),
        "reg_lambda": ("log_uniform", 1e-4, 10.0),
    },
    "rf": {
        "n_estimators": ("int_log", 50, 500),
        "max_depth": ("int", 4, 16),
        "max_features": ("uniform", 0.1, 1.0),
    },
    "glm": {
        "C": ("log_uniform", 0.001, 100.0),
        "max_iter": ("int", 500, 5000),
    },
}


class RandomSearch:
    """Random Search hyperparameter sampler.

    Samples random configurations from per-algorithm search spaces.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        algorithm: str,
        n_configs: int = 10,
        search_space: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Sample random hyperparameter configurations.

        Parameters
        ----------
        algorithm : str
            Algorithm name ("xgboost", "rf", "glm").
        n_configs : int
            Number of configurations to sample.
        search_space : dict, optional
            Custom search space. If None, uses defaults.

        Returns
        -------
        List of parameter dicts.
        """
        space = search_space or _SEARCH_SPACES.get(algorithm, {})
        if not space:
            logger.warning(f"No search space for {algorithm}. Returning empty.")
            return [{}] * n_configs

        configs = []
        for _ in range(n_configs):
            config = {}
            for param, spec in space.items():
                config[param] = self._sample_param(spec)
            configs.append(config)

        return configs

    def _sample_param(self, spec: tuple) -> Any:
        dist_type = spec[0]

        if dist_type == "uniform":
            low, high = spec[1], spec[2]
            return float(self.rng.uniform(low, high))

        elif dist_type == "log_uniform":
            low, high = np.log(spec[1]), np.log(spec[2])
            return float(np.exp(self.rng.uniform(low, high)))

        elif dist_type == "int":
            low, high = spec[1], spec[2]
            return int(self.rng.randint(low, high + 1))

        elif dist_type == "int_log":
            low, high = np.log(spec[1]), np.log(spec[2])
            return int(np.exp(self.rng.uniform(low, high)))

        elif dist_type == "choice":
            return self.rng.choice(spec[1])

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
