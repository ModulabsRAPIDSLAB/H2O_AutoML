"""Leaderboard for tracking model performance and VRAM usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""

    model_id: str
    algorithm: str
    family: str
    metric_name: str
    metric_value: float
    training_time_secs: float
    peak_vram_gb: float = 0.0
    n_params: int = 0
    is_ensemble: bool = False
    params: dict = field(default_factory=dict)


class Leaderboard:
    """Model leaderboard with performance and VRAM metrics.

    Tracks all trained models, sorted by primary metric.
    Includes peak VRAM usage per model for memory analysis.
    """

    def __init__(self, metric_name: str = "auc", higher_is_better: bool = True):
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self._entries: list[LeaderboardEntry] = []

    def add(self, entry: LeaderboardEntry) -> None:
        self._entries.append(entry)
        self._sort()

    def _sort(self) -> None:
        self._entries.sort(
            key=lambda e: e.metric_value, reverse=self.higher_is_better
        )

    @property
    def best(self) -> Optional[LeaderboardEntry]:
        return self._entries[0] if self._entries else None

    @property
    def entries(self) -> list[LeaderboardEntry]:
        return list(self._entries)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert leaderboard to pandas DataFrame for display."""
        if not self._entries:
            return pd.DataFrame()

        rows = []
        for rank, e in enumerate(self._entries, 1):
            rows.append(
                {
                    "rank": rank,
                    "model_id": e.model_id,
                    "algorithm": e.algorithm,
                    self.metric_name: round(e.metric_value, 6),
                    "training_time_secs": round(e.training_time_secs, 2),
                    "peak_vram_gb": round(e.peak_vram_gb, 3),
                    "is_ensemble": e.is_ensemble,
                }
            )
        return pd.DataFrame(rows)

    def get_best_per_family(self) -> dict[str, LeaderboardEntry]:
        """Get best model per algorithm family."""
        best: dict[str, LeaderboardEntry] = {}
        for e in self._entries:
            if e.family not in best:
                best[e.family] = e
        return best

    def __repr__(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "Leaderboard (empty)"
        return f"Leaderboard ({len(self._entries)} models):\n{df.to_string(index=False)}"

    def __len__(self) -> int:
        return len(self._entries)
