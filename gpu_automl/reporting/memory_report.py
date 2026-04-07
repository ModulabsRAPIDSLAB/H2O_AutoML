"""Memory profiling report generation.

Generates per-stage and per-model VRAM usage reports for research analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from gpu_automl.memory.profiler import MemoryProfiler, StageProfile
from gpu_automl.reporting.leaderboard import Leaderboard


class MemoryReport:
    """Generates memory profiling reports for analysis and publication.

    Combines stage-level profiling data with per-model VRAM data
    from the leaderboard.
    """

    def __init__(
        self,
        profiler: Optional[MemoryProfiler] = None,
        leaderboard: Optional[Leaderboard] = None,
    ):
        self.profiler = profiler
        self.leaderboard = leaderboard

    def stage_summary(self) -> pd.DataFrame:
        """Per-stage VRAM usage summary."""
        if self.profiler is None:
            return pd.DataFrame()

        rows = []
        for sp in self.profiler.stage_profiles:
            rows.append(
                {
                    "stage": sp.stage,
                    "start_vram_gb": round(sp.start_vram_gb, 3),
                    "end_vram_gb": round(sp.end_vram_gb, 3),
                    "peak_vram_gb": round(sp.peak_vram_gb, 3),
                    "delta_gb": round(sp.end_vram_gb - sp.start_vram_gb, 3),
                    "duration_secs": round(sp.duration_secs, 2),
                }
            )
        return pd.DataFrame(rows)

    def model_vram_summary(self) -> pd.DataFrame:
        """Per-model peak VRAM from leaderboard."""
        if self.leaderboard is None:
            return pd.DataFrame()

        rows = []
        for entry in self.leaderboard.entries:
            rows.append(
                {
                    "model_id": entry.model_id,
                    "algorithm": entry.algorithm,
                    "peak_vram_gb": round(entry.peak_vram_gb, 3),
                    "training_time_secs": round(entry.training_time_secs, 2),
                }
            )
        return pd.DataFrame(rows)

    def full_report(self) -> dict:
        """Generate complete memory report as dict."""
        report = {
            "stage_profiles": self.stage_summary().to_dict(orient="records"),
            "model_vram": self.model_vram_summary().to_dict(orient="records"),
        }

        if self.profiler is not None:
            report["gpu_info"] = {
                "total_vram_gb": self.profiler.get_total_vram_gb(),
                "device_id": self.profiler.device_id,
            }

        return report

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        report = self.full_report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def __repr__(self) -> str:
        stage_df = self.stage_summary()
        model_df = self.model_vram_summary()

        parts = ["=== Memory Profile Report ==="]
        if not stage_df.empty:
            parts.append("\n--- Stage Profiles ---")
            parts.append(stage_df.to_string(index=False))
        if not model_df.empty:
            parts.append("\n--- Model VRAM Usage ---")
            parts.append(model_df.to_string(index=False))
        return "\n".join(parts)
