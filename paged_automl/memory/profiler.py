"""GPU memory profiler using pynvml and rmm.

Tracks VRAM allocation per stage and per model for the memory report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time GPU memory measurement."""

    timestamp: float
    stage: str
    used_gb: float
    total_gb: float
    peak_gb: float


@dataclass
class StageProfile:
    """Memory profile for a pipeline stage."""

    stage: str
    start_vram_gb: float
    end_vram_gb: float
    peak_vram_gb: float
    duration_secs: float


class MemoryProfiler:
    """GPU VRAM profiler using pynvml for real-time memory queries.

    Records per-stage memory usage for the memory report.
    Optionally uses rmm logging for fine-grained allocation tracking.

    Parameters
    ----------
    device_id : int
        GPU device index.
    enable_rmm_logging : bool
        Enable rmm allocation logging (may have overhead).
    """

    def __init__(self, device_id: int = 0, enable_rmm_logging: bool = False):
        self.device_id = device_id
        self.enable_rmm_logging = enable_rmm_logging
        self._handle = None
        self._snapshots: list[MemorySnapshot] = []
        self._stage_profiles: list[StageProfile] = []
        self._current_stage: Optional[str] = None
        self._stage_start_time: float = 0.0
        self._stage_start_vram: float = 0.0
        self._stage_peak_vram: float = 0.0
        self._initialized = False

    def initialize(self) -> None:
        """Initialize pynvml and optionally rmm logging."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._initialized = True
            logger.info(
                f"MemoryProfiler initialized (GPU {self.device_id}, "
                f"total VRAM: {self.get_total_vram_gb():.1f} GB)"
            )
        except Exception as e:
            logger.warning(f"pynvml initialization failed: {e}. Profiling disabled.")
            self._initialized = False

        if self.enable_rmm_logging and self._initialized:
            self._setup_rmm_logging()

    def _setup_rmm_logging(self) -> None:
        """Configure rmm with logging resource adaptor."""
        try:
            import rmm

            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=None,  # auto
                logging=True,
            )
            logger.info("rmm logging enabled.")
        except Exception as e:
            logger.warning(f"rmm logging setup failed: {e}")

    def get_current_vram_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if not self._initialized:
            return 0.0
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / (1024**3)

    def get_free_vram_gb(self) -> float:
        """Get free GPU memory in GB."""
        if not self._initialized:
            return 0.0
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.free / (1024**3)

    def get_total_vram_gb(self) -> float:
        """Get total GPU memory in GB."""
        if not self._initialized:
            return 0.0
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.total / (1024**3)

    def get_peak_vram_gb_since(self, baseline_gb: float) -> float:
        """Get peak VRAM increase since a baseline measurement."""
        current = self.get_current_vram_gb()
        return max(current - baseline_gb, 0.0)

    def begin_stage(self, stage: str) -> None:
        """Mark the beginning of a pipeline stage."""
        if self._current_stage is not None:
            self.end_stage()

        self._current_stage = stage
        self._stage_start_time = time.perf_counter()
        self._stage_start_vram = self.get_current_vram_gb()
        self._stage_peak_vram = self._stage_start_vram

        self._take_snapshot(stage)

    def end_stage(self) -> None:
        """Mark the end of the current pipeline stage."""
        if self._current_stage is None:
            return

        end_vram = self.get_current_vram_gb()
        self._stage_peak_vram = max(self._stage_peak_vram, end_vram)

        profile = StageProfile(
            stage=self._current_stage,
            start_vram_gb=self._stage_start_vram,
            end_vram_gb=end_vram,
            peak_vram_gb=self._stage_peak_vram,
            duration_secs=time.perf_counter() - self._stage_start_time,
        )
        self._stage_profiles.append(profile)

        self._take_snapshot(f"{self._current_stage}_end")
        self._current_stage = None

    def update_peak(self) -> None:
        """Update peak VRAM for current stage."""
        if self._current_stage is not None:
            current = self.get_current_vram_gb()
            self._stage_peak_vram = max(self._stage_peak_vram, current)

    def _take_snapshot(self, stage: str) -> None:
        """Take a memory snapshot."""
        if not self._initialized:
            return
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self._snapshots.append(
            MemorySnapshot(
                timestamp=time.perf_counter(),
                stage=stage,
                used_gb=info.used / (1024**3),
                total_gb=info.total / (1024**3),
                peak_gb=self._stage_peak_vram,
            )
        )

    @property
    def stage_profiles(self) -> list[StageProfile]:
        return list(self._stage_profiles)

    @property
    def snapshots(self) -> list[MemorySnapshot]:
        return list(self._snapshots)

    def shutdown(self) -> None:
        """Clean up pynvml."""
        if self._current_stage is not None:
            self.end_stage()
        if self._initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
