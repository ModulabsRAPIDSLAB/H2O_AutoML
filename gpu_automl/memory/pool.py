"""rmm pool management for GPU memory optimization.

Supports multiple pool strategies for benchmarking (Phase 4):
  - No pool: default CUDA malloc
  - Fixed pool: pre-allocated fixed size
  - Managed pool: CUDA managed memory
  - Adaptive pool: growing pool with base allocation
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PoolStrategy(Enum):
    """rmm memory pool strategy."""

    NONE = "none"  # Default CUDA malloc (no pool)
    FIXED = "fixed"  # Fixed-size pre-allocated pool
    MANAGED = "managed"  # CUDA managed memory pool
    ADAPTIVE = "adaptive"  # Growing pool with initial allocation


class RMMPoolManager:
    """Manages rmm memory pools for GPU memory optimization.

    Parameters
    ----------
    strategy : PoolStrategy
        Pool allocation strategy.
    initial_pool_size_gb : float, optional
        Initial pool size in GB (for FIXED and ADAPTIVE).
    max_pool_size_gb : float, optional
        Maximum pool size in GB.
    device_id : int
        GPU device index.
    """

    def __init__(
        self,
        strategy: PoolStrategy = PoolStrategy.ADAPTIVE,
        initial_pool_size_gb: Optional[float] = None,
        max_pool_size_gb: Optional[float] = None,
        device_id: int = 0,
    ):
        self.strategy = strategy
        self.initial_pool_size_gb = initial_pool_size_gb
        self.max_pool_size_gb = max_pool_size_gb
        self.device_id = device_id
        self._initialized = False

    def initialize(self) -> None:
        """Initialize rmm with the configured pool strategy."""
        try:
            import rmm
        except ImportError:
            logger.warning("rmm not available. Using default CUDA allocator.")
            return

        if self.strategy == PoolStrategy.NONE:
            rmm.reinitialize(pool_allocator=False)
            logger.info("rmm: No pool (default CUDA malloc)")

        elif self.strategy == PoolStrategy.FIXED:
            size = self._gb_to_bytes(self.initial_pool_size_gb or 8.0)
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=size,
                maximum_pool_size=size,
            )
            logger.info(
                f"rmm: Fixed pool ({self.initial_pool_size_gb or 8.0:.1f} GB)"
            )

        elif self.strategy == PoolStrategy.MANAGED:
            rmm.reinitialize(managed_memory=True)
            logger.info("rmm: Managed memory pool")

        elif self.strategy == PoolStrategy.ADAPTIVE:
            initial = self._gb_to_bytes(self.initial_pool_size_gb or 4.0)
            maximum = (
                self._gb_to_bytes(self.max_pool_size_gb)
                if self.max_pool_size_gb
                else None
            )
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=initial,
                maximum_pool_size=maximum,
            )
            logger.info(
                f"rmm: Adaptive pool (initial={self.initial_pool_size_gb or 4.0:.1f} GB, "
                f"max={self.max_pool_size_gb or 'auto'} GB)"
            )

        self._initialized = True

    @staticmethod
    def _gb_to_bytes(gb: float) -> int:
        return int(gb * 1024**3)

    def shutdown(self) -> None:
        """Reset rmm to default state."""
        if self._initialized:
            try:
                import rmm

                rmm.reinitialize()
            except Exception:
                pass
            self._initialized = False
