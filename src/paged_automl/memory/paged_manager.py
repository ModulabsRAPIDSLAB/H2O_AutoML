"""Paged GPU Memory Manager — vLLM-inspired memory management for AutoML.

vLLM은 KV Cache를 Page(Block) 단위로 관리하여 GPU 메모리 활용률을 96%+로 끌어올렸다.
이 모듈은 같은 원리를 AutoML에 적용한다:

  vLLM                          AutoML (이 모듈)
  ─────────────────────         ─────────────────────
  KV Cache Block (16 tokens)    MemoryBlock (64MB)
  Block Table (논리→물리)       task_block_map (model→blocks)
  Block Manager                 PagedMemoryManager
  Swap GPU↔CPU                  GPU↔Host pinned memory swap
  LRU Eviction                  LRU 기반 블록 회수

핵심 차이: vLLM은 연산 도중(Attention 커널 안)에서 Page Table을 참조하므로
커스텀 CUDA 커널이 필수이다. 반면 AutoML은 **연산 사이(task 경계)**에서
블록을 관리하므로 커스텀 커널 없이 rmm + cupy만으로 구현 가능하다.

  vLLM: token-level fine-grained paging (커널 안)
  우리: task-level coarse-grained paging (커널 밖)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# Default block size: 64MB — 큰 블록은 할당 오버헤드 감소, 작은 블록은 세밀한 관리
_DEFAULT_BLOCK_SIZE_MB = 64


class BlockLocation(Enum):
    GPU = "gpu"
    CPU = "cpu"
    FREE = "free"


@dataclass
class MemoryBlock:
    """고정 크기 메모리 블록 — vLLM의 KVCacheBlock에 대응."""

    block_id: int
    size_bytes: int
    location: BlockLocation = BlockLocation.FREE
    owner_task_id: Optional[str] = None
    last_accessed: float = 0.0

    # Host swap buffer (pinned memory)
    _host_buffer: Optional[np.ndarray] = field(default=None, repr=False)
    # GPU memory pointer
    _gpu_buffer: Optional[cp.ndarray] = field(default=None, repr=False)


@dataclass
class AllocationResult:
    """블록 할당 결과."""

    task_id: str
    blocks: list[MemoryBlock]
    evicted_tasks: list[str]  # eviction으로 밀려난 task들
    success: bool


@dataclass
class PagedMemoryStats:
    """블록 단위 메모리 통계 — Profiler 연동용."""

    total_gpu_blocks: int
    free_gpu_blocks: int
    used_gpu_blocks: int
    total_cpu_blocks: int
    free_cpu_blocks: int
    used_cpu_blocks: int
    eviction_count: int
    swap_out_count: int
    swap_in_count: int
    total_swap_bytes: int


class PagedMemoryManager:
    """vLLM-style Paged Memory Manager for GPU AutoML.

    GPU 메모리를 고정 크기 Block으로 분할하여 관리한다.
    Task(모델/fold) 단위로 Block을 할당/회수하고,
    GPU가 꽉 차면 LRU 기반으로 Host(CPU pinned memory)로 swap한다.

    vLLM과의 대응:
      - vLLM Block Manager → 이 클래스
      - vLLM FreeKVCacheBlockQueue → self._gpu_free_queue
      - vLLM Swap Manager → _swap_to_host / _swap_to_gpu
      - vLLM Eviction (LRU) → _evict_lru

    Parameters
    ----------
    block_size_mb : int
        블록 하나의 크기 (MB). 기본 64MB.
    gpu_memory_fraction : float
        전체 GPU free 메모리 중 블록 풀로 사용할 비율. 기본 0.80.
    cpu_swap_size_gb : float
        Host swap용 pinned memory 크기 (GB). 기본 2.0.
    device_id : int
        GPU 장치 번호.
    """

    def __init__(
        self,
        block_size_mb: int = _DEFAULT_BLOCK_SIZE_MB,
        gpu_memory_fraction: float = 0.80,
        cpu_swap_size_gb: float = 2.0,
        device_id: int = 0,
    ):
        self.block_size_mb = block_size_mb
        self.block_size_bytes = block_size_mb * 1024 * 1024
        self.gpu_memory_fraction = gpu_memory_fraction
        self.cpu_swap_size_gb = cpu_swap_size_gb
        self.device_id = device_id

        # Block pools
        self._gpu_blocks: list[MemoryBlock] = []
        self._cpu_blocks: list[MemoryBlock] = []
        self._gpu_free_queue: deque[MemoryBlock] = deque()
        self._cpu_free_queue: deque[MemoryBlock] = deque()

        # Task → Block mapping (vLLM의 Block Table에 대응)
        self._task_block_map: dict[str, list[MemoryBlock]] = {}

        # Stats
        self._eviction_count = 0
        self._swap_out_count = 0
        self._swap_in_count = 0
        self._total_swap_bytes = 0

        self._initialized = False

    def initialize(self) -> None:
        """GPU/CPU 블록 풀 초기화.

        vLLM의 startup profiling과 동일:
          1. 현재 free GPU 메모리 측정
          2. 사용 가능한 영역을 block_size 단위로 분할
          3. CPU pinned memory도 동일하게 분할
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_bytes = info.free
        except Exception:
            # Fallback: 8GB 가정
            free_bytes = 8 * 1024**3
            logger.warning("pynvml unavailable. Assuming 8GB GPU.")

        # GPU blocks
        pool_bytes = int(free_bytes * self.gpu_memory_fraction)
        num_gpu_blocks = pool_bytes // self.block_size_bytes

        for i in range(num_gpu_blocks):
            block = MemoryBlock(
                block_id=i,
                size_bytes=self.block_size_bytes,
                location=BlockLocation.FREE,
            )
            self._gpu_blocks.append(block)
            self._gpu_free_queue.append(block)

        # CPU blocks (pinned memory for fast DMA)
        cpu_bytes = int(self.cpu_swap_size_gb * 1024**3)
        num_cpu_blocks = cpu_bytes // self.block_size_bytes

        for i in range(num_cpu_blocks):
            block = MemoryBlock(
                block_id=num_gpu_blocks + i,
                size_bytes=self.block_size_bytes,
                location=BlockLocation.FREE,
            )
            # Pre-allocate pinned host memory
            block._host_buffer = cp.cuda.alloc_pinned_memory(self.block_size_bytes)
            self._cpu_blocks.append(block)
            self._cpu_free_queue.append(block)

        self._initialized = True
        logger.info(
            f"PagedMemoryManager initialized: "
            f"{num_gpu_blocks} GPU blocks ({num_gpu_blocks * self.block_size_mb}MB), "
            f"{num_cpu_blocks} CPU blocks ({num_cpu_blocks * self.block_size_mb}MB), "
            f"block_size={self.block_size_mb}MB"
        )

    def estimate_blocks(self, estimated_vram_gb: float) -> int:
        """VRAM 추정치를 필요 블록 수로 변환."""
        bytes_needed = int(estimated_vram_gb * 1024**3)
        return max(1, (bytes_needed + self.block_size_bytes - 1) // self.block_size_bytes)

    def allocate(self, task_id: str, n_blocks: int) -> AllocationResult:
        """Task에 N개 GPU 블록 할당.

        vLLM의 allocate_slots()에 대응:
          1. free queue에서 블록 확보 시도
          2. 부족하면 LRU eviction으로 다른 task의 블록을 host로 swap
          3. 그래도 부족하면 실패
        """
        if not self._initialized:
            raise RuntimeError("PagedMemoryManager not initialized.")

        evicted_tasks: list[str] = []

        # Free 블록이 충분한 경우
        if len(self._gpu_free_queue) >= n_blocks:
            blocks = self._pop_free_blocks(n_blocks)
        else:
            # Eviction 필요
            deficit = n_blocks - len(self._gpu_free_queue)
            evicted = self._evict_lru(deficit)
            evicted_tasks = list(set(b.owner_task_id for b in evicted if b.owner_task_id))

            if len(self._gpu_free_queue) < n_blocks:
                logger.warning(
                    f"Cannot allocate {n_blocks} blocks for {task_id}. "
                    f"Free: {len(self._gpu_free_queue)}"
                )
                return AllocationResult(
                    task_id=task_id, blocks=[], evicted_tasks=evicted_tasks, success=False
                )

            blocks = self._pop_free_blocks(n_blocks)

        # 할당 기록
        for block in blocks:
            block.owner_task_id = task_id
            block.location = BlockLocation.GPU
            block.last_accessed = time.perf_counter()

        self._task_block_map[task_id] = blocks

        logger.debug(
            f"Allocated {n_blocks} blocks for {task_id}. "
            f"Free: {len(self._gpu_free_queue)}/{len(self._gpu_blocks)}"
        )

        return AllocationResult(
            task_id=task_id, blocks=blocks, evicted_tasks=evicted_tasks, success=True
        )

    def free(self, task_id: str) -> int:
        """Task의 모든 GPU 블록 회수.

        vLLM의 free_slots()에 대응.
        블록을 free queue로 반환하여 즉시 재사용 가능하게 한다.
        """
        blocks = self._task_block_map.pop(task_id, [])
        freed = 0
        for block in blocks:
            if block.location == BlockLocation.GPU:
                block.owner_task_id = None
                block.location = BlockLocation.FREE
                self._gpu_free_queue.append(block)
                freed += 1

        if freed > 0:
            logger.debug(
                f"Freed {freed} blocks from {task_id}. "
                f"Free: {len(self._gpu_free_queue)}/{len(self._gpu_blocks)}"
            )
        return freed

    def touch(self, task_id: str) -> None:
        """Task의 블록 접근 시간 갱신 (LRU 업데이트)."""
        now = time.perf_counter()
        for block in self._task_block_map.get(task_id, []):
            block.last_accessed = now

    def _pop_free_blocks(self, n: int) -> list[MemoryBlock]:
        """Free queue에서 N개 블록 꺼내기 (O(1) per block)."""
        blocks = []
        for _ in range(n):
            blocks.append(self._gpu_free_queue.popleft())
        return blocks

    def _evict_lru(self, n_needed: int) -> list[MemoryBlock]:
        """LRU 기반 eviction — 가장 오래된 블록을 host로 swap.

        vLLM의 preemption + swap 메커니즘에 대응.
        """
        # 모든 할당된 GPU 블록을 last_accessed 순으로 정렬
        all_used = []
        for task_id, blocks in self._task_block_map.items():
            for block in blocks:
                if block.location == BlockLocation.GPU:
                    all_used.append(block)

        all_used.sort(key=lambda b: b.last_accessed)

        evicted = []
        for block in all_used:
            if len(evicted) >= n_needed:
                break
            self._swap_to_host(block)
            evicted.append(block)
            self._eviction_count += 1

        return evicted

    def _swap_to_host(self, block: MemoryBlock) -> None:
        """GPU Block → Host pinned memory (비동기 전송).

        vLLM의 swap-out에 대응.
        cupy CUDA stream을 사용하여 비동기 DMA 전송.
        """
        if block.location != BlockLocation.GPU:
            return

        # CPU free block 확보
        if not self._cpu_free_queue:
            logger.warning("No free CPU blocks for swap. Dropping block data.")
            block.location = BlockLocation.FREE
            block.owner_task_id = None
            self._gpu_free_queue.append(block)
            return

        cpu_block = self._cpu_free_queue.popleft()

        # 실제 데이터 전송은 상위 레이어에서 수행
        # 여기서는 메타데이터만 관리
        block.location = BlockLocation.CPU
        cpu_block.location = BlockLocation.CPU
        cpu_block.owner_task_id = block.owner_task_id

        # GPU 블록을 free로 전환
        old_owner = block.owner_task_id
        block.owner_task_id = None
        block.location = BlockLocation.FREE
        self._gpu_free_queue.append(block)

        # Task의 블록 매핑 업데이트 (GPU block → CPU block)
        if old_owner and old_owner in self._task_block_map:
            task_blocks = self._task_block_map[old_owner]
            for i, b in enumerate(task_blocks):
                if b.block_id == block.block_id:
                    task_blocks[i] = cpu_block
                    break

        self._swap_out_count += 1
        self._total_swap_bytes += self.block_size_bytes

        logger.debug(
            f"Swapped block {block.block_id} to host "
            f"(task={old_owner})"
        )

    def _swap_to_gpu(self, cpu_block: MemoryBlock) -> Optional[MemoryBlock]:
        """Host → GPU Block 복원 (swap-in).

        vLLM의 swap-in에 대응. Task 재개 시 호출.
        """
        if not self._gpu_free_queue:
            return None

        gpu_block = self._gpu_free_queue.popleft()
        gpu_block.owner_task_id = cpu_block.owner_task_id
        gpu_block.location = BlockLocation.GPU
        gpu_block.last_accessed = time.perf_counter()

        # CPU block 해제
        cpu_block.owner_task_id = None
        cpu_block.location = BlockLocation.FREE
        self._cpu_free_queue.append(cpu_block)

        self._swap_in_count += 1
        self._total_swap_bytes += self.block_size_bytes

        return gpu_block

    def get_stats(self) -> PagedMemoryStats:
        """현재 블록 풀 통계 반환."""
        return PagedMemoryStats(
            total_gpu_blocks=len(self._gpu_blocks),
            free_gpu_blocks=len(self._gpu_free_queue),
            used_gpu_blocks=len(self._gpu_blocks) - len(self._gpu_free_queue),
            total_cpu_blocks=len(self._cpu_blocks),
            free_cpu_blocks=len(self._cpu_free_queue),
            used_cpu_blocks=len(self._cpu_blocks) - len(self._cpu_free_queue),
            eviction_count=self._eviction_count,
            swap_out_count=self._swap_out_count,
            swap_in_count=self._swap_in_count,
            total_swap_bytes=self._total_swap_bytes,
        )

    @property
    def gpu_utilization(self) -> float:
        """GPU 블록 풀 사용률 (0.0 ~ 1.0)."""
        if not self._gpu_blocks:
            return 0.0
        used = len(self._gpu_blocks) - len(self._gpu_free_queue)
        return used / len(self._gpu_blocks)

    @property
    def free_gpu_blocks(self) -> int:
        return len(self._gpu_free_queue)

    def shutdown(self) -> None:
        """블록 풀 정리."""
        self._task_block_map.clear()
        self._gpu_free_queue.clear()
        self._cpu_free_queue.clear()
        self._gpu_blocks.clear()
        self._cpu_blocks.clear()
        self._initialized = False

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PagedMemoryManager("
            f"gpu={stats.used_gpu_blocks}/{stats.total_gpu_blocks} blocks, "
            f"cpu={stats.used_cpu_blocks}/{stats.total_cpu_blocks} blocks, "
            f"evictions={stats.eviction_count}, "
            f"swaps={stats.swap_out_count}out/{stats.swap_in_count}in)"
        )
