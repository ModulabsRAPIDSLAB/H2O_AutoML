"""Memory-Aware Scheduler for GPU AutoML.

Schedules model training tasks based on available VRAM,
preventing OOM by queuing tasks that would exceed memory budget.

Core contribution of this framework (FR-062).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from paged_automl.memory.estimator import VRAMEstimator
from paged_automl.memory.profiler import MemoryProfiler

logger = logging.getLogger(__name__)


class SchedulingMode(Enum):
    """Scheduling mode for task execution."""

    NAIVE = "naive"  # No VRAM checking (baseline for benchmarks)
    AWARE = "aware"  # VRAM-aware scheduling (our contribution)


@dataclass
class TrainTask:
    """A training task to be scheduled."""

    task_id: str
    model_id: str
    algorithm: str
    params: dict[str, Any]
    n_rows: int
    n_features: int
    estimated_vram_gb: float = 0.0
    priority: int = 0  # lower = higher priority

    # Runtime state
    status: str = "pending"  # pending, running, completed, failed
    actual_vram_gb: float = 0.0
    result: Any = None


class MemoryAwareScheduler:
    """Schedules training tasks based on available GPU VRAM.

    In AWARE mode:
      1. Estimates VRAM for each pending task
      2. Checks available VRAM before launching
      3. Queues tasks that would exceed budget
      4. Retries queued tasks after others complete

    In NAIVE mode:
      Launches all tasks immediately (for benchmark comparison).

    Parameters
    ----------
    profiler : MemoryProfiler
        For real-time VRAM queries.
    estimator : VRAMEstimator
        For VRAM usage estimation.
    mode : SchedulingMode
        Scheduling strategy.
    vram_budget_gb : float, optional
        Maximum VRAM budget. If None, uses 90% of total VRAM.
    """

    def __init__(
        self,
        profiler: MemoryProfiler,
        estimator: VRAMEstimator,
        mode: SchedulingMode = SchedulingMode.AWARE,
        vram_budget_gb: Optional[float] = None,
    ):
        self.profiler = profiler
        self.estimator = estimator
        self.mode = mode
        self.vram_budget_gb = vram_budget_gb
        self._task_queue: list[TrainTask] = []
        self._completed: list[TrainTask] = []
        self._failed: list[TrainTask] = []

    def _get_budget(self) -> float:
        if self.vram_budget_gb is not None:
            return self.vram_budget_gb
        total = self.profiler.get_total_vram_gb()
        return total * 0.9 if total > 0 else 16.0

    def submit(self, task: TrainTask) -> None:
        """Submit a training task to the scheduler."""
        if task.estimated_vram_gb == 0:
            est = self.estimator.estimate(
                task.algorithm, task.n_rows, task.n_features, task.params
            )
            task.estimated_vram_gb = est.estimated_gb

        self._task_queue.append(task)
        logger.debug(
            f"Task {task.task_id} submitted "
            f"(est. {task.estimated_vram_gb:.2f} GB VRAM)"
        )

    def run_sequential(
        self,
        execute_fn: Callable[[TrainTask], Any],
    ) -> list[TrainTask]:
        """Execute all tasks sequentially with memory-aware gating.

        Parameters
        ----------
        execute_fn : callable
            Function that takes a TrainTask and executes it.
            Should return the result and set task.actual_vram_gb.
        """
        # Sort by priority
        self._task_queue.sort(key=lambda t: t.priority)
        budget = self._get_budget()

        results = []
        retry_queue: list[TrainTask] = []

        while self._task_queue or retry_queue:
            # Try main queue first, then retries
            if self._task_queue:
                task = self._task_queue.pop(0)
            elif retry_queue:
                task = retry_queue.pop(0)
            else:
                break

            if self.mode == SchedulingMode.AWARE:
                free_vram = self.profiler.get_free_vram_gb()
                if free_vram < task.estimated_vram_gb:
                    logger.info(
                        f"Task {task.task_id}: insufficient VRAM "
                        f"(need {task.estimated_vram_gb:.2f} GB, "
                        f"free {free_vram:.2f} GB). Queuing."
                    )
                    # If we haven't tried this task before, add to retry
                    if task.status == "pending":
                        task.status = "queued"
                        retry_queue.append(task)
                        continue
                    else:
                        # Already retried, force execute with warning
                        logger.warning(
                            f"Task {task.task_id}: retrying despite low VRAM."
                        )

            task.status = "running"
            logger.info(
                f"Executing task {task.task_id} ({task.algorithm}, "
                f"est. {task.estimated_vram_gb:.2f} GB)"
            )

            try:
                result = execute_fn(task)
                task.status = "completed"
                task.result = result
                self._completed.append(task)

                # Record actual VRAM for future estimation
                if task.actual_vram_gb > 0:
                    self.estimator.record_actual(
                        task.algorithm,
                        task.n_rows,
                        task.n_features,
                        task.params,
                        task.actual_vram_gb,
                    )
            except Exception as e:
                task.status = "failed"
                self._failed.append(task)
                logger.error(f"Task {task.task_id} failed: {e}")

            results.append(task)

        return results

    def run_parallel_dask(
        self,
        execute_fn: Callable[[TrainTask], Any],
        client: Any,
    ) -> list[TrainTask]:
        """Execute tasks in parallel using Dask with memory-aware scheduling.

        Parameters
        ----------
        execute_fn : callable
            Function to execute each task.
        client : dask.distributed.Client
            Dask client for distributed execution.
        """
        import dask
        from dask.distributed import wait

        budget = self._get_budget()
        self._task_queue.sort(key=lambda t: t.priority)

        # Group tasks by estimated VRAM to pack efficiently
        running_futures = {}
        running_vram = 0.0
        results = []

        while self._task_queue or running_futures:
            # Submit tasks that fit in budget
            while self._task_queue:
                task = self._task_queue[0]

                if self.mode == SchedulingMode.AWARE:
                    if running_vram + task.estimated_vram_gb > budget:
                        break  # Wait for running tasks to complete

                task = self._task_queue.pop(0)
                task.status = "running"
                future = client.submit(execute_fn, task)
                running_futures[future] = task
                running_vram += task.estimated_vram_gb

            if not running_futures:
                break

            # Wait for at least one task to complete
            completed_futures = wait(
                list(running_futures.keys()), return_when="FIRST_COMPLETED"
            )

            for future in completed_futures.done:
                task = running_futures.pop(future)
                running_vram -= task.estimated_vram_gb

                try:
                    result = future.result()
                    task.status = "completed"
                    task.result = result
                    self._completed.append(task)
                except Exception as e:
                    task.status = "failed"
                    self._failed.append(task)
                    logger.error(f"Task {task.task_id} failed: {e}")

                results.append(task)

        return results

    @property
    def completed_tasks(self) -> list[TrainTask]:
        return list(self._completed)

    @property
    def failed_tasks(self) -> list[TrainTask]:
        return list(self._failed)


class ContinuousScheduler:
    """vLLM Continuous Batching-style scheduler for GPU AutoML.

    vLLM이 매 iteration마다 완료된 요청을 즉시 제거하고 새 요청을 투입하듯이,
    이 스케줄러는 fold/모델 완료 즉시 블록을 회수하고 다음 task를 투입한다.

    vLLM 대응:
      vLLM Continuous Batching → ContinuousScheduler
      vLLM iteration-level scheduling → task 완료 시 즉시 재스케줄링
      vLLM preemption → LRU eviction via PagedMemoryManager

    핵심 차이 (커스텀 CUDA 커널 불필요):
      vLLM은 Attention 커널 안에서 Page Table을 참조하므로 커스텀 커널이 필수.
      AutoML은 task 경계(모델/fold 시작/끝)에서만 블록을 관리하므로
      rmm + cupy만으로 구현 가능 (Coarse-grained Paging).

    Parameters
    ----------
    paged_manager : PagedMemoryManager
        블록 풀 관리자.
    estimator : VRAMEstimator
        VRAM 사용량 추정기.
    """

    def __init__(
        self,
        paged_manager: Any,  # PagedMemoryManager
        estimator: Any,  # VRAMEstimator
    ):
        self.paged_manager = paged_manager
        self.estimator = estimator
        self._completed: list[TrainTask] = []
        self._failed: list[TrainTask] = []

    def run(
        self,
        tasks: list[TrainTask],
        execute_fn: Callable[[TrainTask], Any],
    ) -> list[TrainTask]:
        """Task들을 연속 실행 — 완료 즉시 블록 회수 → 다음 task 투입.

        vLLM의 continuous batching loop에 대응:
          while waiting or running:
            1) 완료된 task → 블록 회수 (free)
            2) 대기열에서 블록 확보 가능한 task → 투입
        """
        waiting = list(tasks)
        waiting.sort(key=lambda t: t.priority)
        results: list[TrainTask] = []

        for task in waiting:
            # 블록 수 예측
            if task.estimated_vram_gb == 0:
                est = self.estimator.estimate(
                    task.algorithm, task.n_rows, task.n_features, task.params
                )
                task.estimated_vram_gb = est.estimated_gb

            n_blocks = self.paged_manager.estimate_blocks(task.estimated_vram_gb)

            # 블록 할당 시도
            alloc = self.paged_manager.allocate(task.task_id, n_blocks)

            if not alloc.success:
                logger.warning(
                    f"Task {task.task_id}: cannot allocate {n_blocks} blocks. Skipping."
                )
                task.status = "failed"
                self._failed.append(task)
                results.append(task)
                continue

            if alloc.evicted_tasks:
                logger.info(
                    f"Evicted {len(alloc.evicted_tasks)} task(s) to host "
                    f"for {task.task_id}: {alloc.evicted_tasks}"
                )

            # Task 실행
            task.status = "running"
            logger.info(
                f"Executing {task.task_id} ({task.algorithm}, "
                f"{n_blocks} blocks, "
                f"GPU util: {self.paged_manager.gpu_utilization:.0%})"
            )

            try:
                result = execute_fn(task)
                task.status = "completed"
                task.result = result
                self._completed.append(task)

                if task.actual_vram_gb > 0:
                    self.estimator.record_actual(
                        task.algorithm, task.n_rows, task.n_features,
                        task.params, task.actual_vram_gb,
                    )
            except Exception as e:
                task.status = "failed"
                self._failed.append(task)
                logger.error(f"Task {task.task_id} failed: {e}")
            finally:
                # 즉시 블록 회수 — vLLM의 "완료 즉시 free"
                freed = self.paged_manager.free(task.task_id)
                logger.debug(
                    f"Freed {freed} blocks from {task.task_id}. "
                    f"GPU util: {self.paged_manager.gpu_utilization:.0%}"
                )

            results.append(task)

        return results

    @property
    def completed_tasks(self) -> list[TrainTask]:
        return list(self._completed)

    @property
    def failed_tasks(self) -> list[TrainTask]:
        return list(self._failed)
