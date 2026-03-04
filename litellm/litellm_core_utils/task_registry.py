"""
Centralized asyncio task registry to prevent fire-and-forget task leaks.

All background tasks created via asyncio.create_task() should go through
this registry so they are tracked, cleaned up, and properly awaited on shutdown.
"""

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Asyncio registry that tracks background tasks for a single event loop.

    Designed for single-event-loop use (asyncio tasks are event-loop-bound).

    Prevents memory leaks by:
    1. Auto-removing tasks via done callbacks when they complete
    2. Logging warnings when pending tasks exceed a threshold
    3. Providing graceful shutdown to cancel/await all pending tasks
    """

    _instance: Optional["TaskRegistry"] = None
    _class_lock = threading.Lock()

    def __init__(self, max_pending_warning: int = 1000) -> None:
        self._tasks: set = set()
        self._lock = threading.Lock()
        self._max_pending_warning: int = max_pending_warning
        self._warned: bool = False

    @classmethod
    def get_instance(cls) -> "TaskRegistry":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton — intended for testing only."""
        cls._instance = None

    def set_max_pending_warning(self, threshold: int) -> None:
        """Set warning threshold for pending background tasks."""
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self._max_pending_warning = threshold

    def create_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """Create and track a background asyncio task.

        The task is automatically removed from the registry when it completes.
        """
        task = asyncio.create_task(coro, name=name)
        with self._lock:
            prev_pending = len(self._tasks)
            self._tasks.add(task)
            new_pending = len(self._tasks)
        task.add_done_callback(self._task_done)

        if (
            not self._warned
            and prev_pending <= self._max_pending_warning < new_pending
        ):
            logger.warning(
                "TaskRegistry: %d pending tasks (threshold: %d). "
                "Possible task leak detected.",
                new_pending,
                self._max_pending_warning,
            )
            self._warned = True

        return task

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback when a task completes — remove from registry."""
        with self._lock:
            self._tasks.discard(task)
            count = len(self._tasks)
        if self._warned and count <= self._max_pending_warning // 2:
            self._warned = False
        if not task.cancelled() and task.exception() is not None:
            logger.debug(
                "Background task %s failed: %s",
                task.get_name(),
                task.exception(),
            )

    @property
    def pending_count(self) -> int:
        """Number of currently pending tasks."""
        with self._lock:
            return len(self._tasks)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all pending tasks and wait for them to complete."""
        with self._lock:
            if not self._tasks:
                return
            pending = list(self._tasks)
        logger.info("TaskRegistry: shutting down %d pending tasks", len(pending))

        for task in pending:
            try:
                task.cancel()
            except RuntimeError:
                pass

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            remaining = [task for task in pending if not task.done()]
            logger.warning(
                "TaskRegistry: shutdown timed out after %.2f seconds; %d task(s) "
                "still pending",
                timeout,
                len(remaining),
            )
            for task in remaining:
                logger.debug(
                    "Task %s still pending after shutdown timeout",
                    task.get_name(),
                )
        except RuntimeError:
            logger.debug("TaskRegistry: event loop closed during shutdown")
        else:
            for task, result in zip(pending, results):
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.debug(
                        "Task %s raised during shutdown: %s",
                        task.get_name(),
                        result,
                    )
        finally:
            with self._lock:
                self._tasks.clear()


def tracked_create_task(
    coro: Coroutine[Any, Any, Any],
    *,
    name: Optional[str] = None,
) -> asyncio.Task:
    """Create a tracked asyncio task via the global TaskRegistry."""
    return TaskRegistry.get_instance().create_task(coro, name=name)
