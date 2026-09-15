import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, ParamSpec, TypeVar

from litellm._logging import verbose_logger
from litellm.constants import (
    LOGGING_EXECUTOR_DROPPED_TASK_LOG_INTERVAL_SECONDS,
    LOGGING_EXECUTOR_MAX_PENDING_TASKS,
    LOGGING_EXECUTOR_MAX_THREADS,
)

MAX_THREADS: Final = LOGGING_EXECUTOR_MAX_THREADS

_P = ParamSpec("_P")
_T = TypeVar("_T")


class BoundedLoggingThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor with a cap on queued-plus-running tasks.

    The default ThreadPoolExecutor work queue is unbounded, and every queued
    logging task pins its request/response payload in memory, so a sustained
    burst of sync callbacks slower than request arrival grows memory without
    bound. Logging is best-effort: once the cap is reached, new submissions
    are dropped with a rate-limited warning instead of queueing forever.
    """

    def __init__(
        self,
        max_workers: int,
        max_pending_tasks: int,
        drop_log_interval_seconds: float = LOGGING_EXECUTOR_DROPPED_TASK_LOG_INTERVAL_SECONDS,
        logger: logging.Logger = verbose_logger,
    ) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="litellm-logging")
        self._max_pending_tasks: Final = max_pending_tasks
        self._drop_log_interval_seconds: Final = drop_log_interval_seconds
        self._logger: Final = logger
        self._pending_slots: Final = threading.Semaphore(max_pending_tasks)
        self._drop_lock: Final = threading.Lock()
        self._dropped_since_last_log = 0
        self._last_drop_log_time = 0.0

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        if not self._pending_slots.acquire(blocking=False):
            self._record_drop()
            dropped_future: Final[Future[_T]] = Future()
            dropped_future.cancel()
            return dropped_future
        try:
            future: Final = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._pending_slots.release()
            raise
        future.add_done_callback(lambda _: self._pending_slots.release())
        return future

    def _record_drop(self) -> None:
        with self._drop_lock:
            self._dropped_since_last_log += 1
            now: Final = time.monotonic()
            if now - self._last_drop_log_time < self._drop_log_interval_seconds:
                return
            dropped_count: Final = self._dropped_since_last_log
            self._dropped_since_last_log = 0
            self._last_drop_log_time = now

        self._logger.warning(
            "litellm logging executor backlog is full (max_pending_tasks=%s); dropped %s logging task(s) "
            "since the last warning. Set LOGGING_EXECUTOR_MAX_PENDING_TASKS to raise the cap.",
            self._max_pending_tasks,
            dropped_count,
        )


executor: Final = BoundedLoggingThreadPoolExecutor(
    max_workers=MAX_THREADS,
    max_pending_tasks=LOGGING_EXECUTOR_MAX_PENDING_TASKS,
)
