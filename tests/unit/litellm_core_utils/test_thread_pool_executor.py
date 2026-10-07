import logging
import threading
import time
from typing import Final

from litellm._logging import verbose_logger
from litellm.constants import LOGGING_EXECUTOR_MAX_PENDING_TASKS
from litellm.litellm_core_utils.thread_pool_executor import (
    BoundedLoggingThreadPoolExecutor,
    executor,
)


def test_submit_drops_tasks_when_backlog_is_full():
    release: Final = threading.Event()
    started: Final = threading.Event()
    ran_first: Final = threading.Event()
    ran_second: Final = threading.Event()
    ran_dropped: Final = threading.Event()

    def blocking_task(ran: threading.Event) -> None:
        ran.set()
        started.set()
        release.wait(timeout=10)

    pool: Final = BoundedLoggingThreadPoolExecutor(max_workers=1, max_pending_tasks=2)
    try:
        first: Final = pool.submit(blocking_task, ran_first)
        assert started.wait(timeout=10)
        second: Final = pool.submit(blocking_task, ran_second)
        dropped: Final = pool.submit(blocking_task, ran_dropped)

        assert dropped.cancelled()
        assert not first.cancelled()
        assert not second.cancelled()

        release.set()
        first.result(timeout=10)
        second.result(timeout=10)
        assert ran_first.is_set()
        assert ran_second.is_set()
        assert not ran_dropped.is_set()
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_submit_releases_slots_after_completion():
    pool: Final = BoundedLoggingThreadPoolExecutor(max_workers=1, max_pending_tasks=1)

    def submit_and_wait() -> str:
        future: Final = pool.submit(lambda: "ok")
        assert not future.cancelled()
        return future.result(timeout=10)

    try:
        results: Final = tuple(submit_and_wait() for _ in range(5))
        assert results == ("ok",) * 5
    finally:
        pool.shutdown(wait=True)


def test_drop_warning_is_rate_limited(caplog):
    release: Final = threading.Event()
    started: Final = threading.Event()

    def blocking_task() -> None:
        started.set()
        release.wait(timeout=10)

    drop_logger: Final = logging.getLogger("test_bounded_logging_executor")
    pool: Final = BoundedLoggingThreadPoolExecutor(
        max_workers=1,
        max_pending_tasks=1,
        drop_log_interval_seconds=60.0,
        logger=drop_logger,
    )
    try:
        pool.submit(blocking_task)
        assert started.wait(timeout=10)

        with caplog.at_level(logging.WARNING, logger=drop_logger.name):
            assert pool.submit(time.sleep, 0).cancelled()
            assert pool.submit(time.sleep, 0).cancelled()
            assert pool.submit(time.sleep, 0).cancelled()

        warnings: Final = tuple(record for record in caplog.records if record.name == drop_logger.name)
        assert len(warnings) == 1
        assert warnings[0].args == (1, 1)
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_each_drop_warning_counts_only_drops_since_the_last_one(caplog):
    release: Final = threading.Event()
    started: Final = threading.Event()

    def blocking_task() -> None:
        started.set()
        release.wait(timeout=10)

    drop_logger: Final = logging.getLogger("test_bounded_logging_executor_every_drop")
    pool: Final = BoundedLoggingThreadPoolExecutor(
        max_workers=1,
        max_pending_tasks=1,
        drop_log_interval_seconds=0.0,
        logger=drop_logger,
    )
    try:
        pool.submit(blocking_task)
        assert started.wait(timeout=10)

        with caplog.at_level(logging.WARNING, logger=drop_logger.name):
            assert pool.submit(time.sleep, 0).cancelled()
            assert pool.submit(time.sleep, 0).cancelled()

        warnings: Final = tuple(record for record in caplog.records if record.name == drop_logger.name)
        assert tuple(record.args for record in warnings) == ((1, 1), (1, 1))
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_global_executor_is_bounded():
    assert isinstance(executor, BoundedLoggingThreadPoolExecutor)
    assert executor._max_pending_tasks == LOGGING_EXECUTOR_MAX_PENDING_TASKS
    assert executor._logger is verbose_logger
