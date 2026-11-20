import contextvars
import threading
from concurrent.futures import Future
from typing import Final

from litellm.integrations.kafka.worker import KafkaCallbackWorker


class Jobs:
    def __init__(self) -> None:
        self.entered: Final = threading.Event()
        self.release: Final = threading.Event()
        self.failures: tuple[str, ...] = ()
        self.executed: int = 0

    def block(self) -> None:
        self.entered.set()
        assert self.release.wait(5), "test did not release its blocked dependency"

    def execute(self) -> None:
        self.executed += 1

    def drop(self, reason: str) -> None:
        self.failures = (*self.failures, reason)


def test_a_full_preparation_queue_drops_instead_of_growing_or_executing_inline() -> None:
    jobs: Final = Jobs()
    worker: Final = KafkaCallbackWorker(1, jobs.drop)
    active: Final = worker.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        queued: Final = worker.submit(jobs.execute)
        assert worker.submit(jobs.execute) is None
        assert jobs.executed == 0
        assert jobs.failures == ("callback_queue_full",)
    finally:
        jobs.release.set()
        worker.stop(float("inf")).result(5)

    assert active is not None
    assert queued is not None
    active.result(0)
    queued.result(0)
    assert jobs.executed == 1


def test_a_cancelled_queued_callback_is_not_published() -> None:
    jobs: Final = Jobs()
    worker: Final = KafkaCallbackWorker(1, jobs.drop)
    _ = worker.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        queued: Final = worker.submit(jobs.execute)
        assert queued is not None
        assert queued.cancel()
    finally:
        jobs.release.set()
        worker.stop(float("inf")).result(5)

    assert jobs.executed == 0
    assert jobs.failures == ("cancelled_callback",)


def test_shutdown_discards_queued_callbacks_after_its_deadline() -> None:
    jobs: Final = Jobs()
    worker: Final = KafkaCallbackWorker(1, jobs.drop)
    _ = worker.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        queued: Final = worker.submit(jobs.execute)
        _ = worker.stop(0)
        assert worker.submit(jobs.execute) is None
    finally:
        jobs.release.set()
        worker.stop(0).result(5)

    assert queued is not None
    queued.result(0)
    assert jobs.executed == 0
    assert jobs.failures == ("closed", "shutdown_callback")


def test_preparation_keeps_the_callers_context() -> None:
    value: Final = contextvars.ContextVar("kafka-test-context", default="missing")
    observed: Final[Future[str]] = Future()
    worker: Final = KafkaCallbackWorker(1, lambda _: None)
    token: Final = value.set("request-context")
    try:
        finished: Final = worker.submit(lambda: observed.set_result(value.get()))
    finally:
        value.reset(token)
    worker.stop(float("inf")).result(5)

    assert finished is not None
    finished.result(0)
    assert observed.result(0) == "request-context"


def test_a_failed_preparation_does_not_stop_later_jobs_or_replay_the_failed_job() -> None:
    jobs: Final = Jobs()
    worker: Final = KafkaCallbackWorker(2, jobs.drop)

    def fail() -> None:
        raise ValueError("invalid private payload")

    first: Final = worker.submit(fail)
    following: Final = worker.submit(jobs.execute)
    worker.stop(float("inf")).result(5)

    assert first is not None and following is not None
    first.result(0)
    following.result(0)
    assert jobs.failures == ("callback_worker_error",)
    assert jobs.executed == 1
