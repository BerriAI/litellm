import contextvars
import threading
from concurrent.futures import Future
from typing import Final

import pytest

from litellm.integrations.kafka.producer import KafkaSink
from litellm.types.integrations.kafka import KafkaInitParams


class Jobs:
    def __init__(self) -> None:
        self.entered: Final = threading.Event()
        self.release: Final = threading.Event()
        self.failures: int = 0
        self.executed: int = 0

    def block(self) -> None:
        self.entered.set()
        assert self.release.wait(5), "test did not release its blocked dependency"

    def execute(self) -> None:
        self.executed += 1

    def failed(self) -> None:
        self.failures += 1

    def sink(self) -> KafkaSink:
        return KafkaSink(
            KafkaInitParams(bootstrap_servers="broker:9092", topic="events", callback_queue_max_messages=1),
            lambda _: pytest.fail("these jobs do not need a Kafka producer"),
            self.failed,
        )


def test_a_full_preparation_queue_drops_instead_of_growing_or_executing_inline() -> None:
    jobs: Final = Jobs()
    sink: Final = jobs.sink()
    sink.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        sink.submit(jobs.execute)
        sink.submit(jobs.execute)
        assert jobs.executed == 0
        assert sink.statistics.dropped == 1
        assert jobs.failures == 1
    finally:
        jobs.release.set()
        sink.request_close().result(5)

    assert jobs.executed == 1


def test_shutdown_discards_queued_callbacks_after_its_deadline() -> None:
    jobs: Final = Jobs()
    sink: Final = jobs.sink()
    sink.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        sink.submit(jobs.execute)
        _ = sink.request_close(0)
        sink.submit(jobs.execute)
        assert sink.statistics.dropped == 1
    finally:
        jobs.release.set()
        sink.request_close(0).result(5)

    assert jobs.executed == 0
    assert sink.statistics.dropped == 2
    assert jobs.failures == 2


def test_preparation_keeps_the_callers_context() -> None:
    value: Final = contextvars.ContextVar("kafka-test-context", default="missing")
    observed: Final[Future[str]] = Future()
    jobs: Final = Jobs()
    sink: Final = jobs.sink()
    token: Final = value.set("request-context")
    try:
        sink.submit(lambda: observed.set_result(value.get()))
    finally:
        value.reset(token)
        sink.request_close().result(5)

    assert observed.result(0) == "request-context"


def test_a_failed_preparation_does_not_stop_later_jobs_or_replay_the_failed_job() -> None:
    jobs: Final = Jobs()
    sink: Final = jobs.sink()

    def fail() -> None:
        raise ValueError("invalid private payload")

    sink.submit(fail)
    assert sink.request_flush().result(5) == 0
    sink.submit(jobs.execute)
    sink.request_close().result(5)

    assert sink.statistics.dropped == 1
    assert jobs.failures == 1
    assert jobs.executed == 1


def test_flush_waits_for_submitted_jobs_without_closing_the_destination() -> None:
    jobs: Final = Jobs()
    sink: Final = jobs.sink()
    sink.submit(jobs.block)
    try:
        assert jobs.entered.wait(5)
        flushed: Final = sink.request_flush()
        assert not flushed.done()
        sink.submit(jobs.execute)
        again: Final = sink.request_flush()
        assert not again.done()
    finally:
        jobs.release.set()
    try:
        assert flushed.result(5) == 0
        assert again.result(5) == 0
        assert jobs.executed == 1
        sink.submit(jobs.execute)
        assert sink.request_flush().result(5) == 0
        assert jobs.executed == 2
    finally:
        sink.request_close().result(5)
