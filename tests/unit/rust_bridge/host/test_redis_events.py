import asyncio
import queue
import threading
from collections.abc import Iterator
from typing import Final

import pytest
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY
from redis import exceptions as redis_exceptions

import litellm
from litellm.integrations.otel import OpenTelemetryV2Config
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.plumbing import providers
from litellm.rust_bridge.host import redis_events


def _sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    redis_events._service_logging.cache_clear()
    yield
    redis_events._service_logging.cache_clear()


def _failed_requests(error: type[redis_exceptions.RedisError]) -> tuple[str, dict[str, str]]:
    return (
        "litellm_redis_failed_requests_total",
        {"redis": "redis", "error_class": error.__name__, "function_name": "async_increment"},
    )


@pytest.mark.parametrize(
    ("failure", "error", "failure_class"),
    [
        ("timeout", redis_exceptions.TimeoutError, "timeout"),
        ("transport", redis_exceptions.ConnectionError, "connectivity"),
    ],
)
def test_a_health_failure_counts_as_a_failed_request_of_its_redis_py_class_and_a_breaker_failure(
    prometheus: None, failure: redis_events.Failure, error: type[redis_exceptions.RedisError], failure_class: str
):
    breaker: Final = ("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
    before: Final = (_sample(*_failed_requests(error)), _sample(*breaker))

    redis_events.record_redis_call(None, "async_increment", 10.0, 10.25, failure)

    assert (_sample(*_failed_requests(error)), _sample(*breaker)) == (before[0] + 1, before[1] + 1)


@pytest.mark.parametrize(
    ("failure", "error"),
    [("server", redis_exceptions.ResponseError), ("decode", redis_exceptions.InvalidResponse)],
)
def test_a_rejected_command_is_a_failed_request_of_its_redis_py_class_but_not_a_breaker_failure(
    prometheus: None, failure: redis_events.Failure, error: type[redis_exceptions.RedisError]
):
    breaker: Final = {
        failure_class: _sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
        for failure_class in ("timeout", "connectivity")
    }
    before: Final = _sample(*_failed_requests(error))

    redis_events.record_redis_call(None, "async_increment", 10.0, 10.25, failure)

    assert _sample(*_failed_requests(error)) == before + 1
    assert {
        failure_class: _sample("litellm_redis_circuit_breaker_failures_total", {"failure_class": failure_class})
        for failure_class in ("timeout", "connectivity")
    } == breaker


def test_a_success_is_counted_and_a_breaker_rejection_is_not(prometheus: None):
    total: Final = ("litellm_redis_total_requests_total", {"redis": "redis"})
    before: Final = _sample(*total)

    redis_events.record_redis_call(None, "async_get_cache", 10.0, 10.01, None)
    redis_events.record_redis_call(None, "async_get_cache", 10.0, 10.0, "breaker_open")

    assert _sample(*total) == before + 1


def test_a_call_whose_loop_has_closed_is_still_counted(prometheus: None):
    total: Final = ("litellm_redis_total_requests_total", {"redis": "redis"})
    closed: Final = asyncio.new_event_loop()
    closed.close()
    before: Final = _sample(*total)

    redis_events.record_redis_call(closed, "async_get_cache", 10.0, 10.01, None)

    assert _sample(*total) == before + 1


@pytest.mark.parametrize("state", ["open", "half_open", "closed"])
def test_breaker_transitions_are_counted_by_state(prometheus: None, state: redis_events.BreakerState):
    labels: Final = {"state": state}
    before: Final = _sample("litellm_redis_circuit_breaker_transitions_total", labels)

    redis_events.record_redis_breaker(state)

    assert _sample("litellm_redis_circuit_breaker_transitions_total", labels) == before + 1


@pytest.fixture
def otel(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    config: Final = OpenTelemetryV2Config(exporter="in_memory")
    exporter: Final = InMemorySpanExporter()
    provider: Final = providers.build_tracer_provider(config, exporter=exporter)
    monkeypatch.setattr(litellm, "service_callback", [OpenTelemetryV2(config=config, tracer_provider=provider)])
    yield provider, exporter


class _EndedSpans(SpanProcessor):
    def __init__(self) -> None:
        self.ended: Final = queue.SimpleQueue[tuple[ReadableSpan, int]]()

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.put((span, threading.get_ident()))


async def test_a_call_reported_from_a_worker_thread_is_traced_on_the_callers_loop_with_its_times(
    otel: tuple[TracerProvider, InMemorySpanExporter],
):
    provider, _exporter = otel
    ended: Final = _EndedSpans()
    provider.add_span_processor(ended)

    await asyncio.to_thread(
        redis_events.record_redis_call, asyncio.get_running_loop(), "async_get_cache", 10.0, 10.25, None
    )

    span, thread = await asyncio.to_thread(ended.ended.get, timeout=5)
    assert (span.start_time, span.end_time, thread) == (10_000_000_000, 10_250_000_000, threading.get_ident())
