import asyncio
import queue
import threading
from collections.abc import Iterator
from typing import Final

import pytest
from fakeredis import TcpFakeServer
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span
from prometheus_client import REGISTRY
from redis import exceptions as redis_exceptions

import litellm
from litellm._internal_context import service_target, set_request_parent_otel_span
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.otel import OpenTelemetryV2Config
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.plumbing import providers
from litellm.redis.stores.auth_cache import AUTH_CACHE_TARGET, AuthCacheSettings, AuthCacheStore
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
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


@pytest.fixture
def fake_redis_port() -> Iterator[int]:
    server: Final = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _children(exporter: InMemorySpanExporter, parent: Span) -> tuple[ReadableSpan, ...]:
    parent_id: Final = parent.get_span_context().span_id
    return tuple(
        span for span in exporter.get_finished_spans() if span.parent is not None and span.parent.span_id == parent_id
    )


async def _spans_under(exporter: InMemorySpanExporter, parent: Span, polls: int = 100) -> tuple[tuple[str, str], ...]:
    children: Final = _children(exporter, parent)
    if children or polls == 0:
        return tuple(sorted((span.name, span.status.status_code.name) for span in children))
    await asyncio.sleep(0.05)
    return await _spans_under(exporter, parent, polls - 1)


async def test_a_native_auth_cache_read_reports_the_redis_span_redis_cache_reported(
    fake_redis_port: int, otel: tuple[TracerProvider, InMemorySpanExporter]
):
    provider, exporter = otel
    tracer: Final = provider.get_tracer(__name__)
    redis_cache: Final = RedisCache(host="127.0.0.1", port=fake_redis_port)
    store: Final = AuthCacheStore(connect(redis_cache).auth_cache(AuthCacheSettings(200, 60)))
    python_request: Final = tracer.start_span("python request")
    native_request: Final = tracer.start_span("native request")

    with trace.use_span(python_request, end_on_exit=True):
        with service_target(AUTH_CACHE_TARGET):
            await redis_cache.async_get_cache("python-key", parent_otel_span=python_request)
    with trace.use_span(native_request, end_on_exit=True):
        await store.get("native-key")

    reported_by_redis_cache: Final = await _spans_under(exporter, python_request)
    assert reported_by_redis_cache
    assert await _spans_under(exporter, native_request) == reported_by_redis_cache


async def test_a_post_call_batch_flushed_by_its_deadline_reports_its_redis_span_under_the_request(
    fake_redis_port: int, otel: tuple[TracerProvider, InMemorySpanExporter]
):
    provider, exporter = otel
    tracer: Final = provider.get_tracer(__name__)
    store: Final = SpendCounterStore(connect(RedisCache(host="127.0.0.1", port=fake_redis_port)).spend_counters())
    request: Final = tracer.start_span("request")

    async def native_post_call() -> tuple[tuple[str, str], ...]:
        with trace.use_span(request, end_on_exit=True):
            set_request_parent_otel_span(request)
            with request_redis_batch_scope(post_call_deadline=0.05):
                assert store.increment_many_post_call((("spend:key:k", 1.0),))
            return await _spans_under(exporter, request)

    assert await asyncio.create_task(native_post_call())


@pytest.fixture
def legacy_otel(monkeypatch: pytest.MonkeyPatch) -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter: Final = InMemorySpanExporter()
    provider: Final = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    logger: Final = OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter), tracer_provider=provider)
    monkeypatch.setattr(litellm, "service_callback", [logger])
    return provider, exporter


async def test_a_native_read_in_a_legacy_otel_request_reports_its_redis_span_under_the_request(
    fake_redis_port: int, legacy_otel: tuple[TracerProvider, InMemorySpanExporter]
):
    provider, exporter = legacy_otel
    tracer: Final = provider.get_tracer(__name__)
    redis_cache: Final = RedisCache(host="127.0.0.1", port=fake_redis_port)
    store: Final = AuthCacheStore(connect(redis_cache).auth_cache(AuthCacheSettings(200, 60)))
    python_request: Final = tracer.start_span("python request")
    native_request: Final = tracer.start_span("native request")

    async def native_request_read() -> None:
        set_request_parent_otel_span(native_request)
        await store.get("native-key")

    with service_target(AUTH_CACHE_TARGET):
        await redis_cache.async_get_cache("python-key", parent_otel_span=python_request)
    await asyncio.create_task(native_request_read())

    reported_by_redis_cache: Final = await _spans_under(exporter, python_request)
    assert reported_by_redis_cache
    assert await _spans_under(exporter, native_request) == reported_by_redis_cache


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
