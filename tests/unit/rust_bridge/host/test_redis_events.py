import asyncio
import queue
import threading
from collections.abc import Awaitable, Iterator, Mapping
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import asdict
from typing import Final

import pytest
from fakeredis import TcpFakeServer
from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span
from prometheus_client import REGISTRY
from redis import exceptions as redis_exceptions

import litellm
from litellm import Router
from litellm._internal_context import (
    post_response_phase,
    request_parent_otel_span,
    service_target,
    set_request_parent_otel_span,
)
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.otel import OpenTelemetryV2Config
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.baggage import BAGGAGE_PROMOTED_KEYS
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.integrations.otel.model.semconv import LiteLLM
from litellm.integrations.otel.plumbing import providers
from litellm.integrations.otel.plumbing.context import set_request_baggage
from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE
from litellm.proxy import proxy_server
from litellm.proxy.hooks.rust_spend_logger import _SpendLogRustLogger
from litellm.redis.stores.auth_cache import AUTH_CACHE_TARGET, AuthCacheSettings, AuthCacheStore, AuthObject
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import RouterUsageStore, UsageCounter
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_COUNTER_KEY,
    ITPM_RESERVED_KEY,
    OTPM_COUNTER_KEY,
    OTPM_RESERVED_KEY,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
from litellm.rust_bridge.host import redis_events


def _sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    redis_events._service_logging.cache_clear()
    logger = redis_events._service_logging().prometheusServicesLogger
    for collector in (logger.redis_breaker_state, logger.redis_breaker_transitions, logger.redis_breaker_failures):
        if collector not in REGISTRY._collector_to_names:
            REGISTRY.register(collector)
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
    config: Final = OpenTelemetryV2Config(
        exporter="in_memory", baggage_promoted_keys=[*BAGGAGE_PROMOTED_KEYS, LiteLLM.END_USER]
    )
    exporter: Final = InMemorySpanExporter()
    provider: Final = providers.build_tracer_provider(config, exporter=exporter)
    monkeypatch.setattr(litellm, "service_callback", [OpenTelemetryV2(config=config, tracer_provider=provider)])
    yield provider, exporter


@pytest.fixture
def background_otel(
    monkeypatch: pytest.MonkeyPatch, otel: tuple[TracerProvider, InMemorySpanExporter]
) -> Iterator[OpenTelemetryV2]:
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    logger = litellm.service_callback[0]
    assert isinstance(logger, OpenTelemetryV2)
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)
    yield logger
    is_otel_v2_enabled.cache_clear()


@pytest.mark.parametrize("enabled", [True, False])
def test_a_parentless_rust_redis_event_is_grouped_only_when_background_tracing_is_enabled(
    enabled: bool, background_otel: OpenTelemetryV2, otel: tuple[TracerProvider, InMemorySpanExporter]
) -> None:
    _, exporter = otel
    background_otel.background_service_spans_db_override = enabled

    redis_events.record_redis_call(None, "GET", 10.0, 10.25, None)

    spans = exporter.get_finished_spans()
    if not enabled:
        assert spans == ()
        return
    child, root = spans
    assert (root.name, root.kind, root.parent) == ("litellm.background.rust_redis", trace.SpanKind.INTERNAL, None)
    assert child.name == "redis GET"
    assert child.parent == root.context
    assert root.start_time == child.start_time


@pytest.mark.parametrize("enabled", [True, False])
def test_a_rust_redis_event_preserves_an_explicit_request_parent(
    enabled: bool, background_otel: OpenTelemetryV2, otel: tuple[TracerProvider, InMemorySpanExporter]
) -> None:
    provider, exporter = otel
    background_otel.background_service_spans_db_override = enabled
    parent: Final = provider.get_tracer(__name__).start_span("request")

    with request_parent_otel_span(parent):
        redis_events.record_redis_call(None, "GET", 10.0, 10.25, None)
    parent.end()

    assert [(span.name, span.parent) for span in exporter.get_finished_spans()] == [
        ("redis GET", parent.get_span_context()),
        ("request", None),
    ]


class _SpendRedisReport:
    def manifest(self) -> str:
        return ""

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        report_context: Final = copy_context()

        async def run() -> None:
            await asyncio.to_thread(report_context.run, redis_events.record_redis_call, None, "MGET", 10.0, 10.25, None)
            await asyncio.to_thread(redis_events.record_redis_call, None, "GET", 10.0, 10.25, None)

        return run()


@pytest.mark.asyncio
@pytest.mark.parametrize(("post_response", "enabled"), [(True, True), (True, False), (False, False)])
async def test_spend_redis_leaves_the_request_trace_only_after_the_response(
    post_response: bool,
    enabled: bool,
    background_otel: OpenTelemetryV2,
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel
    background_otel.background_service_spans_db_override = enabled
    logger: Final = _SpendLogRustLogger(_SpendRedisReport())
    token: Final = V2_ASYNC_LOGGING_LINE.set(True)
    identity: Final = {LiteLLM.TEAM_ID: "spend-team", LiteLLM.END_USER: "spend-user"}
    baggage_token: Final = attach(set_request_baggage(identity))
    try:
        with (
            provider.get_tracer(__name__).start_as_current_span("request") as request,
            request_parent_otel_span(request),
        ):
            with post_response_phase() if post_response else nullcontext():
                _ = await logger.run_rust(
                    Hook.attempt_ended, frozenset(), object(), CallContext(Hook.attempt_ended, None, None)
                )
            assert trace.get_current_span() is request
    finally:
        detach(baggage_token)
        V2_ASYNC_LOGGING_LINE.reset(token)

    spans: Final = exporter.get_finished_spans()
    if post_response and enabled:
        assert [span.name for span in spans] == ["redis MGET", "redis GET", "litellm.background.spend_log", "request"]
        child, existing, root, _ = spans
        assert (root.kind, root.parent, root.links) == (trace.SpanKind.INTERNAL, None, ())
        assert (child.parent, child.links) == (root.context, ())
        assert root.attributes is not None and child.attributes is not None
        assert {key: root.attributes.get(key) for key in identity} == identity
        assert {key: child.attributes.get(key) for key in identity} == identity
        assert existing.parent is None
        assert [link.context for link in existing.links] == [request.get_span_context()]
    elif post_response:
        assert [span.name for span in spans] == ["redis GET", "request"]
        existing, _ = spans
        assert existing.parent is None
        assert [link.context for link in existing.links] == [request.get_span_context()]
    else:
        assert [(span.name, span.parent, span.links) for span in spans] == [
            ("redis MGET", request.get_span_context(), ()),
            ("redis GET", request.get_span_context(), ()),
            ("request", None, ()),
        ]


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
        await store.get(AuthObject.user("native-key"))

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
        await store.get(AuthObject.user("native-key"))

    with service_target(AUTH_CACHE_TARGET):
        await redis_cache.async_get_cache("python-key", parent_otel_span=python_request)
    await asyncio.create_task(native_request_read())

    reported_by_redis_cache: Final = await _spans_under(exporter, python_request)
    assert reported_by_redis_cache
    assert await _spans_under(exporter, native_request) == reported_by_redis_cache


async def _redis_calls_under(exporter: InMemorySpanExporter, parent: Span, polls: int = 100) -> frozenset[str]:
    calls: Final = frozenset(
        str(span.attributes["call_type"])
        for span in _children(exporter, parent)
        if span.attributes is not None and span.attributes.get("service") == "redis"
    )
    if calls or polls == 0:
        return calls
    await asyncio.sleep(0.05)
    return await _redis_calls_under(exporter, parent, polls - 1)


async def test_a_deadline_flush_reports_under_the_request_span_auth_set_after_the_scope_opened(
    fake_redis_port: int, legacy_otel: tuple[TracerProvider, InMemorySpanExporter]
):
    provider, exporter = legacy_otel
    store: Final = SpendCounterStore(connect(RedisCache(host="127.0.0.1", port=fake_redis_port)).spend_counters())
    request: Final = provider.get_tracer(__name__).start_span("request")

    async def proxy_request() -> frozenset[str]:
        with request_redis_batch_scope(post_call_deadline=0.05):
            set_request_parent_otel_span(request)
            assert store.increment_many_post_call((("spend:key:k", 1.0),))
        return await _redis_calls_under(exporter, request)

    assert await asyncio.create_task(proxy_request()) == {"post_call_redis_batch"}


async def test_a_failed_sdk_calls_io_token_refund_reports_its_redis_span_under_the_span_its_caller_passed(
    fake_redis_port: int, legacy_otel: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
):
    provider, exporter = legacy_otel
    usage: Final = RouterUsageStore(connect(RedisCache(host="127.0.0.1", port=fake_redis_port)).router_usage())
    monkeypatch.setattr(litellm, "callbacks", [ModelRateLimitingCheck(usage)])
    caller: Final = provider.get_tracer(__name__).start_span("sdk caller")
    itpm: Final = UsageCounter("global_router", "d1", "openai/gpt-x", "12-00", "itpm")
    otpm: Final = UsageCounter("global_router", "d1", "openai/gpt-x", "12-00", "otpm")

    with pytest.raises(Exception, match="provider down"):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=Exception("provider down"),
            metadata={
                "litellm_parent_otel_span": caller,
                ITPM_RESERVED_KEY: 5,
                OTPM_RESERVED_KEY: 7,
                ITPM_COUNTER_KEY: asdict(itpm),
                OTPM_COUNTER_KEY: asdict(otpm),
            },
        )

    assert await _redis_calls_under(exporter, caller) == {"async_increment"}


async def test_an_sdk_router_call_reports_its_usage_read_under_the_span_its_caller_passed(
    fake_redis_port: int, legacy_otel: tuple[TracerProvider, InMemorySpanExporter]
):
    provider, exporter = legacy_otel
    caller: Final = provider.get_tracer(__name__).start_span("sdk caller")
    router: Final = Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "sk-fake", "mock_response": "hi", "tpm": 1000},
                "model_info": {"id": "d1"},
            }
        ],
        routing_strategy="usage-based-routing-v2",
        redis_host="127.0.0.1",
        redis_port=fake_redis_port,
    )

    await router.acompletion(
        model="m", messages=[{"role": "user", "content": "hi"}], metadata={"litellm_parent_otel_span": caller}
    )

    assert "async_batch_get_cache" in await _redis_calls_under(exporter, caller)


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
