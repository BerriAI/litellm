from typing import Any, Optional, cast

import pytest

from litellm.litellm_core_utils.otel_span import (
    LiteLLMOtelSpan,
    _DetailedOtelFeatureGate,
    get_current_otel_span,
    litellm_otel_tracer,
    set_litellm_otel_logger,
)
from litellm.types.services import ServiceTypes


class _FakeSpan:
    def __init__(self, name: str, start_time: int) -> None:
        self.name = name
        self.start_time = start_time
        self.end_time: Optional[int] = None
        self.attributes: dict[str, Any] = {}
        self.status: Any = None
        self.recorded_exception: Optional[BaseException] = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exception: BaseException) -> None:
        self.recorded_exception = exception

    def end(self, end_time: int) -> None:
        self.end_time = end_time


class _FakeTracer:
    def __init__(self) -> None:
        self.started_spans: list[_FakeSpan] = []

    def start_span(
        self, name: str, context: Any = None, start_time: Optional[int] = None
    ) -> _FakeSpan:
        span = _FakeSpan(name=name, start_time=start_time or 0)
        self.started_spans.append(span)
        return span


class _FakeOpenTelemetryLogger:
    def __init__(self) -> None:
        self.tracer = _FakeTracer()

    def safe_set_attribute(self, span: _FakeSpan, key: str, value: Any) -> None:
        span.set_attribute(key, value)


def test_litellm_otel_span_sets_and_resets_current_span() -> None:
    logger = _FakeOpenTelemetryLogger()

    assert get_current_otel_span() is None

    with LiteLLMOtelSpan(
        span_name="proxy.auth.read_request_body",
        service=ServiceTypes.AUTH,
        attributes={"route": "/v1/chat/completions", "cache_hit": False},
        require_parent=False,
        otel_logger=logger,
    ) as span_context:
        assert span_context.span is not None
        span = cast(_FakeSpan, span_context.span)
        assert get_current_otel_span() is span_context.span
        assert span.name == "proxy.auth.read_request_body"
        assert span.attributes["call_type"] == ("proxy.auth.read_request_body")
        assert span.attributes["service"] == ServiceTypes.AUTH.value
        assert span.attributes["route"] == "/v1/chat/completions"
        assert span.attributes["cache_hit"] is False

    assert get_current_otel_span() is None
    assert logger.tracer.started_spans[0].end_time is not None


def test_litellm_otel_span_uses_registered_logger() -> None:
    logger = _FakeOpenTelemetryLogger()
    set_litellm_otel_logger(logger)

    try:
        with litellm_otel_tracer.trace(
            "proxy.auth.fetch",
            service=ServiceTypes.AUTH,
            require_parent=False,
        ) as span_context:
            assert span_context.span is not None
            span = cast(_FakeSpan, span_context.span)
            assert span.name == "proxy.auth.fetch"
    finally:
        set_litellm_otel_logger(None)

    assert logger.tracer.started_spans[0].end_time is not None


def test_record_completed_span_uses_explicit_end_time() -> None:
    logger = _FakeOpenTelemetryLogger()
    set_litellm_otel_logger(logger)

    try:
        litellm_otel_tracer.record_completed_span(
            span_name="litellm.logging.async_callback_dispatch",
            service=ServiceTypes.LITELLM,
            start_time=1.0,
            end_time=2.0,
            require_parent=False,
        )
    finally:
        set_litellm_otel_logger(None)

    span = logger.tracer.started_spans[0]
    assert span.start_time == 1_000_000_000
    assert span.end_time == 2_000_000_000


def test_litellm_otel_span_records_exception_and_propagates() -> None:
    logger = _FakeOpenTelemetryLogger()
    raised = ValueError("boom")

    try:
        with LiteLLMOtelSpan(
            span_name="proxy.chat.route_request",
            service=ServiceTypes.PROXY_PRE_CALL,
            require_parent=False,
            otel_logger=logger,
        ):
            raise raised
    except ValueError:
        pass

    span = logger.tracer.started_spans[0]
    assert span.recorded_exception is raised
    assert span.end_time is not None
    assert get_current_otel_span() is None


def test_detailed_otel_feature_gate_reads_env_each_call(monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_ENABLE_DETAILED_OTEL_SPANS", raising=False)

    assert _DetailedOtelFeatureGate.is_enabled() is False

    monkeypatch.setenv("LITELLM_ENABLE_DETAILED_OTEL_SPANS", "true")
    assert _DetailedOtelFeatureGate.is_enabled() is True

    monkeypatch.setenv("LITELLM_ENABLE_DETAILED_OTEL_SPANS", "false")
    assert _DetailedOtelFeatureGate.is_enabled() is False


def test_detailed_otel_span_is_skipped_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_ENABLE_DETAILED_OTEL_SPANS", raising=False)
    logger = _FakeOpenTelemetryLogger()

    with LiteLLMOtelSpan(
        span_name="proxy.auth.fetch",
        service=ServiceTypes.AUTH,
        require_parent=False,
        otel_logger=logger,
        detailed=True,
    ) as span_context:
        assert span_context.span is None

    assert logger.tracer.started_spans == []


def test_detailed_otel_span_emits_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_DETAILED_OTEL_SPANS", "true")
    logger = _FakeOpenTelemetryLogger()

    with LiteLLMOtelSpan(
        span_name="proxy.auth.fetch",
        service=ServiceTypes.AUTH,
        require_parent=False,
        otel_logger=logger,
        attributes={"fetch_label": "user"},
        detailed=True,
    ) as span_context:
        assert span_context.span is not None
        span = cast(_FakeSpan, span_context.span)
        assert span.attributes["fetch_label"] == "user"

    assert logger.tracer.started_spans[0].end_time is not None


@pytest.mark.asyncio
async def test_litellm_otel_span_supports_async_context_manager() -> None:
    logger = _FakeOpenTelemetryLogger()

    async with LiteLLMOtelSpan(
        span_name="router.set_response_headers",
        service=ServiceTypes.ROUTER,
        require_parent=False,
        otel_logger=logger,
    ) as span_context:
        span_context.set_attribute("response_type", "ModelResponse")

    span = logger.tracer.started_spans[0]
    assert span.attributes["response_type"] == "ModelResponse"
    assert span.end_time is not None
    assert get_current_otel_span() is None


async def _return_value() -> str:
    return "ok"


@pytest.mark.asyncio
async def test_litellm_otel_tracer_is_noop_without_registered_logger() -> None:
    set_litellm_otel_logger(None)

    result = await litellm_otel_tracer.trace_async(
        _return_value(),
        span_name="proxy.auth.read_request_body",
        service=ServiceTypes.AUTH,
        require_parent=False,
    )

    assert result == "ok"
    assert get_current_otel_span() is None
