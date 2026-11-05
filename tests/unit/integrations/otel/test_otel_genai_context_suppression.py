"""Tests for OpenTelemetry GenAI client suppression and context attribute propagation.

Covers LiteLLM as the inner layer per open-telemetry/semantic-conventions-genai#682 and
open-telemetry/opentelemetry-python-genai#859:
When an outer framework instrumentation (e.g. LangChain, CrewAI) has already
attached an ``InferenceData`` / ``EmbeddingData`` in ``opentelemetry.context``
under ``CLIENT_<OP>_CONTEXT_KEY``, LiteLLM suppresses its duplicate client span,
metrics, and events and publishes its resolved attributes into the active
context data object.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from opentelemetry.context import attach, detach, set_value
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.otel.genai_context import (
    CLIENT_EMBEDDING_CONTEXT_KEY,
    CLIENT_INFERENCE_CONTEXT_KEY,
    EmbeddingData,
    InferenceData,
)
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.model.semconv import Error, GenAI, LiteLLM, Server
from litellm.types.utils import StandardLoggingModelInformation, StandardLoggingPayload


def _make_payload(
    *,
    call_id: str = "call-123",
    call_type: str = "completion",
    model: str = "gpt-4o",
    custom_llm_provider: str = "openai",
    api_base: str = "https://api.openai.com:443/v1",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    response_cost: float = 0.0015,
    status: str = "success",
    error_str: str | None = None,
    error_information: dict | None = None,
) -> StandardLoggingPayload:
    now = datetime.now(timezone.utc)
    return StandardLoggingPayload(
        id="chatcmpl-xyz789",
        trace_id="trace-1",
        call_type=call_type,
        stream=False,
        response_cost=response_cost,
        status=status,  # type: ignore[typeddict-item]
        custom_llm_provider=custom_llm_provider,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        startTime=now.timestamp(),
        endTime=(now + timedelta(milliseconds=250)).timestamp(),
        completionStartTime=now.timestamp(),
        response_time=0.25,
        model_map_information=StandardLoggingModelInformation(
            model_map_key=model,
            model_map_value=model,
        ),
        model=model,
        model_id="deployment-1",
        model_group=model,
        api_base=api_base,
        metadata={
            "user_api_key_hash": "hash-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_user_id": None,
            "user_api_key_org_id": None,
            "user_api_key_team_alias": "core-team",
            "user_api_key_alias": "key-alias",
            "user_api_key_user_email": None,
            "user_api_key_end_user_id": None,
            "spend_logs_metadata": None,
            "requester_ip_address": None,
            "requester_metadata": None,
            "requester_custom_headers": None,
            "prompt_management_metadata": None,
            "applied_guardrails": None,
            "mcp_tool_call_metadata": None,
            "vector_store_request_metadata": None,
            "usage_object": None,
            "model_info": None,
            "guardrail_information": None,
        },
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address=None,
        messages=[{"role": "user", "content": "Hello world"}],
        response={
            "id": "chatcmpl-xyz789",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hi there!"}}],
        },
        error_str=error_str,
        error_information=error_information,  # type: ignore[typeddict-item]
        model_parameters={
            "temperature": 0.5,
            "max_tokens": 100,
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}],
        },
        hidden_params={"model_id": "deployment-1", "litellm_model_name": model},
        litellm_call_id=call_id,
    )


def _make_v2_harness(*, capture_span_content: bool = False):
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    log_exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    cfg = OpenTelemetryV2Config(
        exporter="in_memory",
        enable_metrics=True,
        enable_events=True,
        capture_message_content="span_and_event" if capture_span_content else "no_content",
    )
    logger = OpenTelemetryV2(
        config=cfg,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    return logger, span_exporter, metric_reader, log_exporter, tracer_provider, meter_provider, logger_provider


def _collected_metric_names(metric_reader: InMemoryMetricReader) -> list[str]:
    data = metric_reader.get_metrics_data()
    if data is None:
        return []
    names: list[str] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.append(metric.name)
    return names


# =========================================================================== #
#  1. LiteLLM V2 as Inner Layer (Suppressed by Outer GenAI Context)
# =========================================================================== #


def test_v2_suppressed_when_outer_inference_context_active():
    """When CLIENT_INFERENCE_CONTEXT_KEY holds an InferenceData, OpenTelemetryV2
    suppresses its own client span and metrics and enriches the outer InferenceData."""
    logger, span_exporter, metric_reader, log_exporter, *_ = _make_v2_harness(capture_span_content=True)

    outer_data = InferenceData(
        request_model="langchain-alias",
        capture_content=True,
    )
    token = attach(set_value(CLIENT_INFERENCE_CONTEXT_KEY, outer_data))
    try:
        call_id = "call-suppressed-v2"
        payload = _make_payload(call_id=call_id, model="gpt-4o", custom_llm_provider="openai")
        kwargs = {
            "litellm_call_id": call_id,
            "call_type": "completion",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello world"}],
            "optional_params": {
                "temperature": 0.5,
                "max_tokens": 100,
                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}],
            },
            "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
            "standard_logging_object": payload,
            "response_cost": 0.0015,
        }
        response_obj = {
            "id": "chatcmpl-xyz789",
            "model": "gpt-4o-2024-08-06",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        now = datetime.now(timezone.utc)

        logger.log_pre_api_call("gpt-4o", kwargs["messages"], kwargs)
        asyncio.run(
            logger.async_log_success_event(
                kwargs,
                response_obj,
                now,
                now + timedelta(milliseconds=250),
            )
        )
    finally:
        detach(token)

    # 1. No span, metrics, or events emitted by LiteLLM
    assert span_exporter.get_finished_spans() == ()
    assert _collected_metric_names(metric_reader) == []
    assert log_exporter.get_finished_logs() == ()

    # 2. Outer InferenceData is enriched via attributes and metric_attributes dicts
    assert outer_data.attributes.get(GenAI.PROVIDER_NAME) == "openai"
    assert outer_data.attributes.get(GenAI.OPERATION_NAME) == "chat"
    assert outer_data.attributes.get(GenAI.REQUEST_MODEL) == "gpt-4o"
    assert outer_data.attributes.get(GenAI.REQUEST_TEMPERATURE) == 0.5
    assert outer_data.attributes.get(GenAI.REQUEST_MAX_TOKENS) == 100
    assert outer_data.attributes.get(GenAI.RESPONSE_ID) == "chatcmpl-xyz789"
    assert list(outer_data.attributes.get(GenAI.RESPONSE_FINISH_REASONS) or []) == ["stop"]
    assert outer_data.attributes.get(GenAI.USAGE_INPUT_TOKENS) == 12
    assert outer_data.attributes.get(GenAI.USAGE_OUTPUT_TOKENS) == 8
    assert outer_data.attributes.get(Server.ADDRESS) == "api.openai.com"
    assert outer_data.attributes.get(Server.PORT) == 443

    # Custom LiteLLM attributes published to outer_data.attributes
    assert outer_data.attributes.get(LiteLLM.CALL_ID) == call_id
    assert outer_data.attributes.get(LiteLLM.CACHE_HIT) is False

    # Metric attributes published to outer_data.metric_attributes
    assert outer_data.metric_attributes.get(GenAI.PROVIDER_NAME) == "openai"
    assert outer_data.metric_attributes.get(GenAI.REQUEST_MODEL) == "gpt-4o"

    # Content capture attributes published into outer_data.attributes when outer capture_content=True
    assert outer_data.attributes.get(GenAI.INPUT_MESSAGES) is not None
    assert outer_data.attributes.get(GenAI.OUTPUT_MESSAGES) is not None


def test_v2_suppressed_respects_outer_content_capture_opt_out():
    """When LiteLLM has capture_span_content=True but the outer InferenceData has
    capture_content=False (default), prompt and response content attributes must not be copied."""
    logger, span_exporter, *_ = _make_v2_harness(capture_span_content=True)

    outer_data = InferenceData(request_model="langchain-alias", capture_content=False)
    token = attach(set_value(CLIENT_INFERENCE_CONTEXT_KEY, outer_data))
    try:
        call_id = "call-suppressed-no-content"
        payload = _make_payload(call_id=call_id, model="gpt-4o", custom_llm_provider="openai")
        kwargs = {
            "litellm_call_id": call_id,
            "call_type": "completion",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Secret user prompt"}],
            "optional_params": {"temperature": 0.5},
            "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
            "standard_logging_object": payload,
        }
        response_obj = {
            "id": "chatcmpl-xyz789",
            "model": "gpt-4o-2024-08-06",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Secret reply"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        now = datetime.now(timezone.utc)

        logger.log_pre_api_call("gpt-4o", kwargs["messages"], kwargs)
        asyncio.run(
            logger.async_log_success_event(
                kwargs,
                response_obj,
                now,
                now + timedelta(milliseconds=250),
            )
        )
    finally:
        detach(token)

    assert span_exporter.get_finished_spans() == ()
    assert outer_data.attributes.get(GenAI.PROVIDER_NAME) == "openai"
    assert GenAI.INPUT_MESSAGES not in outer_data.attributes
    assert GenAI.OUTPUT_MESSAGES not in outer_data.attributes
    assert "input.value" not in outer_data.attributes
    assert "output.value" not in outer_data.attributes
    assert "langfuse.observation.input" not in outer_data.attributes
    assert "langfuse.observation.output" not in outer_data.attributes


def test_v2_suppressed_failure_enriches_outer_inference_context():
    """On a failed LLM call when suppressed, OpenTelemetryV2 suppresses its error span,
    failure duration metric, and exception event, and publishes error details to outer_data."""
    logger, span_exporter, metric_reader, log_exporter, *_ = _make_v2_harness()

    outer_data = InferenceData()
    token = attach(set_value(CLIENT_INFERENCE_CONTEXT_KEY, outer_data))
    try:
        call_id = "call-fail-suppressed-v2"
        payload = _make_payload(
            call_id=call_id,
            model="gpt-4o",
            custom_llm_provider="openai",
            status="failure",
            error_str="Rate limit exceeded",
            error_information={
                "error_code": "429",
                "error_class": "RateLimitError",
                "error_message": "Rate limit exceeded",
                "llm_provider": "openai",
                "traceback": "Traceback ...",
            },
        )
        kwargs = {
            "litellm_call_id": call_id,
            "call_type": "completion",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
            "standard_logging_object": payload,
            "exception": RuntimeError("Rate limit exceeded"),
        }
        now = datetime.now(timezone.utc)

        logger.log_pre_api_call("gpt-4o", kwargs["messages"], kwargs)
        asyncio.run(
            logger.async_log_failure_event(
                kwargs,
                None,
                now,
                now + timedelta(milliseconds=100),
            )
        )
    finally:
        detach(token)

    assert span_exporter.get_finished_spans() == ()
    assert _collected_metric_names(metric_reader) == []
    assert log_exporter.get_finished_logs() == ()

    assert outer_data.attributes.get(Error.TYPE) == "RateLimitError"
    assert outer_data.attributes.get(Error.MESSAGE) == "Rate limit exceeded"


def test_v2_suppressed_with_real_opentelemetry_util_genai_handler():
    """End-to-end test with opentelemetry.util.genai.handler.TelemetryHandler as the outer layer."""
    util_genai = pytest.importorskip("opentelemetry.util.genai.handler")

    (
        logger,
        span_exporter,
        _metric_reader,
        _log_exporter,
        tracer_provider,
        meter_provider,
        logger_provider,
    ) = _make_v2_harness(capture_span_content=True)

    handler = util_genai.TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )

    call_id = "call-e2e-outer-handler"
    payload = _make_payload(
        call_id=call_id,
        model="claude-3-5-sonnet",
        custom_llm_provider="anthropic",
        prompt_tokens=15,
        completion_tokens=9,
    )
    kwargs = {
        "litellm_call_id": call_id,
        "call_type": "completion",
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello Claude"}],
        "optional_params": {"temperature": 0.2},
        "litellm_params": {"custom_llm_provider": "anthropic", "metadata": {}},
        "standard_logging_object": payload,
        "response_cost": 0.003,
    }
    response_obj = {
        "id": "msg-01",
        "model": "claude-3-5-sonnet-20241022",
        "choices": [{"finish_reason": "end_turn", "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 9, "total_tokens": 24},
    }
    now = datetime.now(timezone.utc)

    with handler.inference(None, request_model="outer-model"):
        logger.log_pre_api_call("claude-3-5-sonnet", kwargs["messages"], kwargs)
        asyncio.run(
            logger.async_log_success_event(
                kwargs,
                response_obj,
                now,
                now + timedelta(milliseconds=200),
            )
        )

    spans = span_exporter.get_finished_spans()
    # Exactly ONE client span is emitted (by the outer TelemetryHandler, enriched by LiteLLM)
    assert len(spans) == 1
    outer_span = spans[0]
    attrs = dict(outer_span.attributes or {})
    assert attrs.get(GenAI.PROVIDER_NAME) == "anthropic"
    assert attrs.get(GenAI.USAGE_INPUT_TOKENS) == 15
    assert attrs.get(GenAI.USAGE_OUTPUT_TOKENS) == 9
    assert attrs.get(LiteLLM.CALL_ID) == call_id


def test_v2_embedding_context_suppression():
    """Embedding calls use CLIENT_EMBEDDING_CONTEXT_KEY and EmbeddingData."""
    logger, span_exporter, metric_reader, *_ = _make_v2_harness()

    outer_emb = EmbeddingData(request_model="text-embedding-3-small")
    token = attach(set_value(CLIENT_EMBEDDING_CONTEXT_KEY, outer_emb))
    try:
        call_id = "call-emb-suppressed"
        payload = _make_payload(
            call_id=call_id,
            call_type="embedding",
            model="text-embedding-3-small",
            custom_llm_provider="openai",
            prompt_tokens=10,
            completion_tokens=0,
        )
        kwargs = {
            "litellm_call_id": call_id,
            "call_type": "embedding",
            "model": "text-embedding-3-small",
            "messages": ["embed me"],
            "optional_params": {"encoding_format": "float"},
            "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
            "standard_logging_object": payload,
        }
        now = datetime.now(timezone.utc)
        logger.log_pre_api_call("text-embedding-3-small", kwargs["messages"], kwargs)
        asyncio.run(
            logger.async_log_success_event(
                kwargs,
                None,
                now,
                now + timedelta(milliseconds=100),
            )
        )
    finally:
        detach(token)

    assert span_exporter.get_finished_spans() == ()
    assert _collected_metric_names(metric_reader) == []
    assert outer_emb.attributes.get(GenAI.PROVIDER_NAME) == "openai"
    assert outer_emb.attributes.get(GenAI.REQUEST_MODEL) == "text-embedding-3-small"
    assert outer_emb.attributes.get(GenAI.USAGE_INPUT_TOKENS) == 10
