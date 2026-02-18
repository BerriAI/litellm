import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
import litellm
import asyncio
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger
from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger
from litellm.integrations._types.open_inference import (
    OpenInferenceSpanKindValues,
    SpanAttributes as OISpanAttributes,
)
from litellm.integrations.opentelemetry import (
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    LITELLM_TRACER_NAME,
    LITELLM_REQUEST_SPAN_NAME,
    OpenTelemetry,
    OpenTelemetryConfig,
    RAW_REQUEST_SPAN_NAME,
    Span,
)
from litellm.proxy._types import SpanAttributes

verbose_logger.setLevel(logging.DEBUG)

EXPECTED_SPAN_NAMES = ["litellm_request", "raw_gen_ai_request"]
exporter = InMemorySpanExporter()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_async_otel_callback(streaming):
    litellm.set_verbose = True
    
    # Clear exporter at the start to ensure clean state
    exporter.clear()

    litellm.callbacks = [OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter))]

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        user="OTEL_USER",
        stream=streaming,
    )

    if streaming is True:
        async for chunk in response:
            print("chunk", chunk)

    await asyncio.sleep(4)
    spans = exporter.get_finished_spans()
    print("spans", spans)
    assert len(spans) == 2

    _span_names = [span.name for span in spans]
    print("recorded span names", _span_names)
    assert set(_span_names) == set(EXPECTED_SPAN_NAMES)

    # print the value of a span
    for span in spans:
        print("span name", span.name)
        print("span attributes", span.attributes)

        if span.name == "litellm_request":
            validate_litellm_request(span)
            # Additional specific checks
            assert span._attributes["gen_ai.request.model"] == "gpt-3.5-turbo"
            assert span._attributes["gen_ai.system"] == "openai"
            assert span._attributes["gen_ai.request.temperature"] == 0.1
            assert span._attributes["llm.is_streaming"] == str(streaming)
            assert span._attributes["llm.user"] == "OTEL_USER"
        elif span.name == "raw_gen_ai_request":
            if streaming is True:
                validate_raw_gen_ai_request_openai_streaming(span)
            else:
                validate_raw_gen_ai_request_openai_non_streaming(span)

    # clear in memory exporter
    exporter.clear()


def validate_litellm_request(span):
    expected_attributes = [
        "gen_ai.request.model",
        "gen_ai.system",
        "gen_ai.request.temperature",
        "llm.is_streaming",
        "llm.user",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.usage.total_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.input_tokens",
    ]

    # get the str of all the span attributes
    print("span attributes", span._attributes)

    for attr in expected_attributes:
        value = span._attributes[attr]
        print("value", value)
        assert value is not None, f"Attribute {attr} has None value"


def validate_raw_gen_ai_request_openai_non_streaming(span):
    expected_attributes = [
        "llm.openai.messages",
        "llm.openai.temperature",
        "llm.openai.user",
        "llm.openai.extra_body",
        "llm.openai.id",
        "llm.openai.choices",
        "llm.openai.created",
        "llm.openai.model",
        "llm.openai.object",
        "llm.openai.service_tier",
        "llm.openai.system_fingerprint",
        "llm.openai.usage",
    ]

    print("span attributes", span._attributes)
    for attr in span._attributes:
        print(attr)

    for attr in expected_attributes:
        assert span._attributes[attr] is not None, f"Attribute {attr} has None"


def validate_raw_gen_ai_request_openai_streaming(span):
    expected_attributes = [
        "llm.openai.messages",
        "llm.openai.temperature",
        "llm.openai.user",
        "llm.openai.extra_body",
        "llm.openai.model",
    ]

    print("span attributes", span._attributes)
    for attr in span._attributes:
        print(attr)

    for attr in expected_attributes:
        assert span._attributes[attr] is not None, f"Attribute {attr} has None"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("global_redact", [True, False])
async def test_awesome_otel_with_message_logging_off(streaming, global_redact):
    """
    No content should be logged when message logging is off

    tests when litellm.turn_off_message_logging is set to True
    tests when OpenTelemetry(message_logging=False) is set
    """
    litellm.set_verbose = True
    
    # Clear exporter at the start to ensure clean state
    exporter.clear()
    
    litellm.callbacks = [OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter))]
    if global_redact is False:
        otel_logger = OpenTelemetry(
            message_logging=False, config=OpenTelemetryConfig(exporter="console")
        )
    else:
        # use global redaction
        litellm.turn_off_message_logging = True
        otel_logger = OpenTelemetry(config=OpenTelemetryConfig(exporter="console"))

    litellm.callbacks = [otel_logger]
    litellm.success_callback = []
    litellm.failure_callback = []

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi",
        stream=streaming,
    )
    print("response", response)

    if streaming is True:
        async for chunk in response:
            print("chunk", chunk)

    await asyncio.sleep(1)
    spans = exporter.get_finished_spans()
    print("spans", spans)
    assert len(spans) == 1

    _span = spans[0]
    print("span attributes", _span.attributes)

    validate_redacted_message_span_attributes(_span)

    # clear in memory exporter
    exporter.clear()

    if global_redact is True:
        litellm.turn_off_message_logging = False


def validate_redacted_message_span_attributes(span):
    # Required non-metadata attributes that must be present
    required_attributes = [
        "gen_ai.request.model",
        "gen_ai.system",
        "llm.is_streaming",
        "llm.request.type",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.usage.total_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.input_tokens",
    ]

    _all_attributes = set(
        [
            name.value if isinstance(name, SpanAttributes) else name
            for name in span.attributes.keys()
        ]
    )
    print("all_attributes", _all_attributes)

    for attr in _all_attributes:
        print(f"attr: {attr}, type: {type(attr)}")

    # Check that all required attributes are present
    required_set = set(required_attributes)
    assert required_set.issubset(
        _all_attributes
    ), f"Missing required attributes: {required_set - _all_attributes}"

    # Check that any additional attributes are metadata fields (start with "metadata.") or cost fields
    non_required_attrs = _all_attributes - required_set
    for attr in non_required_attrs:
        assert (
            attr.startswith("metadata.")
            or attr.startswith("hidden_params")
            or attr.startswith("gen_ai.cost.")
            or attr.startswith("gen_ai.operation.")
            or attr.startswith("gen_ai.request.")
        ), f"Non-metadata attribute found: {attr}"

    pass

@pytest.mark.asyncio
async def test_arize_phoenix_adds_openinference_kind_and_avoids_duplicate_litellm_spans():
    """
    Ensure Arize Phoenix spans include OpenInference span kind and do not create
    a duplicate litellm_request span when a proxy parent span is already active.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter.clear()
    litellm.logging_callback_manager._reset_all_callbacks()

    # Set up a global TracerProvider so we can create valid spans
    # This simulates the proxy server's TracerProvider
    global_provider = TracerProvider()
    global_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(global_provider)

    otel_logger = ArizePhoenixLogger(config=OpenTelemetryConfig(exporter=exporter))
    litellm.callbacks = [otel_logger]
    litellm.success_callback = []
    litellm.failure_callback = []

    tracer = trace.get_tracer(LITELLM_TRACER_NAME)
    parent_span = tracer.start_span(LITELLM_PROXY_REQUEST_SPAN_NAME)

    # Keep parent span active; OpenTelemetry logger will attach attributes and end it.
    with trace.use_span(parent_span, end_on_exit=False):
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "ping"}],
            mock_response="pong",
        )

    # Flush span processing
    await asyncio.sleep(1)

    if parent_span.is_recording():
        parent_span.end()

    spans = exporter.get_finished_spans()

    span_names = [span.name for span in spans]
    assert LITELLM_REQUEST_SPAN_NAME not in span_names
    assert span_names.count(LITELLM_PROXY_REQUEST_SPAN_NAME) == 1
    assert span_names.count(RAW_REQUEST_SPAN_NAME) == 1

    # All spans should belong to the same trace (parent + raw child)
    assert len({span.context.trace_id for span in spans}) == 1
    assert len(spans) == 2

    proxy_span = next(span for span in spans if span.name == LITELLM_PROXY_REQUEST_SPAN_NAME)
    assert proxy_span.attributes.get(OISpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value

    exporter.clear()
