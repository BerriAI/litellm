import pytest
import litellm
import asyncio
import logging
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry import (
    OpenTelemetry,
    OpenTelemetryConfig,
)

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
        model="gpt-4.1-mini",
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
            assert span._attributes["gen_ai.request.model"] == "gpt-4.1-mini"
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
