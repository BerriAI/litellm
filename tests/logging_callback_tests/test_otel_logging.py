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
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig, Span
import asyncio
import logging
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger
from litellm.proxy._types import SpanAttributes

verbose_logger.setLevel(logging.DEBUG)

EXPECTED_SPAN_NAMES = ["litellm_request", "raw_gen_ai_request"]
exporter = InMemorySpanExporter()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_async_otel_callback(streaming):
    litellm.set_verbose = True

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
        "llm.usage.total_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.prompt_tokens",
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


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-opus-20240229"],
)
@pytest.mark.flaky(retries=6, delay=2)
def test_completion_claude_3_function_call_with_otel(model):
    litellm.set_verbose = True

    litellm.callbacks = [OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter))]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            drop_params=True,
        )

        print("response from LiteLLM", response)
    except litellm.InternalServerError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        # clear in memory exporter
        exporter.clear()


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
        "llm.usage.total_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.prompt_tokens",
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

    # Check that any additional attributes are metadata fields (start with "metadata.")
    non_required_attrs = _all_attributes - required_set
    for attr in non_required_attrs:
        assert attr.startswith("metadata.") or attr.startswith(
            "hidden_params"
        ), f"Non-metadata attribute found: {attr}"

    pass
