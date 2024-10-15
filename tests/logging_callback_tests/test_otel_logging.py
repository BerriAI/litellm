import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest
import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig, Span
import asyncio
import logging
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger


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
            assert span.attributes["gen_ai.request.model"] == "gpt-3.5-turbo"
            assert span.attributes["gen_ai.system"] == "openai"
            assert span.attributes["gen_ai.request.temperature"] == 0.1
            assert span.attributes["llm.is_streaming"] == str(streaming)
            assert span.attributes["llm.user"] == "OTEL_USER"
            assert span.attributes["gen_ai.prompt.0.content"] == "hi"
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
        "gen_ai.prompt.0.role",
        "gen_ai.prompt.0.content",
        "gen_ai.completion.0.finish_reason",
        "gen_ai.completion.0.role",
        "gen_ai.completion.0.content",
    ]

    for attr in expected_attributes:
        assert attr in span.attributes, f"Attribute {attr} is missing"
        assert span.attributes[attr] is not None, f"Attribute {attr} has None value"


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

    for attr in expected_attributes:
        assert attr in span.attributes, f"Attribute {attr} is missing"
        assert span.attributes[attr] is not None, f"Attribute {attr} has None"


def validate_raw_gen_ai_request_openai_streaming(span):
    expected_attributes = [
        "llm.openai.messages",
        "llm.openai.temperature",
        "llm.openai.user",
        "llm.openai.extra_body",
        "llm.openai.model",
    ]

    for attr in expected_attributes:
        assert attr in span.attributes, f"Attribute {attr} is missing"
        assert span.attributes[attr] is not None, f"Attribute {attr} has None"


# @pytest.mark.parametrize(
#     "model",
#     ["anthropic/claude-3-opus-20240229"],
# )
# def test_completion_claude_3_function_call_with_otel(model):
#     litellm.set_verbose = True

#     litellm.callbacks = [OpenTelemetry(OpenTelemetryConfig())]
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "get_current_weather",
#                 "description": "Get the current weather in a given location",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "location": {
#                             "type": "string",
#                             "description": "The city and state, e.g. San Francisco, CA",
#                         },
#                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                     },
#                     "required": ["location"],
#                 },
#             },
#         }
#     ]
#     messages = [
#         {
#             "role": "user",
#             "content": "What's the weather like in Boston today in Fahrenheit?",
#         }
#     ]
#     try:
#         # test without max tokens
#         response = litellm.completion(
#             model=model,
#             messages=messages,
#             tools=tools,
#             tool_choice={
#                 "type": "function",
#                 "function": {"name": "get_current_weather"},
#             },
#             drop_params=True,
#         )

#         print("response from LiteLLM", response)

#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# @pytest.mark.asyncio
# async def test_awesome_otel_with_message_logging_off():
#     litellm.set_verbose = True

#     otel_logger = OpenTelemetry(
#         message_logging=False, config=OpenTelemetryConfig(exporter="console")
#     )

#     litellm.callbacks = [otel_logger]
#     litellm.success_callback = []
#     litellm.failure_callback = []

#     response = await litellm.acompletion(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": "hi"}],
#         mock_response="hi",
#     )
#     print("response", response)

#     await asyncio.sleep(5)
