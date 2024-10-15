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
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
import asyncio
import logging
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from litellm._logging import verbose_logger


verbose_logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_async_otel_callback():
    exporter = InMemorySpanExporter()
    litellm.set_verbose = True

    os.environ["OTEL_EXPORTER"] = "in_memory"

    litellm.callbacks = [OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter))]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(4)
    spans = exporter.get_finished_spans()
    print("spans", spans)
    assert len(spans) == 2


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
