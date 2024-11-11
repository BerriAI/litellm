# What is this?
## Unit tests for opentelemetry integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_opentelemetry_integration():
    """
    Unit test to confirm the parent otel span is ended
    """

    parent_otel_span = MagicMock()
    litellm.callbacks = ["otel"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Hey!",
        metadata={"litellm_parent_otel_span": parent_otel_span},
    )

    await asyncio.sleep(1)

    parent_otel_span.end.assert_called_once()


def test_parallel_tool_calls():
    from litellm.types.llms.openai import (
        ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk,
    )
    from litellm.integrations.opentelemetry import OpenTelemetry
    from litellm.proxy._types import SpanAttributes

    tool_calls = [
        ChatCompletionToolCallChunk(
            function=ChatCompletionToolCallFunctionChunk(
                arguments='{"city": "New York"}', name="get_weather"
            ),
            id="call_Gv7JsMgS7YRV3rEb5wYsI0fg",
            type="function",
        ),
        ChatCompletionToolCallChunk(
            function=ChatCompletionToolCallFunctionChunk(
                arguments='{"city": "New York"}', name="get_news"
            ),
            id="call_nqac3t38Sth3rThr71xyEARH",
            type="function",
        ),
    ]

    kv_pair_dict = OpenTelemetry._tool_calls_kv_pair(tool_calls)

    assert kv_pair_dict == {
        f"{SpanAttributes.LLM_COMPLETIONS}.0.function_call.arguments": '{"city": "New York"}',
        f"{SpanAttributes.LLM_COMPLETIONS}.0.function_call.name": "get_weather",
        f"{SpanAttributes.LLM_COMPLETIONS}.1.function_call.arguments": '{"city": "New York"}',
        f"{SpanAttributes.LLM_COMPLETIONS}.1.function_call.name": "get_news",
    }
