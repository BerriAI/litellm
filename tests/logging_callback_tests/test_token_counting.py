import os
import sys
import traceback
import uuid
import pytest
from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import io
import os
import time
import json

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import asyncio
from typing import Optional
from litellm.types.utils import StandardLoggingPayload, Usage
from litellm.integrations.custom_logger import CustomLogger


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.recorded_usage: Optional[Usage] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object")
        print(
            "standard_logging_payload",
            json.dumps(standard_logging_payload, indent=4, default=str),
        )

        self.recorded_usage = Usage(
            prompt_tokens=standard_logging_payload.get("prompt_tokens"),
            completion_tokens=standard_logging_payload.get("completion_tokens"),
            total_tokens=standard_logging_payload.get("total_tokens"),
        )
        pass


@pytest.mark.asyncio
async def test_stream_token_counting_gpt_4o():
    """
    When stream_options={"include_usage": True} logging callback tracks Usage == Usage from llm API
    """
    custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?" * 100}],
        stream=True,
        stream_options={"include_usage": True},
    )

    actual_usage = None
    async for chunk in response:
        if "usage" in chunk:
            actual_usage = chunk["usage"]
            print("chunk.usage", json.dumps(chunk["usage"], indent=4, default=str))
        pass

    await asyncio.sleep(2)

    print("\n\n\n\n\n")
    print(
        "recorded_usage",
        json.dumps(custom_logger.recorded_usage, indent=4, default=str),
    )
    print("\n\n\n\n\n")

    assert actual_usage.prompt_tokens == custom_logger.recorded_usage.prompt_tokens
    assert (
        actual_usage.completion_tokens == custom_logger.recorded_usage.completion_tokens
    )
    assert actual_usage.total_tokens == custom_logger.recorded_usage.total_tokens


@pytest.mark.asyncio
async def test_stream_token_counting_without_include_usage():
    """
    When stream_options={"include_usage": True} is not passed, the usage tracked == usage from llm api chunk

    by default, litellm passes `include_usage=True` for OpenAI API
    """
    custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?" * 100}],
        stream=True,
    )

    actual_usage = None
    async for chunk in response:
        if "usage" in chunk:
            actual_usage = chunk["usage"]
            print("chunk.usage", json.dumps(chunk["usage"], indent=4, default=str))
        pass

    await asyncio.sleep(2)

    print("\n\n\n\n\n")
    print(
        "recorded_usage",
        json.dumps(custom_logger.recorded_usage, indent=4, default=str),
    )
    print("\n\n\n\n\n")

    assert actual_usage.prompt_tokens == custom_logger.recorded_usage.prompt_tokens
    assert (
        actual_usage.completion_tokens == custom_logger.recorded_usage.completion_tokens
    )
    assert actual_usage.total_tokens == custom_logger.recorded_usage.total_tokens


@pytest.mark.asyncio
async def test_stream_token_counting_with_redaction():
    """
    When litellm.turn_off_message_logging=True is used, the usage tracked == usage from llm api chunk
    """
    litellm.turn_off_message_logging = True
    custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?" * 100}],
        stream=True,
    )

    actual_usage = None
    async for chunk in response:
        if "usage" in chunk:
            actual_usage = chunk["usage"]
            print("chunk.usage", json.dumps(chunk["usage"], indent=4, default=str))
        pass

    await asyncio.sleep(2)

    print("\n\n\n\n\n")
    print(
        "recorded_usage",
        json.dumps(custom_logger.recorded_usage, indent=4, default=str),
    )
    print("\n\n\n\n\n")

    assert actual_usage.prompt_tokens == custom_logger.recorded_usage.prompt_tokens
    assert (
        actual_usage.completion_tokens == custom_logger.recorded_usage.completion_tokens
    )
    assert actual_usage.total_tokens == custom_logger.recorded_usage.total_tokens
