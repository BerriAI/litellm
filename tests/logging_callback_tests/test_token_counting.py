import os
import sys
import traceback
from litellm._uuid import uuid
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
from litellm.types.utils import StandardLoggingPayload, Usage, ModelInfoBase
from litellm.integrations.custom_logger import CustomLogger


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.recorded_usage: Optional[Usage] = None
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload = kwargs.get("standard_logging_object")
        self.standard_logging_payload = standard_logging_payload
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


@pytest.mark.asyncio
async def test_stream_token_counting_anthropic_with_include_usage():
    """ """
    from anthropic import Anthropic

    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    litellm._turn_on_debug()

    custom_logger = TestCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)

    input_text = "Respond in just 1 word. Say ping"

    response = await litellm.acompletion(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=4096,
        stream=True,
    )

    actual_usage = None
    output_text = ""
    async for chunk in response:
        output_text += chunk["choices"][0]["delta"]["content"] or ""
        pass

    await asyncio.sleep(1)

    print("\n\n\n\n\n")
    print(
        "recorded_usage",
        json.dumps(custom_logger.recorded_usage, indent=4, default=str),
    )
    print("\n\n\n\n\n")

    # print making the same request with anthropic client
    anthropic_response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        messages=[{"role": "user", "content": input_text}],
        stream=True,
    )
    usage = None
    all_anthropic_usage_chunks = []
    for chunk in anthropic_response:
        print("chunk", json.dumps(chunk, indent=4, default=str))
        if hasattr(chunk, "message"):
            if chunk.message.usage:
                print(
                    "USAGE BLOCK",
                    json.dumps(chunk.message.usage, indent=4, default=str),
                )
                all_anthropic_usage_chunks.append(chunk.message.usage)
        elif hasattr(chunk, "usage"):
            print("USAGE BLOCK", json.dumps(chunk.usage, indent=4, default=str))
            all_anthropic_usage_chunks.append(chunk.usage)

    print(
        "all_anthropic_usage_chunks",
        json.dumps(all_anthropic_usage_chunks, indent=4, default=str),
    )

    # Get the most recent value of input tokens (iterate backwards to find last non-zero value)
    anthropic_api_input_tokens = 0
    for usage in reversed(all_anthropic_usage_chunks):
        if getattr(usage, "input_tokens", 0) > 0:
            anthropic_api_input_tokens = getattr(usage, "input_tokens", 0)
            break
    anthropic_api_output_tokens = 0
    for usage in reversed(all_anthropic_usage_chunks):
        if getattr(usage, "output_tokens", 0) > 0:
            anthropic_api_output_tokens = getattr(usage, "output_tokens", 0)
            break
    print("input_tokens_anthropic_api", anthropic_api_input_tokens)
    print("output_tokens_anthropic_api", anthropic_api_output_tokens)

    print("input_tokens_litellm", custom_logger.recorded_usage.prompt_tokens)
    print("output_tokens_litellm", custom_logger.recorded_usage.completion_tokens)

    ## Assert Accuracy of token counting
    # input tokens should be exactly the same
    assert anthropic_api_input_tokens == custom_logger.recorded_usage.prompt_tokens

    # output tokens can have at max abs diff of 10. We can't guarantee the response from two api calls will be exactly the same
    assert (
        abs(
            anthropic_api_output_tokens - custom_logger.recorded_usage.completion_tokens
        )
        <= 10
    )
