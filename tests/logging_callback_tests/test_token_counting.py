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

    input_tokens_anthropic_api = sum(
        [getattr(usage, "input_tokens", 0) for usage in all_anthropic_usage_chunks]
    )
    output_tokens_anthropic_api = sum(
        [getattr(usage, "output_tokens", 0) for usage in all_anthropic_usage_chunks]
    )
    print("input_tokens_anthropic_api", input_tokens_anthropic_api)
    print("output_tokens_anthropic_api", output_tokens_anthropic_api)

    print("input_tokens_litellm", custom_logger.recorded_usage.prompt_tokens)
    print("output_tokens_litellm", custom_logger.recorded_usage.completion_tokens)

    ## Assert Accuracy of token counting
    # input tokens should be exactly the same
    assert input_tokens_anthropic_api == custom_logger.recorded_usage.prompt_tokens

    # output tokens can have at max abs diff of 10. We can't guarantee the response from two api calls will be exactly the same
    assert (
        abs(
            output_tokens_anthropic_api - custom_logger.recorded_usage.completion_tokens
        )
        <= 10
    )


async def _setup_web_search_test():
    """Helper function to setup common test requirements"""
    litellm._turn_on_debug()
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    return test_custom_logger


async def _verify_web_search_cost(test_custom_logger, expected_context_size):
    """Helper function to verify web search costs"""
    await asyncio.sleep(1)

    standard_logging_payload = test_custom_logger.standard_logging_payload
    response_cost = standard_logging_payload.get("response_cost")
    assert response_cost is not None

    # Calculate token cost
    model_map_information = standard_logging_payload["model_map_information"]
    model_map_value: ModelInfoBase = model_map_information["model_map_value"]
    total_token_cost = (
        standard_logging_payload["prompt_tokens"]
        * model_map_value["input_cost_per_token"]
    ) + (
        standard_logging_payload["completion_tokens"]
        * model_map_value["output_cost_per_token"]
    )

    # Verify total cost
    assert (
        response_cost
        == total_token_cost
        + model_map_value["search_context_cost_per_query"][expected_context_size]
    )


@pytest.mark.asyncio
async def test_openai_web_search_logging_cost_tracking_no_explicit_search_context_size():
    """Cost is tracked as `search_context_size_medium` when no `search_context_size` is passed in"""
    test_custom_logger = await _setup_web_search_test()

    response = await litellm.acompletion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {"role": "user", "content": "What was a positive news story from today?"}
        ],
    )

    await _verify_web_search_cost(test_custom_logger, "search_context_size_medium")


@pytest.mark.asyncio
async def test_openai_web_search_logging_cost_tracking_explicit_search_context_size():
    """search_context_size=low passed in, so cost tracked as `search_context_size_low`"""
    test_custom_logger = await _setup_web_search_test()

    response = await litellm.acompletion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {"role": "user", "content": "What was a positive news story from today?"}
        ],
        web_search_options={"search_context_size": "low"},
    )

    await _verify_web_search_cost(test_custom_logger, "search_context_size_low")


@pytest.mark.asyncio
async def test_openai_web_search_with_tool_call_logging_cost_tracking():
    """search_context_size=high passed in tool call, so cost tracked as `search_context_size_high`"""
    test_custom_logger = await _setup_web_search_test()

    response = await litellm.aresponses(
        model="openai/gpt-4o",
        input=[
            {"role": "user", "content": "What was a positive news story from today?"}
        ],
        tools=[{"type": "web_search_preview", "search_context_size": "high"}],
    )

    await _verify_web_search_cost(test_custom_logger, "search_context_size_high")
