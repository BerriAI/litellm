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
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import StandardBuiltInToolCostTracking


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
    response = standard_logging_payload.get("response")
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
    if StandardBuiltInToolCostTracking.response_object_includes_web_search_call(response):
        assert (
            response_cost
            == total_token_cost
            + model_map_value["search_context_cost_per_query"][expected_context_size]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "web_search_options,expected_context_size",
    [
        (None, "search_context_size_medium"),
        ({"search_context_size": "low"}, "search_context_size_low"),
        ({"search_context_size": "high"}, "search_context_size_high"),
    ],
)
async def test_openai_web_search_logging_cost_tracking(
    web_search_options, expected_context_size
):
    """Test web search cost tracking with different search context sizes"""
    test_custom_logger = await _setup_web_search_test()
    from litellm._uuid import uuid

    

    request_kwargs = {
        "model": "openai/gpt-4o-search-preview",
        "messages": [
            {"role": "user", "content": f"What was a positive news story from today? {uuid.uuid4()}"}
        ],
    }
    if web_search_options is not None:
        request_kwargs["web_search_options"] = web_search_options

    response = await litellm.acompletion(**request_kwargs)

    await _verify_web_search_cost(test_custom_logger, expected_context_size)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools_config,expected_context_size,stream",
    [
        (
            [{"type": "web_search_preview", "search_context_size": "low"}],
            "search_context_size_low",
            True,
        ),
        (
            [{"type": "web_search_preview", "search_context_size": "low"}],
            "search_context_size_low",
            False,
        ),
        ([{"type": "web_search_preview"}], "search_context_size_medium", True),
        ([{"type": "web_search_preview"}], "search_context_size_medium", False),
    ],
)
async def test_openai_responses_api_web_search_cost_tracking(
    tools_config, expected_context_size, stream
):
    """Test web search cost tracking with different search context sizes and streaming options"""
    test_custom_logger = await _setup_web_search_test()

    response = await litellm.aresponses(
        model="openai/gpt-4o",
        input=[
            {"role": "user", "content": "What was a positive news story from today?"}
        ],
        tools=tools_config,
        stream=stream,
    )
    if stream is True:
        async for chunk in response:
            print("chunk", chunk)
    else:
        print("response", response)

    await asyncio.sleep(1)

    if StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
        test_custom_logger.standard_logging_payload.get("response")
    ):
        await _verify_web_search_cost(test_custom_logger, expected_context_size)
