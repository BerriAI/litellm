import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


def test_cost_calculator_with_response_cost_in_additional_headers():
    class MockResponse(BaseModel):
        _hidden_params = {
            "additional_headers": {"llm_provider-x-litellm-response-cost": 1000}
        }

    result = response_cost_calculator(
        response_object=MockResponse(),
        model="",
        custom_llm_provider=None,
        call_type="",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    assert result == 1000


def test_cost_calculator_with_usage():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10, audio_tokens=90
        ),
    )
    mr = ModelResponse(usage=usage, model="gemini-2.0-flash-001")

    result = response_cost_calculator(
        response_object=mr,
        model="",
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    model_info = litellm.model_cost["gemini-2.0-flash-001"]

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens
        * model_info["input_cost_per_audio_token"]
        + usage.prompt_tokens_details.text_tokens * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"


def test_handle_realtime_stream_cost_calculation():
    # Setup test data
    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-3.5-turbo"}},
        {
            "type": "response.done",
            "response": {
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            },
        },
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                }
            },
        },
    ]

    # Test with explicit model name
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost
    # gpt-3.5-turbo costs: $0.0015/1K tokens input, $0.002/1K tokens output
    expected_cost = (300 * 0.0015 / 1000) + (  # input tokens (100 + 200)
        150 * 0.002 / 1000
    )  # output tokens (50 + 100)
    assert (
        abs(cost - expected_cost) <= 0.00075
    )  # Allow small floating point differences

    # Test with different model name in session
    results[0]["session"]["model"] = "gpt-4"
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost using gpt-4 rates
    # gpt-4 costs: $0.03/1K tokens input, $0.06/1K tokens output
    expected_cost = (300 * 0.03 / 1000) + (  # input tokens
        150 * 0.06 / 1000
    )  # output tokens
    assert abs(cost - expected_cost) < 0.00076

    # Test with no response.done events
    results = [{"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}]
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )
    assert cost == 0.0  # No usage, no cost
