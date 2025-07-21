import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.types.llms.openai import FileSearchTool, WebSearchOptions
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelInfo,
    ModelResponse,
    PromptTokensDetailsWrapper,
    StandardBuiltInToolsParams,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def test_reasoning_tokens_no_price_set():
    model = "o1-mini"
    custom_llm_provider = "openai"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model_cost_map = litellm.model_cost[model]
    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    print(f"completion_cost: {completion_cost}")
    expected_completion_cost = (
        model_cost_map["output_cost_per_token"] * usage.completion_tokens
    )
    print(f"expected_completion_cost: {expected_completion_cost}")
    assert round(completion_cost, 10) == round(
        expected_completion_cost,
        10,
    )


def test_reasoning_tokens_gemini():
    model = "gemini-2.5-flash"
    custom_llm_provider = "gemini"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        (
            model_cost_map["output_cost_per_token"]
            * usage.completion_tokens_details.text_tokens
        )
        + (
            model_cost_map["output_cost_per_reasoning_token"]
            * usage.completion_tokens_details.reasoning_tokens
        ),
        10,
    )


def test_generic_cost_per_token_above_200k_tokens():
    model = "gemini-2.5-pro-exp-03-25"
    custom_llm_provider = "vertex_ai"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 220 * 1e6
    completion_tokens = 150
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token_above_200k_tokens"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        model_cost_map["output_cost_per_token_above_200k_tokens"]
        * usage.completion_tokens,
        10,
    )


def test_generic_cost_per_token_anthropic_prompt_caching():
    model = "claude-sonnet-4@20250514"
    usage = Usage(
        completion_tokens=90,
        prompt_tokens=28436,
        total_tokens=28526,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=0, text_tokens=None, image_tokens=None
        ),
        cache_creation_input_tokens=118,
        cache_read_input_tokens=28432,
    )

    custom_llm_provider = "vertex_ai"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    print(f"prompt_cost: {prompt_cost}")
    assert prompt_cost < 0.085


def test_string_cost_values():
    """Test that cost values defined as strings are properly converted to floats."""
    from unittest.mock import patch

    # Mock model info with string cost values (as might be read from config.yaml)
    mock_model_info = {
        "input_cost_per_token": "3e-7",  # String representation of scientific notation
        "output_cost_per_token": "6e-7",  # String representation of scientific notation
        "input_cost_per_audio_token": "0.000001",  # String representation of decimal
        "output_cost_per_audio_token": "0.000002",  # String representation of decimal
        "cache_read_input_token_cost": "1.5e-8",  # String representation of scientific notation
        "cache_creation_input_token_cost": "2.5e-8",  # String representation of scientific notation
    }

    # Test usage with various token types
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100, cached_tokens=200, text_tokens=700, image_tokens=None
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=450,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
        _cache_creation_input_tokens=150,
    )

    # Mock get_model_info to return our mock model info
    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Calculate expected costs manually
    # Prompt cost = text_tokens * input_cost + audio_tokens * audio_cost + cached_tokens * cache_read_cost + cache_creation_tokens * cache_creation_cost
    expected_prompt_cost = (
        700 * 3e-7  # text tokens
        + 100 * 1e-6  # audio tokens
        + 200 * 1.5e-8  # cached tokens
        + 150 * 2.5e-8  # cache creation tokens
    )

    # Completion cost = text_tokens * output_cost + audio_tokens * audio_output_cost
    expected_completion_cost = 450 * 6e-7 + 50 * 2e-6  # text tokens  # audio tokens

    # Assert costs are calculated correctly
    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_calculate_cost_component_with_string_values():
    """Test the calculate_cost_component function directly with string cost values."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import calculate_cost_component

    # Test with valid string scientific notation
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with valid string decimal notation
    model_info = {"output_cost_per_token": "0.000001"}
    cost = calculate_cost_component(model_info, "output_cost_per_token", 500)
    assert cost == 500 * 0.000001

    # Test with float value (should work as before)
    model_info = {"input_cost_per_token": 3e-7}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with invalid string value (should return 0.0)
    model_info = {"input_cost_per_token": "invalid_number"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with None value (should return 0.0)
    model_info = {"input_cost_per_token": None}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with missing key (should return 0.0)
    model_info = {}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with zero usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 0)
    assert cost == 0.0

    # Test with None usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", None)
    assert cost == 0.0


def test_string_cost_values_edge_cases():
    """Test edge cases for string cost value handling."""
    from unittest.mock import patch

    # Test with mixed string and float cost values
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String
        "output_cost_per_token": 2e-6,  # Float
        "input_cost_per_audio_token": "invalid",  # Invalid string
        "output_cost_per_audio_token": None,  # None value
    }

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100, cached_tokens=0, text_tokens=1000, image_tokens=None
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=500,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs:
    # Prompt: 1000 * 1e-6 + 100 * 0 (invalid string becomes 0)
    # Completion: 500 * 2e-6 (text_tokens == completion_tokens, so is_text_tokens_total=True, no separate audio cost)
    expected_prompt_cost = 1000 * 1e-6
    expected_completion_cost = 500 * 2e-6

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_string_cost_values_with_threshold():
    """Test that string cost values work correctly with threshold pricing."""
    from unittest.mock import patch

    # Mock model info with string cost values including threshold pricing
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String base cost
        "output_cost_per_token": "2e-6",  # String base cost
        "input_cost_per_token_above_200k_tokens": "5e-7",  # String threshold cost (lower)
        "output_cost_per_token_above_200k_tokens": "1e-6",  # String threshold cost (lower)
    }

    # Test usage above threshold
    usage = Usage(
        prompt_tokens=250000,  # Above 200k threshold
        completion_tokens=1000,
        total_tokens=251000,
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs using threshold pricing (string values converted to float)
    expected_prompt_cost = 250000 * 5e-7  # threshold cost
    expected_completion_cost = 1000 * 1e-6  # threshold cost

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)
