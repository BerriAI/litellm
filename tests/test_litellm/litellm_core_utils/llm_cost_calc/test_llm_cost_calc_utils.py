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

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_cache_writing_cost,
    generic_cost_per_token,
)
from litellm.types.utils import CacheCreationTokenDetails, Usage


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


def test_image_tokens_with_custom_pricing():
    """Test that image_tokens in completion are properly costed with output_cost_per_image_token."""
    from unittest.mock import patch

    # Mock model info with image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_image_token": 5e-6,  # Custom pricing for image tokens in output
    }

    usage = Usage(
        completion_tokens=1720,  # text_tokens (600) + image_tokens (1120)
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 5e-6)
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 5e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_image_tokens_fallback_to_base_cost():
    """Test that image_tokens fall back to base cost when output_cost_per_image_token is not set."""
    from unittest.mock import patch

    # Mock model info without image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        # No output_cost_per_image_token defined
    }

    usage = Usage(
        completion_tokens=1720,
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 2e-6)  # image_tokens use base cost
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 2e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


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


def test_generic_cost_per_token_anthropic_prompt_caching_with_cache_creation():
    model = "claude-3-5-haiku-20241022"
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
        prompt_tokens_details=None,
        cache_creation_input_tokens=2000,
    )

    custom_llm_provider = "anthropic"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    print(f"prompt_cost: {prompt_cost}")
    assert round(prompt_cost, 3) == 0.023


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
            audio_tokens=100, cached_tokens=200, text_tokens=700, image_tokens=None, cache_creation_tokens=150
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=450,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
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


def test_calculate_cache_writing_cost():
    """Test the calculate_cache_writing_cost function with detailed cache creation token breakdown."""

    # Test case 1: With cache creation token details (matching the provided input)
    cache_creation_tokens = 14055
    cache_creation_token_details = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=56, ephemeral_1h_input_tokens=13999
    )
    cache_creation_cost_above_1hr = 6e-06
    cache_creation_cost = 3.75e-06

    result = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost,
    )

    # Expected calculation:
    # 5m tokens: 56 * 3.75e-06 = 0.00021
    # 1h tokens: 13999 * 6e-06 = 0.083994
    # Total: 0.00021 + 0.083994 = 0.084204
    expected_cost = (56 * 3.75e-06) + (13999 * 6e-06)

    assert round(result, 6) == round(expected_cost, 6)
    assert round(result, 6) == 0.084204

    # Test case 2: Without cache creation token details (fallback behavior)
    cache_creation_tokens_no_details = 1000
    cache_creation_token_details_none = None
    cache_creation_cost_fallback = 5e-06

    result_no_details = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens_no_details,
        cache_creation_token_details=cache_creation_token_details_none,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost_fallback,
    )

    # Expected calculation when no details: 1000 * 5e-06 = 0.005
    expected_cost_no_details = 1000 * 5e-06

    assert round(result_no_details, 6) == round(expected_cost_no_details, 6)
    assert result_no_details == 0.005

    # Test case 3: With cache creation token details but None values
    cache_creation_token_details_partial = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=None, ephemeral_1h_input_tokens=100
    )

    result_partial = calculate_cache_writing_cost(
        cache_creation_tokens=500,
        cache_creation_token_details=cache_creation_token_details_partial,
        cache_creation_cost_above_1hr=6e-06,
        cache_creation_cost=3e-06,
    )

    # Expected calculation: 0 (for None 5m tokens) + (100 * 6e-06) = 0.0006
    expected_cost_partial = (0.0) + (100 * 6e-06)

    assert round(result_partial, 6) == round(expected_cost_partial, 6)
    assert round(result_partial, 6) == 0.0006

    # Test case 4: Zero costs
    result_zero = calculate_cache_writing_cost(
        cache_creation_tokens=1000,
        cache_creation_token_details=CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=50, ephemeral_1h_input_tokens=950
        ),
        cache_creation_cost_above_1hr=0.0,
        cache_creation_cost=0.0,
    )

    assert result_zero == 0.0


def test_service_tier_flex_pricing():
    """Test that flex service tier uses correct pricing (approximately 50% of standard)."""
    # Set up environment for local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # Test with gpt-5-nano which has flex pricing
    model = "gpt-5-nano"
    custom_llm_provider = "openai"
    
    # Create usage object
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500
    )
    
    # Test standard pricing
    std_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None
    )
    std_total = std_cost[0] + std_cost[1]
    
    # Test flex pricing
    flex_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="flex"
    )
    flex_total = flex_cost[0] + flex_cost[1]
    
    # Verify flex is approximately 50% of standard
    assert std_total > 0, "Standard cost should be greater than 0"
    assert flex_total > 0, "Flex cost should be greater than 0"
    
    flex_ratio = flex_total / std_total
    assert 0.45 <= flex_ratio <= 0.55, f"Flex pricing should be ~50% of standard, got {flex_ratio:.2f}"
    
    # Verify specific costs match expected values
    # gpt-5-nano flex: input=2.5e-08, output=2e-07
    expected_flex_prompt = 1000 * 2.5e-08  # 0.000025
    expected_flex_completion = 500 * 2e-07  # 0.0001
    expected_flex_total = expected_flex_prompt + expected_flex_completion
    
    assert abs(flex_cost[0] - expected_flex_prompt) < 1e-10, f"Flex prompt cost mismatch: {flex_cost[0]} vs {expected_flex_prompt}"
    assert abs(flex_cost[1] - expected_flex_completion) < 1e-10, f"Flex completion cost mismatch: {flex_cost[1]} vs {expected_flex_completion}"
    assert abs(flex_total - expected_flex_total) < 1e-10, f"Flex total cost mismatch: {flex_total} vs {expected_flex_total}"


def test_service_tier_default_pricing():
    """Test that when no service tier is provided, standard pricing is used."""
    # Set up environment for local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # Test with gpt-5-nano
    model = "gpt-5-nano"
    custom_llm_provider = "openai"
    
    # Create usage object
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500
    )
    
    # Test with no service tier (should use standard)
    default_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None
    )
    
    # Test with explicit standard service tier
    standard_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="standard"
    )
    
    # Both should be identical
    assert abs(default_cost[0] - standard_cost[0]) < 1e-10, "Default and standard prompt costs should be identical"
    assert abs(default_cost[1] - standard_cost[1]) < 1e-10, "Default and standard completion costs should be identical"
    
    # Verify specific costs match expected standard values
    # gpt-5-nano standard: input=5e-08, output=4e-07
    expected_standard_prompt = 1000 * 5e-08  # 0.00005
    expected_standard_completion = 500 * 4e-07  # 0.0002
    expected_standard_total = expected_standard_prompt + expected_standard_completion
    
    assert abs(default_cost[0] - expected_standard_prompt) < 1e-10, f"Standard prompt cost mismatch: {default_cost[0]} vs {expected_standard_prompt}"
    assert abs(default_cost[1] - expected_standard_completion) < 1e-10, f"Standard completion cost mismatch: {default_cost[1]} vs {expected_standard_completion}"


def test_service_tier_fallback_pricing():
    """Test that when service tier is provided but model doesn't have those keys, it falls back to standard pricing."""
    # Set up environment for local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # Test with gpt-4 which doesn't have flex pricing keys
    model = "gpt-4"
    custom_llm_provider = "openai"
    
    # Create usage object
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500
    )
    
    # Test standard pricing
    std_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier=None
    )
    std_total = std_cost[0] + std_cost[1]
    
    # Test flex pricing (should fall back to standard since gpt-4 doesn't have flex keys)
    flex_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="flex"
    )
    flex_total = flex_cost[0] + flex_cost[1]
    
    # Test priority pricing (should fall back to standard since gpt-4 doesn't have priority keys)
    priority_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        service_tier="priority"
    )
    priority_total = priority_cost[0] + priority_cost[1]
    
    # All should be identical (fallback to standard)
    assert abs(std_total - flex_total) < 1e-10, f"Standard and flex costs should be identical (fallback): {std_total} vs {flex_total}"
    assert abs(std_total - priority_total) < 1e-10, f"Standard and priority costs should be identical (fallback): {std_total} vs {priority_total}"
    
    # Verify costs are reasonable (not zero)
    assert std_total > 0, "Standard cost should be greater than 0"
    assert flex_total > 0, "Flex cost should be greater than 0 (fallback)"
    assert priority_total > 0, "Priority cost should be greater than 0 (fallback)"
    
    # Verify specific costs match expected gpt-4 values
    # gpt-4 standard: input=3e-05, output=6e-05
    expected_standard_prompt = 1000 * 3e-05  # 0.03
    expected_standard_completion = 500 * 6e-05  # 0.03
    expected_standard_total = expected_standard_prompt + expected_standard_completion
    
    assert abs(std_cost[0] - expected_standard_prompt) < 1e-10, f"Standard prompt cost mismatch: {std_cost[0]} vs {expected_standard_prompt}"
    assert abs(std_cost[1] - expected_standard_completion) < 1e-10, f"Standard completion cost mismatch: {std_cost[1]} vs {expected_standard_completion}"


def test_bedrock_anthropic_prompt_caching():
    """Test Bedrock Anthropic models with prompt caching return correct costs."""
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    usage = Usage(
        prompt_tokens=52123,
        completion_tokens=497,
        total_tokens=52620,
        cache_creation_input_tokens=7183,
        cache_read_input_tokens=22465,
    )

    custom_llm_provider = "bedrock"

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert prompt_cost >= 0
    assert completion_cost >= 0
    assert round(prompt_cost, 3) == 0.111
    assert round(completion_cost, 5) == 0.00820
