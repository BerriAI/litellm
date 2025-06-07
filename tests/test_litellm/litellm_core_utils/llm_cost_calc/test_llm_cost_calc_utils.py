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


def test_anthropic_provider_cache_token_priority_over_generic():
    """
    Test that provider-specific cache tokens (_cache_read_input_tokens) 
    are prioritized over generic cached_tokens in prompt_tokens_details.
    
    Without the fix, this test would fail because it would use generic cache tokens
    instead of provider-specific ones.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    model = "claude-3-5-sonnet-20240620"
    custom_llm_provider = "anthropic"
    
    # Create usage object with BOTH provider-specific and generic cache tokens
    # The provider-specific ones should take priority
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=800,
            cached_tokens=50,  # Generic cache tokens (should be ignored)
        ),
        _cache_read_input_tokens=200,  # Provider-specific cache tokens (should be used)
        _cache_creation_input_tokens=0,
    )
    
    # Get the cost calculation
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    
    # Get model info for expected cost calculation
    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    
    # Calculate expected costs using provider-specific cache tokens (200, not 50)
    expected_text_tokens = usage.prompt_tokens - usage._cache_read_input_tokens  # 1000 - 200 = 800
    expected_prompt_cost = (
        expected_text_tokens * model_info["input_cost_per_token"] +
        usage._cache_read_input_tokens * model_info["cache_read_input_token_cost"]
    )
    expected_completion_cost = usage.completion_tokens * model_info["output_cost_per_token"]
    
    # Assert the costs match expectations (using provider-specific cache tokens)
    assert round(prompt_cost, 10) == round(expected_prompt_cost, 10), \
        f"Expected prompt cost {expected_prompt_cost}, got {prompt_cost}. Should use provider-specific cache tokens (200), not generic ones (50)."
    assert round(completion_cost, 10) == round(expected_completion_cost, 10), \
        f"Expected completion cost {expected_completion_cost}, got {completion_cost}"



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
    model = "gemini-2.5-flash-preview-04-17"
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
