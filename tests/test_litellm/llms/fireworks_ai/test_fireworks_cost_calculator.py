"""
Tests for Fireworks AI cost calculator — cache token pricing.
"""

import pytest

from litellm.types.utils import Usage
from litellm.llms.fireworks_ai.cost_calculator import (
    cost_per_token,
    get_base_model_for_pricing,
)
from litellm.utils import get_model_info


# ---------------------------------------------------------------------------
# NOTE: unused imports (patch, MagicMock, PromptTokensDetailsWrapper) that
# were present in the initial commit have been removed per greptile review.
# ---------------------------------------------------------------------------


def test_cost_per_token_with_cache_tokens():
    """
    Test that cache_read_input_tokens are priced at cache_read_input_token_cost
    instead of the full input_cost_per_token.

    Regression test for https://github.com/BerriAI/litellm/issues/24774

    For kimi-k2p5:
        input_cost_per_token = 6e-7
        cache_read_input_token_cost = 1e-7

    With 100 prompt tokens (60 cache read, 40 non-cached):
        Correct cost  = 40 * 6e-7 + 60 * 1e-7 = 3e-5
        Buggy cost    = 100 * 6e-7             = 6e-5
    """
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cache_read_input_tokens=60,
        cache_creation_input_tokens=0,
    )

    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/kimi-k2p5", usage=usage
    )

    # Verify the cache discount is applied
    model_info = get_model_info(
        model="accounts/fireworks/models/kimi-k2p5",
        custom_llm_provider="fireworks_ai",
    )
    input_cost = model_info["input_cost_per_token"]
    cache_read_cost = model_info.get("cache_read_input_token_cost", input_cost)

    non_cached_tokens = 100 - 60  # prompt_tokens - cache_read_input_tokens
    expected_prompt_cost = non_cached_tokens * input_cost + 60 * cache_read_cost

    assert prompt_cost == pytest.approx(expected_prompt_cost), (
        f"Cache discount not applied: got {prompt_cost}, "
        f"expected {expected_prompt_cost}"
    )
    # Verify that cache discount actually reduces cost vs full-price calculation
    full_price_prompt_cost = 100 * input_cost
    assert prompt_cost < full_price_prompt_cost, (
        f"Cache pricing should be cheaper than full price: "
        f"got {prompt_cost}, full price would be {full_price_prompt_cost}"
    )
    assert completion_cost >= 0


def test_cost_per_token_without_cache_tokens():
    """
    Test basic cost calculation without any cache tokens.
    """
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct", usage=usage
    )

    assert prompt_cost >= 0
    assert completion_cost >= 0


def test_cost_per_token_unmapped_model_falls_back():
    """
    Test that an unmapped model name falls back to parameter-based pricing.
    """
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    # This model name should trigger the fallback path
    prompt_cost, completion_cost = cost_per_token(
        model="accounts/fireworks/models/some-custom-7b-model", usage=usage
    )

    assert prompt_cost >= 0
    assert completion_cost >= 0


def test_get_base_model_for_pricing_moe():
    """Test MoE model parameter extraction."""
    assert get_base_model_for_pricing("mixtral-8x7b") == "fireworks-ai-moe-up-to-56b"


def test_get_base_model_for_pricing_standard():
    """Test standard model parameter extraction."""
    assert get_base_model_for_pricing("llama-3b") == "fireworks-ai-up-to-4b"
    assert get_base_model_for_pricing("llama-8b") == "fireworks-ai-4.1b-to-16b"
    assert get_base_model_for_pricing("llama-70b") == "fireworks-ai-above-16b"


def test_get_base_model_for_pricing_unknown():
    """Test unknown model returns default."""
    assert get_base_model_for_pricing("custom-model") == "fireworks-ai-default"
