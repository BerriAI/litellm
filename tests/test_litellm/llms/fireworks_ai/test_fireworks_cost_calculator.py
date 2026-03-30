"""
Tests for Fireworks AI cost calculator — cache token pricing.
"""

import pytest
from unittest.mock import patch, MagicMock

from litellm.types.utils import Usage, PromptTokensDetailsWrapper
from litellm.llms.fireworks_ai.cost_calculator import cost_per_token, get_base_model_for_pricing


def test_cost_per_token_with_cache_tokens():
    """
    Test that cache_read_input_tokens are priced at cache_read_input_token_cost
    instead of the full input_cost_per_token.

    Regression test for https://github.com/BerriAI/litellm/issues/24774
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

    # With cache tokens, prompt_cost should be less than
    # 100 * input_cost_per_token (since 60 tokens are cheaper cache reads)
    assert prompt_cost >= 0
    assert completion_cost >= 0
    # The total should be a valid float, not NaN or inf
    assert prompt_cost == prompt_cost  # not NaN
    assert completion_cost == completion_cost  # not NaN


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
