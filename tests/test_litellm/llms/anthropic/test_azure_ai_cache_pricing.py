"""
Test that Azure AI Anthropic models have cache pricing configured.
Verifies the fix for issue #19532.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("../../../../../"))

import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
import pytest


@pytest.fixture(autouse=True)
def reload_model_costs():
    """Reload model costs from JSON before each test."""
    litellm.model_cost = get_model_cost_map(url=None)
    yield


@pytest.mark.parametrize(
    "model,expected_cache_creation_cost,expected_cache_read_cost",
    [
        ("claude-haiku-4-5", 1.25e-06, 1e-07),
        ("claude-opus-4-5", 6.25e-06, 5e-07),
        ("claude-opus-4-1", 1.875e-05, 1.5e-06),
        ("claude-sonnet-4-5", 3.75e-06, 3e-07),
    ],
)
def test_azure_ai_claude_cache_pricing(
    model, expected_cache_creation_cost, expected_cache_read_cost
):
    """Test that Azure AI Claude models have correct cache pricing."""
    model_info = get_model_info(model=model, custom_llm_provider="azure_ai")

    assert model_info.get("cache_creation_input_token_cost") is not None
    assert model_info.get("cache_read_input_token_cost") is not None
    assert model_info.get("cache_creation_input_token_cost") == expected_cache_creation_cost
    assert model_info.get("cache_read_input_token_cost") == expected_cache_read_cost
