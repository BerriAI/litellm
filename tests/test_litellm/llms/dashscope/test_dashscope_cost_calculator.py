"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- Correctly calculates graduated tiered pricing.
- Falls back to flat-rate pricing for non-tiered models.
- Handles interactions with cached tokens.
- Correctly calculates costs for token counts exceeding the highest defined tier.
"""

import json
import math
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.dashscope.cost_calculator import (
    cost_per_token as dashscope_cost_per_token,
)
from litellm.types.utils import Usage, PromptTokensDetailsWrapper


class TestDashscopeCostCalculator:
    """Test suite for Dashscope cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing by loading it locally."""
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_dashscope_flat_pricing_fallback(self):
        """
        Tests that the dashscope calculator falls back to flat pricing for models
        without a 'tiered_pricing' key (e.g., qwen-max).
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        # We call the specific calculator for dashscope
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-max", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-max")
        expected_prompt_cost = 1000 * model_info["input_cost_per_token"]
        expected_completion_cost = 500 * model_info["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_within_first_tier(self):
        """
        Tests the dashscope tiered pricing when token count is entirely within the first tier.
        Uses 'dashscope/qwen-flash' as a real-world example.
        """
        # Tier 1 for qwen-flash is [0, 256,000] tokens
        usage = Usage(prompt_tokens=100000, completion_tokens=50000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1_pricing = model_info["tiered_pricing"][0]

        expected_prompt_cost = 100000 * tier_1_pricing["input_cost_per_token"]
        expected_completion_cost = 50000 * tier_1_pricing["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_spanning_multiple_tiers(self):
        """
        Tests the dashscope tiered pricing with the corrected graduated calculation logic.
        This is the most important test for validating the fix.
        """
        # Tiering for qwen-flash: Tier 1: [0, 256k], Tier 2: [256k, 1M]
        usage = Usage(prompt_tokens=300000, completion_tokens=300000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        # Expected prompt cost: (256,000 tokens * tier_1_price) + (44,000 tokens * tier_2_price)
        expected_prompt_cost = (256000 * tier_1["input_cost_per_token"]) + (
            44000 * tier_2["input_cost_per_token"]
        )

        # Expected completion cost: (256,000 tokens * tier_1_price) + (44,000 tokens * tier_2_price)
        expected_completion_cost = (256000 * tier_1["output_cost_per_token"]) + (
            44000 * tier_2["output_cost_per_token"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Tests tiered pricing with cached tokens. This replaces the old, incorrect test.
        Uses qwen3-coder-plus, which has cache-specific pricing defined.
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        # 10k cached tokens are all in the first tier
        expected_cache_cost = 10000 * tier_1["cache_read_input_token_cost"]

        # 40k new tokens: 32k in tier 1, and the remaining 8k in tier 2
        expected_text_cost = (32000 * tier_1["input_cost_per_token"]) + (
            8000 * tier_2["input_cost_per_token"]
        )

        expected_total_prompt_cost = expected_cache_cost + expected_text_cost

        assert math.isclose(prompt_cost, expected_total_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Tests tiered pricing when token count exceeds the highest defined tier range.
        This replaces the old, incorrect test and validates the new fallback logic.
        """
        usage = Usage(
            prompt_tokens=1200000, completion_tokens=1000
        )  # Max defined range for qwen-flash is 1M

        prompt_cost, _ = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        # Expected cost: (tier_1_tokens * tier_1_price) + (tokens_up_to_max_range_in_tier_2 * tier_2_price) + (remaining_tokens * tier_2_price)
        tokens_in_tier_2_range = 1000000 - 256000
        remaining_tokens_over_max = 1200000 - 1000000

        expected_prompt_cost = (
            (256000 * tier_1["input_cost_per_token"])
            + (tokens_in_tier_2_range * tier_2["input_cost_per_token"])
            + (remaining_tokens_over_max * tier_2["input_cost_per_token"])
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
