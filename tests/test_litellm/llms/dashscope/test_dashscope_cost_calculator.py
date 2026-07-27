"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- Selects one pricing tier from the request's total input tokens.
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
        Tests the dashscope tiered pricing when the request's input size falls in the
        first tier. Uses 'dashscope/qwen-flash' as a real-world example.
        """
        # Tier 1 for qwen-flash is [0, 256,000] input tokens
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

    def test_dashscope_input_above_first_tier_bills_whole_request_at_that_tier(self):
        """
        Regression for the graduated-slicing bug: Model Studio selects one tier from the
        request's total input tokens and bills every token in the request at that tier,
        instead of charging the first 256k input tokens at the tier 1 rate.
        """
        # Tiering for qwen-flash: Tier 1: [0, 256k], Tier 2: [256k, 1M]
        usage = Usage(prompt_tokens=300000, completion_tokens=2000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 300000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 2000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        graduated_prompt_cost = (256000 * tier_1["input_cost_per_token"]) + (
            44000 * tier_2["input_cost_per_token"]
        )
        assert prompt_cost > graduated_prompt_cost

    def test_dashscope_tier_boundary_is_inclusive_of_range_end(self):
        """
        A request of exactly the tier's range end stays in that tier; one token more
        moves the whole request up to the next tier.
        """
        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        at_boundary = Usage(prompt_tokens=256000, completion_tokens=100)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=at_boundary
        )
        assert math.isclose(
            prompt_cost, 256000 * tier_1["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 100 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

        past_boundary = Usage(prompt_tokens=256001, completion_tokens=100)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=past_boundary
        )
        assert math.isclose(
            prompt_cost, 256001 * tier_2["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 100 * tier_2["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_output_tokens_do_not_select_their_own_tier(self):
        """
        The tier is chosen by input size only. A small-input request with a large
        completion is billed at the input's tier, not at a tier the completion count
        would have landed in on its own.
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=300000)
        _, completion_cost = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        assert math.isclose(
            completion_cost, 300000 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Cached and plain input tokens are billed at the rates of the single tier picked
        from the request's total input size (cached tokens included in that total).
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        # 50,000 total input tokens falls in the [32k, 128k] tier
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = (40000 * tier_2["input_cost_per_token"]) + (
            10000 * tier_2["cache_read_input_token_cost"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_cached_tokens_count_toward_tier_selection(self):
        """
        Cached tokens are part of the request's input size. A request whose plain input
        alone would sit in the first tier moves up when the cached tokens push the total
        past the boundary.
        """
        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        usage = Usage(
            prompt_tokens=40000,  # 30k cached + 10k new; plain input alone is under 32k
            completion_tokens=100,
            total_tokens=40100,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        expected_prompt_cost = (10000 * tier_2["input_cost_per_token"]) + (
            30000 * tier_2["cache_read_input_token_cost"]
        )
        first_tier_cost = (10000 * tier_1["input_cost_per_token"]) + (
            30000 * tier_1["cache_read_input_token_cost"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert prompt_cost > first_tier_cost

    def _register_string_valued_tiered_model(self, model_key: str) -> None:
        """Register a model whose tier costs are strings, mimicking YAML config parsing."""
        litellm.model_cost[model_key] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "tiered_pricing": [
                {
                    "range": [0, 1000],
                    "input_cost_per_token": "4e-07",
                    "output_cost_per_token": "1.6e-06",
                },
                {
                    "range": [1000, 2000],
                    "input_cost_per_token": "8e-07",
                    "output_cost_per_token": "3.2e-06",
                },
            ],
        }

    def test_dashscope_tiered_pricing_string_costs_within_tier(self):
        """
        Regression: YAML-parsed tier costs can be strings (e.g. "4e-07"). Costs that
        fall entirely within a single tier must still be computed as floats.
        """
        self._register_string_valued_tiered_model("dashscope/qwen-str-tier-test")

        usage = Usage(prompt_tokens=500, completion_tokens=200)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-str-tier-test", usage=usage
        )

        expected_prompt_cost = 500 * float("4e-07")
        expected_completion_cost = 200 * float("1.6e-06")

        assert prompt_cost > 0
        assert completion_cost > 0
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_string_costs_exceeding_highest_tier(self):
        """
        Regression: string-valued tier costs must also be coerced when the request is
        larger than the highest declared range and falls back to the last tier.
        """
        self._register_string_valued_tiered_model("dashscope/qwen-str-tier-test")

        usage = Usage(prompt_tokens=2500, completion_tokens=3000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-str-tier-test", usage=usage
        )

        expected_prompt_cost = 2500 * float("8e-07")
        expected_completion_cost = 3000 * float("3.2e-06")

        assert prompt_cost > 0
        assert completion_cost > 0
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Tests tiered pricing when the request's input size exceeds the highest defined
        tier range; the whole request falls back to the last (most expensive) tier.
        """
        usage = Usage(
            prompt_tokens=1200000, completion_tokens=1000
        )  # Max defined range for qwen-flash is 1M

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 1200000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 1000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)
