"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- Selects one pricing tier from the request's total input token count (step pricing).
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

    def test_dashscope_input_above_first_tier_bills_all_tokens_at_selected_tier(self):
        """
        Regression for #34729: a request whose total input exceeds the first tier is
        billed entirely at the selected tier's rates, with no graduated slicing, and
        completion tokens follow the tier chosen by the input size.
        """
        # Tiering for qwen-flash: Tier 1: [0, 256k], Tier 2: [256k, 1M]
        usage = Usage(prompt_tokens=300000, completion_tokens=2000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 300000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 2000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_small_input_with_large_output_stays_in_first_tier(self):
        """
        Regression for #34729: output tokens never select their own tier. A small
        input with an output count that would land in a higher tier is still billed
        at the tier picked by the input size.
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=300000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        assert math.isclose(
            prompt_cost, 1000 * tier_1["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 300000 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_input_at_tier_boundary_stays_in_lower_tier(self):
        """
        A request of exactly the first tier's upper bound stays in that tier, matching
        the official ``0 < Token <= 256K`` phrasing.
        """
        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]
        boundary = tier_1["range"][1]

        at_boundary, _ = dashscope_cost_per_token(
            model="qwen-flash",
            usage=Usage(prompt_tokens=boundary, completion_tokens=0),
        )
        just_above, _ = dashscope_cost_per_token(
            model="qwen-flash",
            usage=Usage(prompt_tokens=boundary + 1, completion_tokens=0),
        )

        assert math.isclose(
            at_boundary, boundary * tier_1["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            just_above, (boundary + 1) * tier_2["input_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Cached and plain input tokens are both billed at the tier selected by the
        request's total input size, using the tier's cache-specific rate for the
        cached portion. Uses qwen3-coder-plus, which has cache-specific pricing.
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        # 50k total input selects the tier containing 50k (first tier ends at 32k)
        tier = next(
            t
            for t in model_info["tiered_pricing"]
            if t["range"][0] < 50000 <= t["range"][1]
        )

        expected_total_prompt_cost = (10000 * tier["cache_read_input_token_cost"]) + (
            40000 * tier["input_cost_per_token"]
        )

        assert math.isclose(prompt_cost, expected_total_prompt_cost, rel_tol=1e-10)

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

        expected_prompt_cost = 500 * 4e-07
        expected_completion_cost = 200 * 1.6e-06

        assert prompt_cost > 0
        assert completion_cost > 0
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_string_costs_exceeding_highest_tier(self):
        """
        Regression: string-valued tier costs must also be coerced in the
        remaining-tokens path that charges tokens above the highest tier.
        """
        self._register_string_valued_tiered_model("dashscope/qwen-str-tier-test")

        usage = Usage(prompt_tokens=2500, completion_tokens=3000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-str-tier-test", usage=usage
        )

        # 2500 input tokens exceed the highest range, so the last tier's rates apply
        expected_prompt_cost = 2500 * 8e-07
        expected_completion_cost = 3000 * 3.2e-06

        assert prompt_cost > 0
        assert completion_cost > 0
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Input beyond the highest declared range falls back to the last (most
        expensive) tier for every token in the request.
        """
        usage = Usage(
            prompt_tokens=1200000, completion_tokens=1000
        )  # Max defined range for qwen-flash is 1M

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        last_tier = model_info["tiered_pricing"][-1]

        assert math.isclose(
            prompt_cost, 1200000 * last_tier["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 1000 * last_tier["output_cost_per_token"], rel_tol=1e-10
        )
