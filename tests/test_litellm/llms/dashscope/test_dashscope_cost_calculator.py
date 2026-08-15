"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- Selects one pricing tier by total input tokens and bills the whole request at it.
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

    def test_dashscope_tiered_pricing_selects_tier_by_total_input(self):
        """
        Dashscope tiered pricing is all-or-nothing: the tier is selected by the total
        input token count of the request, and every input and output token is billed at
        that one tier's rate (not graduated, income-tax-style slicing). A 300k input
        request exceeds the 256k first-tier range, so the whole request bills at tier 2.
        """
        usage = Usage(prompt_tokens=300000, completion_tokens=300000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 300000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 300000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        graduated_prompt_cost = (256000 * model_info["tiered_pricing"][0]["input_cost_per_token"]) + (
            44000 * tier_2["input_cost_per_token"]
        )
        assert not math.isclose(prompt_cost, graduated_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_matches_request_size_tier_selection(self):
        """
        Regression for #34729: the calculator must agree with the request-size tier model
        the proxy budget code uses (select_tier_for_input), not graduated slicing. For a
        300k input / 2k output qwen-flash request the whole thing bills at tier 2, and the
        result must not equal the old graduated total that under-charged.
        """
        from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import (
            calculate_tiered_cost,
            select_tier_for_input,
            tier_rate,
        )

        input_tokens = 300000
        output_tokens = 2000
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        tiered_pricing = litellm.get_model_info("dashscope/qwen-flash")["tiered_pricing"]
        selected_tier = select_tier_for_input(tiered_pricing, input_tokens)
        expected_prompt_cost = input_tokens * tier_rate(selected_tier, "input_cost_per_token")
        expected_completion_cost = output_tokens * tier_rate(selected_tier, "output_cost_per_token")

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        graduated_prompt_cost = calculate_tiered_cost(
            input_tokens, tiered_pricing, "input_cost_per_token"
        )
        assert not math.isclose(prompt_cost, graduated_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Tiered pricing with cached tokens. The tier is chosen by the total input token
        count (cached + new), then cached tokens bill at that tier's cache rate and new
        tokens at that tier's input rate. qwen3-coder-plus tiers start at [0, 32k], so a
        50k input request lands in the [32k, 128k] tier and every input token bills there.
        """
        usage = Usage(
            prompt_tokens=50000,
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        selected_tier = model_info["tiered_pricing"][1]

        expected_cache_cost = 10000 * selected_tier["cache_read_input_token_cost"]
        expected_text_cost = 40000 * selected_tier["input_cost_per_token"]
        expected_total_prompt_cost = expected_cache_cost + expected_text_cost

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

        expected_prompt_cost = 500 * float("4e-07")
        expected_completion_cost = 200 * float("1.6e-06")

        assert prompt_cost > 0
        assert completion_cost > 0
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_string_costs_exceeding_highest_tier(self):
        """
        Regression: string-valued tier costs must be coerced to float. An input above the
        highest declared range falls back to the last (most expensive) tier, and the whole
        request bills at that tier's rate.
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
        Tests tiered pricing when the input token count exceeds the highest defined tier
        range. The request falls back to the last (most expensive) tier and every input
        token bills at that tier's rate.
        """
        usage = Usage(prompt_tokens=1200000, completion_tokens=1000)

        prompt_cost, _ = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 1200000 * tier_2["input_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
