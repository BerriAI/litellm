"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- All-or-nothing tiered pricing, selected by the request's total input tokens.
- Falls back to flat-rate pricing for non-tiered models.
- Handles cache read and cache creation tokens.
- Correctly prices requests exceeding the highest defined tier.
"""

import math
import os

import pytest

# Add the project root to Python path

import litellm
from litellm.llms.dashscope.cost_calculator import (
    cost_per_token as dashscope_cost_per_token,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)


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
        Tests the dashscope tiered pricing when the request's input falls in the first tier.
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

    def test_dashscope_tiered_pricing_bills_whole_request_at_selected_tier(self):
        """
        Regression: Model Studio tiered pricing is all-or-nothing, not graduated. An input
        above the first tier's range must bill every token at the higher tier's rate.
        """
        # Tiering for qwen-flash: Tier 1: [0, 256k], Tier 2: [256k, 1M]
        usage = Usage(prompt_tokens=300000, completion_tokens=300000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]
        tier_2 = model_info["tiered_pricing"][1]

        expected_prompt_cost = 300000 * tier_2["input_cost_per_token"]
        expected_completion_cost = 300000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

        graduated_prompt_cost = (256000 * tier_1["input_cost_per_token"]) + (
            44000 * tier_2["input_cost_per_token"]
        )
        assert prompt_cost > graduated_prompt_cost

    def test_dashscope_tiered_pricing_boundary_stays_in_lower_tier(self):
        """
        A request of exactly range_end tokens stays in the lower tier, matching the
        official `0 < Token <= 256K` phrasing.
        """
        usage = Usage(prompt_tokens=256000, completion_tokens=1000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        tier_1 = litellm.get_model_info("dashscope/qwen-flash")["tiered_pricing"][0]

        assert math.isclose(
            prompt_cost, 256000 * tier_1["input_cost_per_token"], rel_tol=1e-10
        )
        assert math.isclose(
            completion_cost, 1000 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_tiered_pricing_output_uses_input_selected_tier(self):
        """
        The tier is chosen by input volume only: a small input with a huge output stays
        on the first tier's output rate.
        """
        usage = Usage(prompt_tokens=1000, completion_tokens=400000)
        _, completion_cost = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        tier_1 = litellm.get_model_info("dashscope/qwen-flash")["tiered_pricing"][0]

        assert math.isclose(
            completion_cost, 400000 * tier_1["output_cost_per_token"], rel_tol=1e-10
        )

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Tests tiered pricing with cached tokens: the tier is selected from the total
        input (text + cached), and cache reads bill at that tier's cache rate.
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, _ = dashscope_cost_per_token(model="qwen3-coder-plus", usage=usage)

        # 50k total input falls in qwen3-coder-plus tier 2 ([32k, 128k])
        tier_2 = litellm.get_model_info("dashscope/qwen3-coder-plus")["tiered_pricing"][1]

        expected_prompt_cost = (40000 * tier_2["input_cost_per_token"]) + (
            10000 * tier_2["cache_read_input_token_cost"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Requests above the highest declared range bill entirely at the last tier's rate.
        """
        usage = Usage(
            prompt_tokens=1200000, completion_tokens=1000
        )  # Max defined range for qwen-flash is 1M

        prompt_cost, _ = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        tier_2 = litellm.get_model_info("dashscope/qwen-flash")["tiered_pricing"][1]

        assert math.isclose(
            prompt_cost, 1200000 * tier_2["input_cost_per_token"], rel_tol=1e-10
        )

    def _register_tiered_model(self, model_key: str, tiered_pricing: list[dict]) -> None:
        litellm.model_cost[model_key] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "tiered_pricing": tiered_pricing,
        }

    def _register_string_valued_tiered_model(self, model_key: str) -> None:
        """Register a model whose tier costs are strings, mimicking YAML config parsing."""
        self._register_tiered_model(
            model_key,
            [
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
        )

    def test_dashscope_tiered_pricing_string_costs_within_tier(self):
        """
        Regression: YAML-parsed tier costs can be strings (e.g. "4e-07") and must still
        be computed as floats.
        """
        self._register_string_valued_tiered_model("dashscope/qwen-str-tier-test")

        usage = Usage(prompt_tokens=500, completion_tokens=200)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-str-tier-test", usage=usage
        )

        assert math.isclose(prompt_cost, 500 * float("4e-07"), rel_tol=1e-10)
        assert math.isclose(completion_cost, 200 * float("1.6e-06"), rel_tol=1e-10)

    def test_dashscope_tiered_pricing_string_costs_exceeding_highest_tier(self):
        """
        Regression: string-valued tier costs must also be coerced on the last-tier
        fallback path used by requests above the highest range.
        """
        self._register_string_valued_tiered_model("dashscope/qwen-str-tier-test")

        usage = Usage(prompt_tokens=2500, completion_tokens=3000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-str-tier-test", usage=usage
        )

        assert math.isclose(prompt_cost, 2500 * float("8e-07"), rel_tol=1e-10)
        assert math.isclose(completion_cost, 3000 * float("3.2e-06"), rel_tol=1e-10)

    def test_dashscope_tiered_cache_creation_tokens_use_tier_rate(self):
        """
        Regression (tiered cache creation): cache-creation tokens must bill at the
        selected tier's cache_creation_input_token_cost, not the input rate.
        """
        self._register_tiered_model(
            "dashscope/qwen-cache-write-test",
            [
                {
                    "range": [0, 256000],
                    "input_cost_per_token": 3.25e-07,
                    "output_cost_per_token": 1.95e-06,
                    "cache_creation_input_token_cost": 4.063e-07,
                    "cache_read_input_token_cost": 3.25e-08,
                },
                {
                    "range": [256000, 1000000],
                    "input_cost_per_token": 6.5e-07,
                    "output_cost_per_token": 3.9e-06,
                    "cache_creation_input_token_cost": 8.125e-07,
                    "cache_read_input_token_cost": 6.5e-08,
                },
            ],
        )

        usage = Usage(
            prompt_tokens=300000,  # 200k new + 60k cache creation + 40k cache read
            completion_tokens=1000,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=40000, cache_creation_tokens=60000
            ),
        )

        prompt_cost, _ = dashscope_cost_per_token(
            model="qwen-cache-write-test", usage=usage
        )

        expected_prompt_cost = (
            (200000 * 6.5e-07) + (60000 * 8.125e-07) + (40000 * 6.5e-08)
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_nested_cache_creation_input_tokens_bill_at_cache_write_rate(self):
        """
        Regression (LIT-5757): DashScope nests cache_creation_input_tokens inside
        prompt_tokens_details; those tokens must bill at the tier's cache-creation
        rate instead of being folded into text tokens at the input rate.
        """
        self._register_tiered_model(
            "dashscope/qwen-nested-cache-write-test",
            [
                {
                    "range": [0, 128000],
                    "input_cost_per_token": 4e-07,
                    "cache_read_input_token_cost": 1.6e-07,
                    "cache_creation_input_token_cost": 5e-07,
                    "output_cost_per_token": 1.6e-06,
                }
            ],
        )

        usage = Usage(
            prompt_tokens=2059,
            completion_tokens=201,
            total_tokens=2260,
            prompt_tokens_details={
                "cached_tokens": 0,
                "text_tokens": 2059,
                "cache_type": "ephemeral",
                "cache_creation_input_tokens": 2048,
                "cache_creation": {"ephemeral_5m_input_tokens": 2048},
            },
            completion_tokens_details={"reasoning_tokens": 170},
        )

        prompt_cost, _ = dashscope_cost_per_token(
            model="qwen-nested-cache-write-test", usage=usage
        )

        assert math.isclose(
            prompt_cost, (2048 * 5e-07) + (11 * 4e-07), rel_tol=1e-10
        )

    def test_dashscope_tiered_cache_creation_falls_back_to_tier_input_rate(self):
        """
        Tiers without a cache_creation_input_token_cost bill cache-creation tokens at
        that tier's input rate.
        """
        self._register_tiered_model(
            "dashscope/qwen-no-cache-write-test",
            [
                {
                    "range": [0, 256000],
                    "input_cost_per_token": 3.25e-07,
                    "output_cost_per_token": 1.95e-06,
                }
            ],
        )

        usage = Usage(
            prompt_tokens=10000,
            completion_tokens=100,
            prompt_tokens_details=PromptTokensDetailsWrapper(cache_creation_tokens=4000),
        )

        prompt_cost, _ = dashscope_cost_per_token(
            model="qwen-no-cache-write-test", usage=usage
        )

        assert math.isclose(prompt_cost, 10000 * 3.25e-07, rel_tol=1e-10)

    def test_dashscope_flat_cache_creation_tokens_use_flat_rate(self):
        """Flat-priced models bill cache-creation tokens at their cache-creation rate."""
        litellm.model_cost["dashscope/qwen-flat-cache-write-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "input_cost_per_token": 3.25e-07,
            "output_cost_per_token": 1.95e-06,
            "cache_creation_input_token_cost": 4.063e-07,
            "cache_read_input_token_cost": 3.25e-08,
        }

        usage = Usage(
            prompt_tokens=10000,
            completion_tokens=100,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=2000, cache_creation_tokens=3000
            ),
        )

        prompt_cost, _ = dashscope_cost_per_token(
            model="qwen-flat-cache-write-test", usage=usage
        )

        expected_prompt_cost = (
            (5000 * 3.25e-07) + (3000 * 4.063e-07) + (2000 * 3.25e-08)
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tier_without_an_output_rate_bills_the_model_rate(self):
        """
        Regression: a tier declaring only an input rate served every completion for free,
        since a missing tier output rate had no tier-level fallback to stand in for it.
        """
        litellm.model_cost["dashscope/qwen-input-only-tier-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "output_cost_per_token": 1.6e-06,
            "tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 4e-07}],
        }

        usage = Usage(prompt_tokens=500, completion_tokens=200)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-input-only-tier-test", usage=usage
        )

        assert math.isclose(prompt_cost, 500 * 4e-07, rel_tol=1e-10)
        assert math.isclose(completion_cost, 200 * 1.6e-06, rel_tol=1e-10)

    def test_dashscope_tier_without_an_output_rate_bills_the_model_reasoning_rate(self):
        """
        Regression: a tier declaring only an input rate billed reasoning tokens at the model's
        plain output rate, ignoring the model's dedicated reasoning rate.
        """
        litellm.model_cost["dashscope/qwen-input-only-reasoning-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "output_cost_per_token": 1.6e-06,
            "output_cost_per_reasoning_token": 4e-06,
            "tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 4e-07}],
        }

        usage = Usage(
            prompt_tokens=500,
            completion_tokens=200,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=150),
        )
        _, completion_cost = dashscope_cost_per_token(
            model="qwen-input-only-reasoning-test", usage=usage
        )

        assert math.isclose(
            completion_cost, (50 * 1.6e-06) + (150 * 4e-06), rel_tol=1e-10
        )

    def test_dashscope_tier_output_rate_wins_over_the_model_reasoning_rate(self):
        """
        A tier declaring its own output rate keeps reasoning tokens on that tier rather than
        mixing in a model-level reasoning rate.
        """
        litellm.model_cost["dashscope/qwen-tier-output-reasoning-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "output_cost_per_reasoning_token": 4e-06,
            "tiered_pricing": [
                {
                    "range": [0, 1000],
                    "input_cost_per_token": 4e-07,
                    "output_cost_per_token": 1.6e-06,
                }
            ],
        }

        usage = Usage(
            prompt_tokens=500,
            completion_tokens=200,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=150),
        )
        _, completion_cost = dashscope_cost_per_token(
            model="qwen-tier-output-reasoning-test", usage=usage
        )

        assert math.isclose(completion_cost, 200 * 1.6e-06, rel_tol=1e-10)

    def test_dashscope_model_zero_reasoning_rate_bills_reasoning_free(self):
        """
        Regression: a model declaring an explicit zero reasoning rate had it treated as
        missing, billing reasoning tokens at the plain output rate instead of free.
        """
        litellm.model_cost["dashscope/qwen-zero-reasoning-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "input_cost_per_token": 4e-07,
            "output_cost_per_token": 1.6e-06,
            "output_cost_per_reasoning_token": 0,
        }

        usage = Usage(
            prompt_tokens=500,
            completion_tokens=200,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=150),
        )
        _, completion_cost = dashscope_cost_per_token(
            model="qwen-zero-reasoning-test", usage=usage
        )

        assert math.isclose(completion_cost, 50 * 1.6e-06, rel_tol=1e-10)

    def test_dashscope_tier_zero_reasoning_rate_bills_reasoning_free(self):
        """
        Regression: a tier declaring an explicit zero reasoning rate had it treated as
        missing, billing reasoning tokens at the tier's output rate instead of free.
        """
        litellm.model_cost["dashscope/qwen-tier-zero-reasoning-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "tiered_pricing": [
                {
                    "range": [0, 1000],
                    "input_cost_per_token": 4e-07,
                    "output_cost_per_token": 1.6e-06,
                    "output_cost_per_reasoning_token": 0,
                }
            ],
        }

        usage = Usage(
            prompt_tokens=500,
            completion_tokens=200,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=150),
        )
        _, completion_cost = dashscope_cost_per_token(
            model="qwen-tier-zero-reasoning-test", usage=usage
        )

        assert math.isclose(completion_cost, 50 * 1.6e-06, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_zero_input_falls_back_to_flat_rates(self):
        """
        No tier can be selected without input tokens, so an empty-prompt request must
        not be charged at the most expensive tier.
        """
        litellm.model_cost["dashscope/qwen-zero-input-test"] = {
            "litellm_provider": "dashscope",
            "mode": "chat",
            "input_cost_per_token": 4e-07,
            "output_cost_per_token": 1.6e-06,
            "tiered_pricing": [
                {
                    "range": [0, 1000],
                    "input_cost_per_token": 4e-07,
                    "output_cost_per_token": 1.6e-06,
                },
                {
                    "range": [1000, 2000],
                    "input_cost_per_token": 8e-07,
                    "output_cost_per_token": 3.2e-06,
                },
            ],
        }

        usage = Usage(prompt_tokens=0, completion_tokens=500)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-zero-input-test", usage=usage
        )

        assert prompt_cost == 0.0
        assert math.isclose(completion_cost, 500 * 1.6e-06, rel_tol=1e-10)
