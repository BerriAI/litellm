"""
Regression tests for litellm/llms/fireworks_ai/cost_calculator.py

Covers:
- Cache-read tokens billed at cache_read_input_token_cost (issue #31714)
- Fallback to parameter-size-based pricing for unknown models
- Basic non-cached cost calculation still works
"""

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.fireworks_ai.cost_calculator import (
    cost_per_token,
    get_base_model_for_pricing,
)
from litellm.types.utils import PromptTokensDetails, Usage


class TestCacheReadTokenCost:
    """Regression: cached prompt tokens must be billed at cache_read_input_token_cost."""

    def test_cached_tokens_billed_at_reduced_rate(self):
        """
        Given a model with cache_read_input_token_cost defined and a usage object
        containing cached_tokens in prompt_tokens_details, the prompt cost must
        reflect the reduced rate for cached tokens rather than charging all
        prompt tokens at the full input rate.

        Reproduces issue #31714.
        """
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=800),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="accounts/fireworks/models/deepseek-v4-flash",
            usage=usage,
        )

        from litellm.utils import get_model_info

        model_info = get_model_info(
            model="accounts/fireworks/models/deepseek-v4-flash",
            custom_llm_provider="fireworks_ai",
        )
        input_rate = model_info["input_cost_per_token"]
        cache_rate = model_info["cache_read_input_token_cost"]

        non_cached_tokens = 1000 - 800
        expected_prompt_cost = non_cached_tokens * input_rate + 800 * cache_rate
        expected_completion_cost = 100 * model_info["output_cost_per_token"]

        assert prompt_cost == pytest.approx(expected_prompt_cost)
        assert completion_cost == pytest.approx(expected_completion_cost)

        naive_full_rate_cost = 1000 * input_rate
        assert prompt_cost < naive_full_rate_cost

    def test_no_cached_tokens_charges_full_rate(self):
        """Without cached tokens, all prompt tokens are billed at the full input rate."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
        )

        prompt_cost, completion_cost = cost_per_token(
            model="accounts/fireworks/models/deepseek-v4-flash",
            usage=usage,
        )

        from litellm.utils import get_model_info

        model_info = get_model_info(
            model="accounts/fireworks/models/deepseek-v4-flash",
            custom_llm_provider="fireworks_ai",
        )

        assert prompt_cost == pytest.approx(1000 * model_info["input_cost_per_token"])
        assert completion_cost == pytest.approx(100 * model_info["output_cost_per_token"])


class TestBaseModelFallback:
    """Verify that unknown models fall back to parameter-size-based pricing."""

    @pytest.mark.parametrize(
        "model_name,expected_category",
        [
            ("custom-3b-chat", "fireworks-ai-up-to-4b"),
            ("custom-8b-instruct", "fireworks-ai-4.1b-to-16b"),
            ("custom-70b-instruct", "fireworks-ai-above-16b"),
            ("custom-8x7b-moe", "fireworks-ai-moe-up-to-56b"),
            ("custom-8x22b-moe", "fireworks-ai-56b-to-176b"),
        ],
    )
    def test_base_model_resolution(self, model_name: str, expected_category: str):
        assert get_base_model_for_pricing(model_name) == expected_category

    def test_unknown_model_still_calculates_cost(self):
        """An unregistered model with a recognizable param count yields a valid cost."""
        usage = Usage(
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
        )

        prompt_cost, completion_cost = cost_per_token(
            model="my-custom-70b-chat",
            usage=usage,
        )

        assert prompt_cost > 0
        assert completion_cost > 0
