"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- All-or-nothing tiered pricing (single tier rate applied to the whole request).
- Flat-rate pricing fallback for models without ``tiered_pricing``.
- Cached-token interactions: cache cost is billed at the input-tier's rate
  (selected by total input tokens, not by cached-token count alone).
- Output cost uses the same tier as the input.
- Boundary semantics at ``range_end``.
- Requests exceeding the highest declared tier fall back to the last tier.
"""

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
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


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
        Tokens entirely within tier 1 are billed at tier 1's rate.
        Uses 'dashscope/qwen-flash' (tier 1 = [0, 256k]).
        """
        usage = Usage(prompt_tokens=100000, completion_tokens=50000)
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        expected_prompt_cost = 100000 * tier_1["input_cost_per_token"]
        expected_completion_cost = 50000 * tier_1["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_crossing_into_higher_tier(self):
        """
        All-or-nothing: a 300k input request on qwen-flash (tier 1 = [0, 256k],
        tier 2 = [256k, 1M]) is billed entirely at tier 2's rate — *not* split
        as 256k @ tier 1 + 44k @ tier 2 (which would be income-tax / graduated).

        Reference: https://help.aliyun.com/zh/model-studio/billing-for-model-studio
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

    def test_dashscope_tiered_pricing_with_caching(self):
        """
        Cache cost is billed at the input-tier's rate.

        qwen3-coder-plus tiers: [0, 32k], (32k, 128k], (128k, 256k], (256k, 1M].
        50k total input falls in tier 2 (32k, 128k], so both the text and the
        cache portions are billed at tier 2 — even though the cached portion
        alone (10k) would individually sit in tier 1.
        """
        usage = Usage(
            prompt_tokens=50000,  # 10k cached + 40k new
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
        )

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen3-coder-plus")
        # 50k total input falls in tier 2 = index 1
        tier_2 = model_info["tiered_pricing"][1]

        expected_text_cost = 40000 * tier_2["input_cost_per_token"]
        expected_cache_cost = 10000 * tier_2["cache_read_input_token_cost"]
        expected_prompt_cost = expected_text_cost + expected_cache_cost

        expected_completion_cost = 1000 * tier_2["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_exceeding_highest_tier(self):
        """
        Requests larger than the highest declared range fall back to the last
        tier's rate. qwen-flash declares up to 1M; a 1.2M request is still
        billed entirely at tier 2's rate.
        """
        usage = Usage(prompt_tokens=1200000, completion_tokens=1000)

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        last_tier = model_info["tiered_pricing"][-1]

        expected_prompt_cost = 1200000 * last_tier["input_cost_per_token"]
        expected_completion_cost = 1000 * last_tier["output_cost_per_token"]

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_boundary_at_range_end(self):
        """
        A request of exactly ``range_end`` tokens belongs to the lower tier,
        matching the official phrasing ``0 < Token ≤ 32K``.

        On qwen-flash (tier 1 ends at 256000) an input of exactly 256000 must
        bill at tier 1, not tier 2.
        """
        usage = Usage(prompt_tokens=256000, completion_tokens=10)

        prompt_cost, _ = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        expected_prompt_cost = 256000 * tier_1["input_cost_per_token"]
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_output_uses_input_tier(self):
        """
        Output token rate is taken from the tier selected by *input* tokens,
        regardless of how many output tokens were produced. Sending 10k input
        (tier 1) + 50k output on qwen-flash must bill the 50k output at tier 1.
        """
        usage = Usage(prompt_tokens=10000, completion_tokens=50000)
        _, completion_cost = dashscope_cost_per_token(model="qwen-flash", usage=usage)

        model_info = litellm.get_model_info("dashscope/qwen-flash")
        tier_1 = model_info["tiered_pricing"][0]

        expected_completion_cost = 50000 * tier_1["output_cost_per_token"]
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_dashscope_tiered_pricing_with_zero_input(self):
        """
        Edge case: a request with zero input tokens has no tier to match —
        ``_select_tier_for_input`` returns ``None`` and the caller falls back
        to flat pricing (which is 0 for purely-tiered models like qwen-flash).
        This guards against the regression of charging zero-input completions
        at the highest tier's rate.
        """
        usage = Usage(prompt_tokens=0, completion_tokens=100)

        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-flash", usage=usage
        )

        # qwen-flash has no flat input_cost_per_token / output_cost_per_token,
        # so the fallback yields 0 for both — not the (incorrect) last-tier rate.
        assert prompt_cost == 0.0
        assert completion_cost == 0.0
