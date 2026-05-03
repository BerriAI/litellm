import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.cost_calculator import (
    cost_per_token,
    get_base_model_for_pricing,
)
from litellm.types.utils import Usage


class TestGetBaseModelForPricing:
    """Tests for the parameter-size heuristic."""

    def test_should_return_up_to_4b_for_small_model(self):
        assert (
            get_base_model_for_pricing("llama-3b-instruct") == "fireworks-ai-up-to-4b"
        )

    def test_should_return_4_1b_to_16b_for_medium_model(self):
        assert (
            get_base_model_for_pricing("llama-8b-instruct")
            == "fireworks-ai-4.1b-to-16b"
        )

    def test_should_return_above_16b_for_large_model(self):
        assert (
            get_base_model_for_pricing("llama-70b-instruct") == "fireworks-ai-above-16b"
        )

    def test_should_return_moe_up_to_56b(self):
        assert (
            get_base_model_for_pricing("mixtral-8x7b-instruct")
            == "fireworks-ai-moe-up-to-56b"
        )

    def test_should_return_moe_56b_to_176b(self):
        assert (
            get_base_model_for_pricing("mixtral-8x22b-instruct")
            == "fireworks-ai-56b-to-176b"
        )

    def test_should_return_default_for_unknown(self):
        assert get_base_model_for_pricing("some-random-model") == "fireworks-ai-default"


class TestCostPerToken:
    """Tests for cost_per_token with generic_cost_per_token and fallback."""

    def test_should_calculate_cost_for_mapped_model(self):
        """Mapped models (in model_prices_and_context_window.json) should succeed."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        prompt_cost, completion_cost = cost_per_token(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            usage=usage,
        )
        assert prompt_cost >= 0
        assert completion_cost >= 0

    def test_should_fallback_for_unmapped_model(self):
        """Unmapped models should fall back to the size-heuristic path."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        prompt_cost, completion_cost = cost_per_token(
            model="accounts/fireworks/models/custom-3b-finetune",
            usage=usage,
        )
        assert prompt_cost >= 0
        assert completion_cost >= 0

    def test_should_handle_zero_tokens(self):
        """Zero-token usage should yield zero cost."""
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        prompt_cost, completion_cost = cost_per_token(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            usage=usage,
        )
        assert prompt_cost == 0.0
        assert completion_cost == 0.0
