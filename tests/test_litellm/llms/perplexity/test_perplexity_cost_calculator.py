"""
Test file for Perplexity cost calculator functionality.

Tests the cost calculation for Perplexity models including citation tokens,
search queries, and reasoning tokens.
"""

import json
import math
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# Add the project root to Python path
import litellm
from litellm.llms.perplexity.cost_calculator import (
    cost_per_token as perplexity_cost_per_token,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    OffPeakPricing,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestPerplexityCostCalculator:
    """Test suite for Perplexity cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing."""
        # Ensure we use local model cost map for consistent testing
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

        # Load the model cost map
        try:
            with open("model_prices_and_context_window.json", "r") as f:
                model_cost_map = json.load(f)
            litellm.model_cost = model_cost_map
        except FileNotFoundError:
            # Fallback to ensure we have the Perplexity model configuration
            litellm.model_cost = {
                "perplexity/sonar-deep-research": {
                    "max_tokens": 128000,
                    "max_input_tokens": 128000,
                    "input_cost_per_token": 2e-06,
                    "output_cost_per_token": 8e-06,
                    "output_cost_per_reasoning_token": 3e-06,
                    "citation_cost_per_token": 2e-06,
                    "search_context_cost_per_query": {
                        "search_context_size_low": 0.005,
                        "search_context_size_medium": 0.005,
                        "search_context_size_high": 0.005,
                    },
                    "litellm_provider": "perplexity",
                    "mode": "chat",
                    "supports_reasoning": True,
                    "supports_web_search": True,
                }
            }

    def test_missing_model_info_fields(self):
        """Test behavior when model info is missing some fields."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=2),
        )

        usage.citation_tokens = 25

        # Mock get_model_info to return incomplete model info
        with patch("litellm.llms.perplexity.cost_calculator.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 8e-6,
                # Missing search_queries_cost_per_query
            }

            prompt_cost, completion_cost = perplexity_cost_per_token(model="sonar-deep-research", usage=usage)

            # Should only calculate basic costs when fields are missing
            expected_prompt_cost = 100 * 2e-6
            expected_completion_cost = 50 * 8e-6

            assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
            assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_uses_perplexity_provided_cost_when_available(self):
        """
        Test that when Perplexity provides pre-calculated cost in usage.cost.total_cost,
        it is used directly instead of manual calculation.

        This is the fix for issue #15337 - Perplexity returns accurate costs including
        request_cost (fixed per-request fee) that LiteLLM cannot calculate.
        """
        # Create usage with Perplexity's cost object (as returned by the API)
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        # Add the cost object that Perplexity returns
        usage.cost = {
            "input_tokens_cost": 0.0,
            "output_tokens_cost": 0.002,
            "request_cost": 0.006,
            "total_cost": 0.008,
        }

        prompt_cost, completion_cost = perplexity_cost_per_token(model="sonar-pro", usage=usage)

        # When Perplexity provides total_cost, we use it directly
        # prompt_cost should be 0, completion_cost should be total_cost
        assert prompt_cost == 0.0
        assert completion_cost == 0.008
        assert prompt_cost + completion_cost == 0.008

    def test_uses_perplexity_provided_cost_when_normalized_to_float(self):
        """
        Regression: for Responses API / Agent API models, `ResponseAPIUsage.parse_cost`
        (litellm/types/llms/openai.py) already flattens Perplexity's
        `usage.cost.total_cost` dict down to a plain float before
        `_transform_response_api_usage_to_chat_usage` (litellm/responses/utils.py) copies
        it onto the chat `Usage` object. So `usage.cost` arrives here as a float, not a
        dict, on that path.

        Pre-fix, the `isinstance(cost_info, dict)` check was always False for a float,
        so the pre-calculated cost branch was dead code for every Responses-mode
        Perplexity model and it silently fell back to manual token-rate calculation,
        recording $0 for any model missing static per-token rates (e.g.
        perplexity/openai/gpt-5.2 before rates existed).
        """
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage.cost = 0.008

        prompt_cost, completion_cost = perplexity_cost_per_token(model="sonar-pro", usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.008

    OFF_PEAK_MODEL = "sonar-off-peak-test"
    OFF_PEAK_WINDOW = "14:00-00:00"
    INSIDE_WINDOW = datetime(2026, 9, 3, 17, 25, tzinfo=timezone.utc)
    OUTSIDE_WINDOW = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)

    def _register_off_peak_model(self, off_peak_pricing: OffPeakPricing) -> None:
        litellm.model_cost[f"perplexity/{self.OFF_PEAK_MODEL}"] = {
            "litellm_provider": "perplexity",
            "mode": "chat",
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 1e-06,
            "output_cost_per_reasoning_token": 3e-06,
            "citation_cost_per_token": 2e-06,
            "search_context_cost_per_query": {"search_context_size_low": 0.005},
            "off_peak_pricing": off_peak_pricing,
        }

    def test_off_peak_window_swaps_in_the_off_peak_rates(self):
        """
        Regression (LIT-6874): a deployment configured with off_peak_pricing kept billing the
        standard perplexity rates inside its window, while the same block on a deepseek
        deployment billed the off-peak rates.
        """
        self._register_off_peak_model(
            {"hours_utc": self.OFF_PEAK_WINDOW, "input_cost_per_token": 1e-07, "output_cost_per_token": 2e-07}
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)

        prompt_cost, completion_cost = perplexity_cost_per_token(
            model=self.OFF_PEAK_MODEL, usage=usage, current_time=self.INSIDE_WINDOW
        )

        assert math.isclose(prompt_cost, 1000 * 1e-07, rel_tol=1e-10)
        assert math.isclose(completion_cost, 200 * 2e-07, rel_tol=1e-10)

        peak_prompt_cost, peak_completion_cost = perplexity_cost_per_token(
            model=self.OFF_PEAK_MODEL, usage=usage, current_time=self.OUTSIDE_WINDOW
        )

        assert math.isclose(peak_prompt_cost, 1000 * 1e-06, rel_tol=1e-10)
        assert math.isclose(peak_completion_cost, 200 * 1e-06, rel_tol=1e-10)

    def test_off_peak_rates_leave_citation_search_and_reasoning_fees_alone(self):
        """Inside the window only the plain input and output rates change: citation tokens, the
        per-request search fee, and a dedicated reasoning rate keep billing as published."""
        self._register_off_peak_model(
            {"hours_utc": self.OFF_PEAK_WINDOW, "input_cost_per_token": 1e-07, "output_cost_per_token": 2e-07}
        )
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1),
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=50),
        )
        usage.citation_tokens = 100

        prompt_cost, completion_cost = perplexity_cost_per_token(
            model=self.OFF_PEAK_MODEL, usage=usage, current_time=self.INSIDE_WINDOW
        )

        assert math.isclose(prompt_cost, (1000 * 1e-07) + (100 * 2e-06), rel_tol=1e-10)
        assert math.isclose(completion_cost, (150 * 2e-07) + (50 * 3e-06) + 0.005, rel_tol=1e-10)

    def test_provider_stated_cost_still_wins_inside_an_off_peak_window(self):
        """A response that carries Perplexity's own metered cost bills that cost whatever the
        window says; the caller strips it when the deployment carries custom pricing."""
        self._register_off_peak_model(
            {"hours_utc": "00:00-00:00", "input_cost_per_token": 1e-07, "output_cost_per_token": 2e-07}
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)
        usage.cost = {"total_cost": 0.00501}

        prompt_cost, completion_cost = perplexity_cost_per_token(model=self.OFF_PEAK_MODEL, usage=usage)

        assert prompt_cost == 0.0
        assert completion_cost == 0.00501
