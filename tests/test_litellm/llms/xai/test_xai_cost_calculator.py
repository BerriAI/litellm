"""
Test suite for XAI cost calculation functionality.
"""

import math
import os
import sys

import litellm
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.xai.cost_calculator import cost_per_token, cost_per_web_search_request


class TestXAICostCalculator:
    """Test suite for XAI cost calculation functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Load the main model cost map directly to ensure we have the latest pricing
        import json
        try:
            with open("model_prices_and_context_window.json", "r") as f:
                model_cost_map = json.load(f)
            litellm.model_cost = model_cost_map
        except FileNotFoundError:
            # Fallback to default behavior
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_basic_cost_calculation(self):
        """Test basic cost calculation without reasoning tokens."""
        usage = Usage(prompt_tokens=12, completion_tokens=125, total_tokens=137)

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Output: 125 tokens * $5e-7 = $0.0000625
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = 125 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_reasoning_tokens_cost_calculation(self):
        """Test cost calculation with reasoning tokens from completion_tokens_details."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=125,
            total_tokens=1086,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=949,
                rejected_prediction_tokens=0,
                text_tokens=None,  # Not set, but doesn't matter for XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (125 + 949) tokens * $5e-7 = $0.000537
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (125 + 949) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_reasoning_and_text_tokens_cost_calculation(self):
        """Test cost calculation with both reasoning and text tokens."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=125,
            total_tokens=1086,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=949,
                rejected_prediction_tokens=0,
                text_tokens=76,  # Explicitly set (but ignored in XAI billing)
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs for grok-3-mini:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (125 + 949) tokens * $5e-7 = $0.000537
        # Note: text_tokens field is ignored, only completion_tokens + reasoning_tokens matters
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (125 + 949) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_4_cost_calculation(self):
        """Test cost calculation for grok-4 model."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=200,
            total_tokens=210,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=150,
                rejected_prediction_tokens=0,
                text_tokens=50,  # Ignored in XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-4", usage=usage)

        # Expected costs for grok-4:
        # Input: 10 tokens * $3e-6 = $0.00003
        # Completion: (200 + 150) tokens * $1.5e-5 = $0.00525
        expected_prompt_cost = 10 * 3e-6
        expected_completion_cost = (200 + 150) * 1.5e-5

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_grok_3_fast_beta_cost_calculation(self):
        """Test cost calculation for grok-3-fast-beta model."""
        usage = Usage(
            prompt_tokens=20,
            completion_tokens=300,
            total_tokens=320,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=200,
                rejected_prediction_tokens=0,
                text_tokens=100,  # Ignored in XAI billing
            ),
        )

        prompt_cost, completion_cost = cost_per_token(
            model="grok-3-fast-beta", usage=usage
        )

        # Expected costs for grok-3-fast-beta:
        # Input: 20 tokens * $5e-6 = $0.0001
        # Completion: (300 + 200) tokens * $2.5e-5 = $0.0125
        expected_prompt_cost = 20 * 5e-6
        expected_completion_cost = (300 + 200) * 2.5e-5

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_edge_case_no_completion_tokens_details(self):
        """Test cost calculation when completion_tokens_details is not present."""
        usage = Usage(prompt_tokens=12, completion_tokens=125, total_tokens=137)

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Should fall back to basic calculation
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = 125 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_edge_case_large_reasoning_tokens(self):
        """Test cost calculation when reasoning_tokens is larger than completion_tokens."""
        usage = Usage(
            prompt_tokens=12,
            completion_tokens=50,  # Less than reasoning_tokens
            total_tokens=62,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=100,  # More than completion_tokens
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # Expected costs:
        # Input: 12 tokens * $3e-7 = $0.0000036
        # Completion: (50 + 100) tokens * $5e-7 = $0.000075
        expected_prompt_cost = 12 * 3e-7
        expected_completion_cost = (50 + 100) * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_above_128k_tokens(self):
        """Test tiered pricing for tokens above 128k."""
        # Test with grok-4-fast-reasoning which has tiered pricing
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=100000,  # Above 128k threshold
            total_tokens=250000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=50000,  # Total completion tokens = 100000 + 50000 = 150000 > 128k
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="xai/grok-4-fast-reasoning", usage=usage)

        # Expected costs for grok-4-fast-reasoning with tiered pricing:
        # Input: 150000 tokens * $0.4e-6 (ALL tokens at tiered rate since input > 128k) = $0.06
        # Completion: (100000 + 50000) tokens * $1e-6 (tiered rate since input > 128k) = $0.15
        expected_prompt_cost = 150000 * 0.4e-6
        expected_completion_cost = (100000 + 50000) * 1e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_below_128k_tokens(self):
        """Test that regular pricing is used for tokens below 128k threshold."""
        # Test with grok-4-fast-reasoning which has tiered pricing
        usage = Usage(
            prompt_tokens=100000,  # Below 128k threshold
            completion_tokens=50000,
            total_tokens=150000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=10000,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="xai/grok-4-fast-reasoning", usage=usage)

        # Expected costs for grok-4-fast-reasoning with regular pricing:
        # Input: 100000 tokens * $0.2e-6 (regular rate) = $0.02
        # Completion: (50000 + 10000) tokens * $0.5e-6 (regular rate) = $0.03
        expected_prompt_cost = 100000 * 0.2e-6
        expected_completion_cost = (50000 + 10000) * 0.5e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_grok_4_latest(self):
        """Test tiered pricing for grok-4-latest model."""
        usage = Usage(
            prompt_tokens=200000,  # Above 128k threshold
            completion_tokens=100000,
            total_tokens=300000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=50000,
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="xai/grok-4-latest", usage=usage)

        # Expected costs for grok-4-latest with tiered pricing:
        # Input: 200000 tokens * $6e-6 (ALL tokens at tiered rate since input > 128k) = $1.2
        # Completion: (100000 + 50000) tokens * $30e-6 (tiered rate since input > 128k) = $4.5
        expected_prompt_cost = 200000 * 6e-6
        expected_completion_cost = (100000 + 50000) * 30e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_output_tokens_below_128k(self):
        """Test that output tokens get tiered rate when input tokens > 128k, even if output tokens < 128k."""
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=50000,  # Below 128k threshold
            total_tokens=200000,
            completion_tokens_details=CompletionTokensDetailsWrapper(
                accepted_prediction_tokens=0,
                audio_tokens=0,
                reasoning_tokens=10000,  # Total completion tokens = 50000 + 10000 = 60000 < 128k
                rejected_prediction_tokens=0,
                text_tokens=None,
            ),
        )

        prompt_cost, completion_cost = cost_per_token(model="xai/grok-4-fast-reasoning", usage=usage)

        # Expected costs for grok-4-fast-reasoning:
        # Input: 150000 tokens * $0.4e-6 (ALL tokens at tiered rate since input > 128k) = $0.06
        # Completion: (50000 + 10000) tokens * $1e-6 (tiered rate since input > 128k) = $0.06
        expected_prompt_cost = 150000 * 0.4e-6
        expected_completion_cost = (50000 + 10000) * 1e-6

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_model_without_tiered_pricing(self):
        """Test that models without tiered pricing use regular pricing even above 128k."""
        usage = Usage(
            prompt_tokens=150000,  # Above 128k threshold
            completion_tokens=50000,
            total_tokens=200000,
        )

        prompt_cost, completion_cost = cost_per_token(model="grok-3-mini", usage=usage)

        # grok-3-mini doesn't have tiered pricing, so should use regular rates:
        # Input: 150000 tokens * $3e-7 (regular rate) = $0.045
        # Completion: 50000 tokens * $5e-7 (regular rate) = $0.025
        expected_prompt_cost = 150000 * 3e-7
        expected_completion_cost = 50000 * 5e-7

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_web_search_cost_calculation(self):
        """Test web search cost calculation for X.AI models."""
        # Test with web_search_requests in prompt_tokens_details (primary path)
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
                web_search_requests=3,  # 3 sources used
            )
        )
        
        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})
        
        # Expected cost: 3 sources * $0.025 per source = $0.075
        expected_cost = 3 * (25.0 / 1000.0)  # 3 * $0.025
        
        assert math.isclose(web_search_cost, expected_cost, rel_tol=1e-10)
        assert math.isclose(web_search_cost, 0.075, rel_tol=1e-10)

    def test_web_search_cost_fallback_calculation(self):
        """Test web search cost calculation using fallback num_sources_used."""
        # Test fallback: num_sources_used on usage object
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        # Manually set num_sources_used (as done by transformation layer)
        setattr(usage, "num_sources_used", 5)
        
        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})
        
        # Expected cost: 5 sources * $0.025 per source = $0.125
        expected_cost = 5 * (25.0 / 1000.0)  # 5 * $0.025
        
        assert math.isclose(web_search_cost, expected_cost, rel_tol=1e-10)
        assert math.isclose(web_search_cost, 0.125, rel_tol=1e-10)

    def test_web_search_no_sources_used(self):
        """Test web search cost calculation when no sources are used."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=100,
                web_search_requests=0,  # No web search
            )
        )
        
        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})
        
        # Expected cost: 0 sources * $0.025 per source = $0.0
        assert web_search_cost == 0.0

    def test_web_search_cost_without_prompt_tokens_details(self):
        """Test web search cost calculation when prompt_tokens_details is None."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        web_search_cost = cost_per_web_search_request(usage=usage, model_info={})
        
        # Expected cost: No web search data = $0.0
        assert web_search_cost == 0.0
