"""
Test suite for XAI cost calculation functionality.
"""

import math
import os
import sys

import litellm
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    Usage,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.xai.cost_calculator import cost_per_token


class TestXAICostCalculator:
    """Test suite for XAI cost calculation functionality."""

    def setup_method(self):
        """Set up test environment."""
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
