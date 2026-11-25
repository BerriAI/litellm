"""
Test Azure OpenAI cost calculator — service_tier pricing.
"""

import pytest

import litellm
from litellm.llms.azure.cost_calculation import cost_per_token
from litellm.types.utils import Usage


# Register a test model with tier-specific pricing
TEST_MODEL = "test-azure-gpt-4.1"
TEST_MODEL_COST = {
    TEST_MODEL: {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_priority": 0.01,
        "output_cost_per_token_priority": 0.02,
        "input_cost_per_token_flex": 0.0005,
        "output_cost_per_token_flex": 0.001,
        "litellm_provider": "azure",
        "max_tokens": 8192,
    }
}


class TestAzureServiceTierCostCalculation:
    """Test that service_tier is passed through Azure cost calculation."""

    @pytest.fixture(autouse=True)
    def register_test_model(self):
        litellm.register_model(model_cost=TEST_MODEL_COST)

    def test_service_tier_priority_higher_cost(self):
        """Priority tier should cost more than standard."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage
        )
        priority_prompt, priority_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="priority"
        )

        assert priority_prompt > standard_prompt
        assert priority_completion > standard_completion

    def test_service_tier_flex_lower_cost(self):
        """Flex tier should cost less than standard."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage
        )
        flex_prompt, flex_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="flex"
        )

        assert flex_prompt < standard_prompt
        assert flex_completion < standard_completion

    def test_service_tier_none_returns_standard(self):
        """service_tier=None should return standard pricing."""
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        none_prompt, none_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier=None
        )
        standard_prompt, standard_completion = cost_per_token(
            model=TEST_MODEL, usage=usage, service_tier="standard"
        )

        assert abs(none_prompt - standard_prompt) < 1e-10
        assert abs(none_completion - standard_completion) < 1e-10
