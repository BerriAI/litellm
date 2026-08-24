"""
Test Azure OpenAI cost calculator — service_tier pricing.
"""

import pytest

import litellm
from litellm.llms.azure.cost_calculation import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from litellm.utils import get_model_info


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


@pytest.mark.parametrize(
    "model",
    ["azure/us/gpt-4o-2024-11-20", "azure/eu/gpt-4o-2024-11-20"],
)
def test_gpt4o_2024_11_20_data_zone_cache_read_cost(model, local_model_cost_map):
    model_info = get_model_info(model=model, custom_llm_provider="azure")
    assert model_info["cache_read_input_token_cost"] == 1.375e-06
    assert model_info["input_cost_per_token"] == 2.75e-06
    assert model_info["output_cost_per_token"] == 1.1e-05

    usage = Usage(
        prompt_tokens=6000,
        completion_tokens=10,
        total_tokens=6010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=5000),
    )
    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx((1000 * 2.75e-06) + (5000 * 1.375e-06))
    assert completion_cost == pytest.approx(10 * 1.1e-05)
