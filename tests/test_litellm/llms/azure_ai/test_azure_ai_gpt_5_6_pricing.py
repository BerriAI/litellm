"""
Test Azure AI GPT-5.6 family pricing and cost calculations.
Issue: #37282 - Missing pricing for GPT 5.6 models on azure foundry
"""

import pytest
import litellm
from litellm.llms.azure_ai.cost_calculator import cost_per_token
from litellm.types.utils import Usage


class TestAzureAIGPT56Pricing:
    """Test pricing and cost calculation for Azure AI GPT 5.6 model family."""

    @pytest.mark.parametrize(
        "model,expected_input_cost,expected_output_cost,expected_cache_read",
        [
            ("azure_ai/gpt-5.6", 5e-06, 3e-05, 5e-07),
            ("azure_ai/gpt-5.6-sol", 5e-06, 3e-05, 5e-07),
            ("azure_ai/gpt-5.6-terra", 2e-06, 1.2e-05, 2e-07),
            ("azure_ai/gpt-5.6-luna", 2e-07, 1.2e-06, 2e-08),
            ("azure_ai/openai/gpt-5.6", 5e-06, 3e-05, 5e-07),
            ("azure_ai/openai/gpt-5.6-sol", 5e-06, 3e-05, 5e-07),
            ("azure_ai/openai/gpt-5.6-terra", 2e-06, 1.2e-05, 2e-07),
            ("azure_ai/openai/gpt-5.6-luna", 2e-07, 1.2e-06, 2e-08),
        ],
    )
    def test_gpt_5_6_model_cost_registration(
        self, model, expected_input_cost, expected_output_cost, expected_cache_read
    ):
        """Verify model pricing is properly registered in model_cost for azure_ai provider and aliases."""
        model_info = litellm.model_cost.get(model)
        assert model_info is not None, f"Model {model} must be registered in litellm.model_cost"
        assert model_info.get("litellm_provider") == "azure_ai"
        assert model_info.get("input_cost_per_token") == pytest.approx(expected_input_cost)
        assert model_info.get("output_cost_per_token") == pytest.approx(expected_output_cost)
        assert model_info.get("cache_read_input_token_cost") == pytest.approx(expected_cache_read)
        assert model_info.get("supports_vision") is True
        assert model_info.get("supports_reasoning") is True
        assert model_info.get("supports_function_calling") is True

    def test_gpt_5_6_cost_calculation(self):
        """Verify cost calculation for standard usage with azure_ai/gpt-5.6-sol."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        prompt_cost, completion_cost = cost_per_token(
            model="azure_ai/gpt-5.6-sol",
            usage=usage,
        )
        expected_prompt = 1000 * 5e-06
        expected_completion = 500 * 3e-05
        assert prompt_cost == pytest.approx(expected_prompt)
        assert completion_cost == pytest.approx(expected_completion)

    def test_gpt_5_6_alias_cost_calculation(self):
        """Verify cost calculation for standard usage with azure_ai/openai/gpt-5.6-sol alias."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        prompt_cost, completion_cost = cost_per_token(
            model="azure_ai/openai/gpt-5.6-sol",
            usage=usage,
        )
        expected_prompt = 1000 * 5e-06
        expected_completion = 500 * 3e-05
        assert prompt_cost == pytest.approx(expected_prompt)
        assert completion_cost == pytest.approx(expected_completion)

    def test_gpt_5_6_terra_cost_calculation(self):
        """Verify cost calculation for standard usage with azure_ai/gpt-5.6-terra."""
        usage = Usage(
            prompt_tokens=2000,
            completion_tokens=1000,
            total_tokens=3000,
        )
        prompt_cost, completion_cost = cost_per_token(
            model="azure_ai/gpt-5.6-terra",
            usage=usage,
        )
        expected_prompt = 2000 * 2e-06
        expected_completion = 1000 * 1.2e-05
        assert prompt_cost == pytest.approx(expected_prompt)
        assert completion_cost == pytest.approx(expected_completion)

    def test_gpt_5_6_luna_cost_calculation(self):
        """Verify cost calculation for standard usage with azure_ai/gpt-5.6-luna."""
        usage = Usage(
            prompt_tokens=5000,
            completion_tokens=2000,
            total_tokens=7000,
        )
        prompt_cost, completion_cost = cost_per_token(
            model="azure_ai/gpt-5.6-luna",
            usage=usage,
        )
        expected_prompt = 5000 * 2e-07
        expected_completion = 2000 * 1.2e-06
        assert prompt_cost == pytest.approx(expected_prompt)
        assert completion_cost == pytest.approx(expected_completion)
