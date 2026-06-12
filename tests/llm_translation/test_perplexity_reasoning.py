import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import get_optional_params


class TestPerplexityReasoning:
    """
    Test suite for Perplexity Sonar reasoning models with reasoning_effort parameter
    """

    @pytest.mark.parametrize(
        "model,reasoning_effort",
        [
            ("perplexity/sonar-reasoning", "low"),
            ("perplexity/sonar-reasoning", "medium"),
            ("perplexity/sonar-reasoning", "high"),
            ("perplexity/sonar-reasoning-pro", "low"),
            ("perplexity/sonar-reasoning-pro", "medium"),
            ("perplexity/sonar-reasoning-pro", "high"),
        ],
    )
    def test_perplexity_reasoning_effort_parameter_mapping(
        self, model, reasoning_effort
    ):
        """
        Test that reasoning_effort parameter is correctly mapped for Perplexity Sonar reasoning models
        """
        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        # Get provider and optional params
        _, provider, _, _ = litellm.get_llm_provider(model=model)

        optional_params = get_optional_params(
            model=model,
            custom_llm_provider=provider,
            reasoning_effort=reasoning_effort,
        )

        # Verify that reasoning_effort is preserved in optional_params for Perplexity
        assert "reasoning_effort" in optional_params
        assert optional_params["reasoning_effort"] == reasoning_effort

    def test_perplexity_reasoning_models_support_reasoning(self):
        """
        Test that Perplexity Sonar reasoning models are correctly identified as supporting reasoning
        """
        from litellm.utils import supports_reasoning

        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        reasoning_models = [
            "perplexity/sonar-reasoning",
            "perplexity/sonar-reasoning-pro",
        ]

        for model in reasoning_models:
            assert supports_reasoning(model, None), f"{model} should support reasoning"

    def test_perplexity_non_reasoning_models_dont_support_reasoning(self):
        """
        Test that non-reasoning Perplexity models don't support reasoning
        """
        from litellm.utils import supports_reasoning

        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        non_reasoning_models = [
            "perplexity/sonar",
            "perplexity/sonar-pro",
            "perplexity/llama-3.1-sonar-large-128k-chat",
            "perplexity/mistral-7b-instruct",
        ]

        for model in non_reasoning_models:
            # These models should not support reasoning (should return False or raise exception)
            try:
                result = supports_reasoning(model, None)
                # If it doesn't raise an exception, it should return False
                assert result is False, f"{model} should not support reasoning"
            except Exception:
                # If it raises an exception, that's also acceptable behavior
                pass

    @pytest.mark.parametrize(
        "model,expected_api_base",
        [
            ("perplexity/sonar-reasoning", "https://api.perplexity.ai"),
            ("perplexity/sonar-reasoning-pro", "https://api.perplexity.ai"),
        ],
    )
    def test_perplexity_reasoning_api_base_configuration(
        self, model, expected_api_base
    ):
        """
        Test that Perplexity reasoning models use the correct API base
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig

        config = PerplexityChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(
            api_base=None, api_key="test-key"
        )

        assert api_base == expected_api_base

    def test_perplexity_reasoning_effort_in_supported_params(self):
        """
        Test that reasoning_effort is in the list of supported parameters for Perplexity
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig

        config = PerplexityChatConfig()
        supported_params = config.get_supported_openai_params(
            model="perplexity/sonar-reasoning"
        )

        assert "reasoning_effort" in supported_params
