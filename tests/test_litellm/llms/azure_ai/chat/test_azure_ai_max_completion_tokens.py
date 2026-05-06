"""
Unit tests for AzureAIStudioConfig.map_openai_params — ensures max_completion_tokens
is correctly renamed to max_tokens for Azure AI Foundry's Model Inference endpoint.

Regression tests for https://github.com/BerriAI/litellm/issues/26322
"""
import pytest

from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.fixture
def config():
    return AzureAIStudioConfig()


class TestAzureAIMapOpenAIParamsMaxTokens:
    def test_max_completion_tokens_renamed_to_max_tokens(self, config):
        """max_completion_tokens must be rewritten to max_tokens for the Foundry endpoint."""
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 1000},
            optional_params={},
            model="mistral-large-3",
            drop_params=False,
        )
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result

    def test_plain_max_tokens_passes_through(self, config):
        """Plain max_tokens must still be forwarded unchanged."""
        result = config.map_openai_params(
            non_default_params={"max_tokens": 500},
            optional_params={},
            model="mistral-large-3",
            drop_params=False,
        )
        assert result.get("max_tokens") == 500
        assert "max_completion_tokens" not in result

    def test_max_completion_tokens_wins_when_both_present(self, config):
        """When both keys are provided, max_completion_tokens takes precedence."""
        result = config.map_openai_params(
            non_default_params={"max_tokens": 500, "max_completion_tokens": 1000},
            optional_params={},
            model="mistral-large-3",
            drop_params=False,
        )
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result

    def test_does_not_mutate_caller_dict(self, config):
        """The incoming non_default_params dict must not be mutated."""
        non_default_params = {"max_completion_tokens": 1000}
        config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="mistral-large-3",
            drop_params=False,
        )
        assert non_default_params == {"max_completion_tokens": 1000}

    def test_no_max_tokens_key_untouched(self, config):
        """Params with no token-limit key must pass through unchanged."""
        result = config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="mistral-large-3",
            drop_params=False,
        )
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result

    @pytest.mark.parametrize("model", ["o3-mini", "gpt-5", "mistral-large-3", "cohere-command"])
    def test_rename_applies_to_all_models(self, config, model):
        """The rename must apply regardless of model name."""
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 2048},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert result.get("max_tokens") == 2048
        assert "max_completion_tokens" not in result
