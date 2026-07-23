import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import litellm
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig


class TestAzureOpenAIConfig:
    def test_is_response_format_supported_model(self):
        config = AzureOpenAIConfig()
        # New logic: Azure deployment names with suffixes and prefixes
        assert config._is_response_format_supported_model("azure/gpt-4.1-suffix")
        assert config._is_response_format_supported_model("gpt-4.1-suffix")
        assert config._is_response_format_supported_model("azure/gpt-4-1-suffix")
        assert config._is_response_format_supported_model("gpt-4-1-suffix")
        # 4o models (should always be supported)
        assert config._is_response_format_supported_model("gpt-4o")
        assert config._is_response_format_supported_model("azure/gpt-4o-custom")
        # Backwards compatibility: base names
        assert config._is_response_format_supported_model("gpt-4.1")
        assert config._is_response_format_supported_model("gpt-4-1")
        # Negative test: clearly unsupported model
        assert not config._is_response_format_supported_model("gpt-3.5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo")

    def test_prompt_cache_key_supported(self):
        """Test that 'prompt_cache_key' is in supported params for Azure OpenAI chat completion models.

        OpenAI's Chat Completions API supports prompt_cache_key for cache routing optimization.
        """
        config = AzureOpenAIConfig()
        supported_params = config.get_supported_openai_params("gpt-4.1-nano")
        assert "prompt_cache_key" in supported_params

        supported_params = config.get_supported_openai_params("gpt-4.1")
        assert "prompt_cache_key" in supported_params


def test_map_openai_params_with_preview_api_version():
    config = AzureOpenAIConfig()
    non_default_params = {
        "response_format": {"type": "json_object"},
    }
    optional_params = {}
    model = "azure/gpt-4-1"
    drop_params = False
    api_version = "preview"
    assert config.map_openai_params(non_default_params, optional_params, model, drop_params, api_version)


class TestAzureMaxTokensConflict:
    """Azure rejects requests carrying both max_tokens and max_completion_tokens.

    Regression coverage for https://github.com/BerriAI/litellm/issues/31614 -
    map_openai_params must never emit both; max_completion_tokens is preferred.
    """

    def _map(self, non_default_params, optional_params=None):
        return AzureOpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params if optional_params is not None else {},
            model="azure/gpt-4.1",
            drop_params=False,
        )

    def test_max_tokens_alone_is_preserved(self):
        result = self._map({"max_tokens": 100})
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

    def test_max_completion_tokens_alone_is_preserved(self):
        result = self._map({"max_completion_tokens": 100})
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_both_in_non_default_params_keeps_only_max_completion_tokens(self):
        result = self._map({"max_tokens": 50, "max_completion_tokens": 100})
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_preseeded_max_tokens_is_dropped_when_max_completion_tokens_present(self):
        result = self._map({"max_completion_tokens": 100}, optional_params={"max_tokens": 50})
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_max_completion_tokens_wins_even_when_smaller(self):
        result = self._map({"max_tokens": 1000, "max_completion_tokens": 5})
        assert result["max_completion_tokens"] == 5
        assert "max_tokens" not in result


class TestAzureMaxTokensConflictGetOptionalParams:
    """End-to-end coverage through the public get_optional_params entrypoint."""

    def _get(self, **kwargs):
        return litellm.utils.get_optional_params(model="gpt-4.1", custom_llm_provider="azure", **kwargs)

    def test_max_tokens_alone_does_not_add_max_completion_tokens(self):
        result = self._get(max_tokens=100)
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

    def test_max_completion_tokens_alone_does_not_add_max_tokens(self):
        result = self._get(max_completion_tokens=100)
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_both_keeps_only_max_completion_tokens(self):
        result = self._get(max_tokens=50, max_completion_tokens=100)
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result
