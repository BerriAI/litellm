import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

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
    assert config.map_openai_params(
        non_default_params, optional_params, model, drop_params, api_version
    )
