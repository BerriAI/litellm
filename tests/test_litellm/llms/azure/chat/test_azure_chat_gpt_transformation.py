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


def test_azure_strips_output_config_with_effort():
    """Test that Azure strips output_config containing effort parameter.

    See: https://github.com/BerriAI/litellm/issues/22963
    """
    config = AzureOpenAIConfig()
    request = config.transform_request(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "output_config": {"effort": "high"},
            "temperature": 0.7,
        },
        litellm_params={},
        headers={},
    )
    assert "output_config" not in request
    assert request["temperature"] == 0.7


def test_azure_strips_output_config_with_format():
    """Test that Azure strips output_config containing format/schema.

    See: https://github.com/BerriAI/litellm/issues/22963
    """
    config = AzureOpenAIConfig()
    request = config.transform_request(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            },
        },
        litellm_params={},
        headers={},
    )
    assert "output_config" not in request


def test_azure_works_without_output_config():
    """Test that Azure requests work normally when output_config is not present.

    output_config is only sent when extended thinking or effort is enabled.
    Most requests won't include it.
    """
    config = AzureOpenAIConfig()
    request = config.transform_request(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "temperature": 0.7,
            "max_tokens": 100,
        },
        litellm_params={},
        headers={},
    )
    assert "output_config" not in request
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 100
