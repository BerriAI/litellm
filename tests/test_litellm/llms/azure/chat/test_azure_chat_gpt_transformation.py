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


def test_transform_request_hoists_tool_message_image():
    """Azure builds its request via convert_to_azure_openai_messages without the
    OpenAIGPTConfig._transform_messages pipeline, so transform_request must hoist
    tool-message images itself; Azure rejects non-text tool content."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    messages = [
        {"role": "user", "content": "read the screenshot"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
        },
    ]

    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    transformed = request["messages"]
    assert [m.get("role") for m in transformed] == ["user", "assistant", "tool", "user"]
    assert isinstance(transformed[2]["content"], str)
    assert transformed[3]["content"] == [{"type": "image_url", "image_url": {"url": data_uri}}]
