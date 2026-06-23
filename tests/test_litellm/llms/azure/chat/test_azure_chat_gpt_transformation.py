import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from litellm.utils import _invalidate_model_cost_lowercase_map


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


def test_map_openai_params_translates_max_tokens_for_flagged_model(monkeypatch):
    """Models flagged in the cost map translate max_tokens -> max_completion_tokens.

    Azure gpt-chat-latest rejects `max_tokens` and requires
    `max_completion_tokens`, but it does not route through the GPT-5/o-series
    config. The `map_max_tokens_to_max_completion_tokens` flag drives the
    translation for normal chat completions too (not just health checks).
    """
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    config = AzureOpenAIConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_tokens": 42},
        optional_params={},
        model="azure/gpt-chat-latest",
        drop_params=False,
    )
    assert optional_params["max_completion_tokens"] == 42
    assert "max_tokens" not in optional_params


def test_map_openai_params_keeps_max_tokens_for_unflagged_model(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    config = AzureOpenAIConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_tokens": 42},
        optional_params={},
        model="azure/gpt-4o",
        drop_params=False,
    )
    assert optional_params["max_tokens"] == 42
    assert "max_completion_tokens" not in optional_params
