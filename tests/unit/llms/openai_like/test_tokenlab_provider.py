"""
Tests for TokenLab provider configuration (JSON-configured OpenAI-like provider).
"""

import litellm


class TestTokenLabProviderConfig:
    def test_tokenlab_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.TOKENLAB.value == "tokenlab"
        assert "tokenlab" in litellm.provider_list

    def test_tokenlab_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("tokenlab")
        tokenlab = JSONProviderRegistry.get("tokenlab")
        assert tokenlab is not None
        assert tokenlab.base_url == "https://api.tokenlab.sh/v1"
        assert tokenlab.api_key_env == "TOKENLAB_API_KEY"
        assert tokenlab.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_tokenlab_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="tokenlab/gpt-5.4-mini",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-5.4-mini"
        assert provider == "tokenlab"
        assert api_base == "https://api.tokenlab.sh/v1"

    def test_tokenlab_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _model, provider, api_key, api_base = get_llm_provider(
            model="tokenlab/gpt-5.4-mini",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "tokenlab"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"
