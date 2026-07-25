"""
Tests for the OpenInfer LLM provider configuration and integration.
"""

import litellm


class TestOpenInferProviderConfig:
    def test_openinfer_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.OPENINFER.value == "openinfer"
        assert "openinfer" in litellm.provider_list

    def test_openinfer_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("openinfer")
        assert provider is not None
        assert provider.base_url == "https://api.openinfer.ai/v1"
        assert provider.api_key_env == "OPENINFER_API_KEY"
        assert provider.api_base_env == "OPENINFER_API_BASE"
        assert not JSONProviderRegistry.supports_responses_api("openinfer")

    def test_openinfer_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "openinfer" in openai_compatible_providers

    def test_provider_prefixed_model_routes_to_openinfer(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openinfer/llama-3.1-8b-instruct",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "llama-3.1-8b-instruct"
        assert provider == "openinfer"
        assert api_key == "sk-test"
        assert api_base == "https://api.openinfer.ai/v1"

    def test_api_key_and_base_resolved_from_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")
        monkeypatch.setenv("OPENINFER_API_BASE", "https://proxy.internal/v1")

        _, provider, api_key, api_base = get_llm_provider(
            model="openinfer/llama-3.1-8b-instruct",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "openinfer"
        assert api_key == "sk-env-key"
        assert api_base == "https://proxy.internal/v1"

    def test_url_autodetection_from_api_base(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")

        _, provider, api_key, api_base = get_llm_provider(
            model="llama-3.1-8b-instruct",
            custom_llm_provider=None,
            api_base="https://api.openinfer.ai/v1",
            api_key=None,
        )

        assert provider == "openinfer"
        assert api_key == "sk-env-key"

    def test_chat_completions_url(self):
        config = litellm.ProviderConfigManager.get_provider_chat_config(
            model="llama-3.1-8b-instruct", provider=litellm.LlmProviders.OPENINFER
        )
        assert config is not None
        assert (
            config.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="llama-3.1-8b-instruct",
                optional_params={},
                litellm_params={},
            )
            == "https://api.openinfer.ai/v1/chat/completions"
        )
