"""
Tests for the Kimchi provider identity.

Kimchi exposes an OpenAI-compatible /v1/chat/completions surface at
https://llm.kimchi.dev/openai/v1. Its provider identity must resolve to a
distinct `kimchi` slug so OpenAI-specific pricing and provider-level reporting
never apply to traffic routed through Kimchi.
"""

import litellm


class TestKimchiProviderIdentity:
    def test_kimchi_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.KIMCHI.value == "kimchi"
        assert "kimchi" in litellm.provider_list

    def test_kimchi_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        kimchi = JSONProviderRegistry.get("kimchi")
        assert kimchi is not None
        assert kimchi.base_url == "https://llm.kimchi.dev/openai/v1"
        assert kimchi.api_key_env == "KIMCHI_API_KEY"
        assert kimchi.api_base_env == "KIMCHI_API_BASE"

    def test_kimchi_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "kimchi" in openai_compatible_providers

    def test_prefixed_model_resolves_to_kimchi_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="kimchi/kimi-k3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "kimi-k3"
        assert provider == "kimchi"
        assert api_base == "https://llm.kimchi.dev/openai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="kimchi/kimi-k3",
            custom_llm_provider=None,
            api_base="https://kimchi.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "kimchi"
        assert api_base == "https://kimchi.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_kimchi(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("KIMCHI_API_KEY", "sk-test")

        _, provider, api_key, api_base = get_llm_provider(
            model="kimi-k3",
            custom_llm_provider=None,
            api_base="https://llm.kimchi.dev/openai/v1",
            api_key=None,
        )

        assert provider == "kimchi"
        assert api_base == "https://llm.kimchi.dev/openai/v1"
        assert api_key == "sk-test"
