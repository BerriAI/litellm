"""
Tests for go2llm provider configuration and integration.
"""

import litellm


class TestGo2llmProviderConfig:
    """Test go2llm provider configuration"""

    def test_go2llm_in_provider_list(self):
        """Test that go2llm is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "GO2LLM")
        assert LlmProviders.GO2LLM.value == "go2llm"
        assert "go2llm" in litellm.provider_list

    def test_go2llm_json_config_exists(self):
        """Test that go2llm is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("go2llm")

        go2llm = JSONProviderRegistry.get("go2llm")
        assert go2llm is not None
        assert go2llm.base_url == "https://go2llm.tech/v1"
        assert go2llm.api_key_env == "GO2LLM_API_KEY"
        assert go2llm.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_go2llm_in_openai_compatible_providers(self):
        """Test that go2llm is in the openai_compatible_providers list"""
        from litellm.constants import openai_compatible_providers

        assert "go2llm" in openai_compatible_providers

    def test_go2llm_provider_resolution(self):
        """Test that provider resolution finds go2llm and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="go2llm/claude-haiku-4-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-haiku-4-5"
        assert provider == "go2llm"
        assert api_base == "https://go2llm.tech/v1"

    def test_go2llm_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="go2llm/claude-haiku-4-5",
            custom_llm_provider=None,
            api_base="https://self-hosted.example/v1",
            api_key="gk_sk_test",
        )

        assert model == "claude-haiku-4-5"
        assert provider == "go2llm"
        assert api_base == "https://self-hosted.example/v1"
        assert api_key == "gk_sk_test"

    def test_go2llm_resolution_from_base_url(self):
        """A caller who passes only the base url still lands on the go2llm provider"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="claude-haiku-4-5",
            custom_llm_provider=None,
            api_base="https://go2llm.tech/v1",
            api_key="gk_sk_test",
        )

        assert model == "claude-haiku-4-5"
        assert provider == "go2llm"
        assert api_base == "https://go2llm.tech/v1"
