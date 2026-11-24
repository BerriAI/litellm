"""
Tests for OpenImpact provider configuration and integration.
"""

import litellm


class TestOpenImpactProviderConfig:
    """Test OpenImpact provider configuration"""

    def test_openimpact_in_provider_list(self):
        """Test that openimpact is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "OPENIMPACT")
        assert LlmProviders.OPENIMPACT.value == "openimpact"
        assert "openimpact" in litellm.provider_list

    def test_openimpact_json_config_exists(self):
        """Test that openimpact is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("openimpact")

        openimpact = JSONProviderRegistry.get("openimpact")
        assert openimpact is not None
        assert openimpact.base_url == "https://api.oimpact.ai/v1"
        assert openimpact.api_key_env == "OPENIMPACT_API_KEY"
        assert openimpact.api_base_env == "OPENIMPACT_API_BASE"

    def test_openimpact_provider_resolution(self):
        """Test that provider resolution finds openimpact and the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openimpact/glm-5.3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "glm-5.3"
        assert provider == "openimpact"
        assert api_base == "https://api.oimpact.ai/v1"

    def test_openimpact_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openimpact/glm-5.3",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "openimpact"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"
