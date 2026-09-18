"""
Tests for Pinstripes provider configuration and integration.
"""

import litellm


class TestPinstripeProviderConfig:
    """Test Pinstripes provider configuration"""

    def test_pinstripes_in_provider_list(self):
        """Test that pinstripes is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "PINSTRIPES")
        assert LlmProviders.PINSTRIPES.value == "pinstripes"
        assert "pinstripes" in litellm.provider_list

    def test_pinstripes_json_config_exists(self):
        """Test that pinstripes is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("pinstripes")

        pinstripes = JSONProviderRegistry.get("pinstripes")
        assert pinstripes is not None
        assert pinstripes.base_url == "https://pinstripes.io/v1"
        assert pinstripes.api_key_env == "PINSTRIPES_API_KEY"
        assert pinstripes.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_pinstripes_in_openai_compatible_providers(self):
        """Test that pinstripes is in the openai_compatible_providers list"""
        from litellm.constants import openai_compatible_providers

        assert "pinstripes" in openai_compatible_providers

    def test_pinstripes_provider_resolution(self):
        """Test that provider resolution finds pinstripes and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pinstripes/ps/glm-4.5-air",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "ps/glm-4.5-air"
        assert provider == "pinstripes"
        assert api_base == "https://pinstripes.io/v1"

    def test_pinstripes_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pinstripes/ps/glm-4.5-air",
            custom_llm_provider=None,
            api_base="https://custom.pinstripes.io/v1",
            api_key="sk-test",
        )

        assert provider == "pinstripes"
        assert api_base == "https://custom.pinstripes.io/v1"
        assert api_key == "sk-test"

    def test_pinstripes_url_autodetection(self):
        """Test that api_base=pinstripes.io/v1 auto-sets custom_llm_provider=pinstripes"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="ps/glm-4.5-air",
            custom_llm_provider=None,
            api_base="https://pinstripes.io/v1",
            api_key=None,
        )
        assert provider == "pinstripes"
        assert api_base == "https://pinstripes.io/v1"

    def test_pinstripes_router_config(self):
        """Test that pinstripes can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "pinstripes-chat",
                    "litellm_params": {
                        "model": "pinstripes/ps/glm-4.5-air",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "pinstripes-chat"
