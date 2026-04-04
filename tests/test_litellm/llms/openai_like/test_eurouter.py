"""
Tests for EUrouter provider configuration and integration.

EUrouter is an EU-hosted, GDPR-compliant AI routing service that is
OpenAI-compatible (similar to OpenRouter).
"""

import os
import sys

try:
    import pytest
except ImportError:
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestEUrouterProviderConfig:
    """Test EUrouter provider configuration"""

    def test_eurouter_in_provider_list(self):
        """Test that eurouter is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "EUROUTER")
        assert LlmProviders.EUROUTER.value == "eurouter"
        assert "eurouter" in litellm.provider_list

    def test_eurouter_json_config_exists(self):
        """Test that eurouter is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("eurouter")

        eurouter = JSONProviderRegistry.get("eurouter")
        assert eurouter is not None
        assert eurouter.base_url == "https://api.eurouter.ai/api/v1"
        assert eurouter.api_key_env == "EUROUTER_API_KEY"

    def test_eurouter_provider_resolution(self):
        """Test that provider resolution finds eurouter"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="eurouter/mistral/mistral-large-3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "eurouter"
        assert api_base == "https://api.eurouter.ai/api/v1"

    def test_eurouter_router_config(self):
        """Test that eurouter can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "eurouter/mistral/mistral-large-3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "gpt-4o"
