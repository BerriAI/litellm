"""
Tests for Charity Engine provider configuration and integration.
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


class TestCharityEngineProviderConfig:
    """Test Charity Engine provider configuration"""

    def test_charity_engine_in_provider_list(self):
        """Test that charity_engine is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "CHARITY_ENGINE")
        assert LlmProviders.CHARITY_ENGINE.value == "charity_engine"
        assert "charity_engine" in litellm.provider_list

    def test_charity_engine_json_config_exists(self):
        """Test that charity_engine is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("charity_engine")

        charity_engine = JSONProviderRegistry.get("charity_engine")
        assert charity_engine is not None
        assert charity_engine.base_url == "https://api.charityengine.services/remotejobs/v2/inference/"
        assert charity_engine.api_key_env == "CHARITY_ENGINE_API_KEY"
        assert charity_engine.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_charity_engine_provider_resolution(self):
        """Test that provider resolution finds charity_engine"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="charity_engine/gemma3:270m",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gemma3:270m"
        assert provider == "charity_engine"
        assert api_base == "https://api.charityengine.services/remotejobs/v2/inference/"

    def test_charity_engine_router_config(self):
        """Test that charity_engine can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "gemma3-270m",
                    "litellm_params": {
                        "model": "charity_engine/gemma3:270m",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "gemma3-270m"


if __name__ == "__main__":
    print("Testing Charity Engine Provider...")

    test_config = TestCharityEngineProviderConfig()

    print("\n1. Testing provider in list...")
    test_config.test_charity_engine_in_provider_list()
    print("   ✓ charity_engine in provider list")

    print("\n2. Testing JSON config...")
    test_config.test_charity_engine_json_config_exists()
    print("   ✓ charity_engine JSON config loaded")

    print("\n3. Testing provider resolution...")
    test_config.test_charity_engine_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n4. Testing router configuration...")
    test_config.test_charity_engine_router_config()
    print("   ✓ Router configuration works")

    print("\n" + "=" * 50)
    print("✓ All configuration tests passed!")
    print("=" * 50)
