"""
Tests for Cafe24 LLM Router provider configuration and integration.
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


class TestCafe24ProviderConfig:
    """Test Cafe24 LLM Router provider configuration"""

    def test_cafe24_in_provider_list(self):
        """Test that cafe24 is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "CAFE24")
        assert LlmProviders.CAFE24.value == "cafe24"
        assert "cafe24" in litellm.provider_list

    def test_cafe24_json_config_exists(self):
        """Test that cafe24 is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("cafe24")

        cafe24 = JSONProviderRegistry.get("cafe24")
        assert cafe24 is not None
        assert cafe24.base_url == "https://llm-router.cafe24.com/api/v1"
        assert cafe24.api_key_env == "CAFE24_API_KEY"

    def test_cafe24_provider_resolution(self):
        """Test that provider resolution finds cafe24.

        Cafe24 LLM Router model IDs contain slashes
        (e.g. deepseek-ai/DeepSeek-V3.1), so this also verifies that
        nested slashes in the model name resolve correctly.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="cafe24/deepseek-ai/DeepSeek-V3.1",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-ai/DeepSeek-V3.1"
        assert provider == "cafe24"
        assert api_base == "https://llm-router.cafe24.com/api/v1"

    def test_cafe24_router_config(self):
        """Test that cafe24 can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "qwen3-8b",
                    "litellm_params": {
                        "model": "cafe24/Qwen/Qwen3-8B",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "qwen3-8b"


if __name__ == "__main__":
    print("Testing Cafe24 LLM Router Provider...")

    test_config = TestCafe24ProviderConfig()

    print("\n1. Testing provider in list...")
    test_config.test_cafe24_in_provider_list()
    print("   ✓ cafe24 in provider list")

    print("\n2. Testing JSON config...")
    test_config.test_cafe24_json_config_exists()
    print("   ✓ cafe24 JSON config loaded")

    print("\n3. Testing provider resolution...")
    test_config.test_cafe24_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n4. Testing router configuration...")
    test_config.test_cafe24_router_config()
    print("   ✓ Router configuration works")

    print("\n" + "=" * 50)
    print("✓ All configuration tests passed!")
    print("=" * 50)
