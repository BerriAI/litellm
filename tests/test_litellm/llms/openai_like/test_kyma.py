"""
Tests for Kyma API provider configuration and integration.
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


class TestKymaProviderConfig:
    """Test Kyma API provider configuration"""

    def test_kyma_in_provider_list(self):
        """Test that kyma is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "KYMA")
        assert LlmProviders.KYMA.value == "kyma"
        assert "kyma" in litellm.provider_list

    def test_kyma_json_config_exists(self):
        """Test that kyma is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("kyma")

        kyma = JSONProviderRegistry.get("kyma")
        assert kyma is not None
        assert kyma.base_url == "https://kymaapi.com/v1"
        assert kyma.api_key_env == "KYMA_API_KEY"

    def test_kyma_provider_resolution(self):
        """Test that provider resolution finds kyma"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="kyma/llama-3.3-70b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "llama-3.3-70b"
        assert provider == "kyma"
        assert api_base == "https://kymaapi.com/v1"

    def test_kyma_router_config(self):
        """Test that kyma can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "llama-3.3-70b",
                    "litellm_params": {
                        "model": "kyma/llama-3.3-70b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "llama-3.3-70b"

    def test_kyma_completion_skipped_without_key(self):
        """Test that completion is skipped when API key is not set"""
        if not os.environ.get("KYMA_API_KEY"):
            if pytest:
                pytest.skip("KYMA_API_KEY not set")
            return

        response = litellm.completion(
            model="kyma/llama-3.3-70b",
            messages=[{"role": "user", "content": "Say 'test successful' and nothing else"}],
            max_tokens=10,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None


if __name__ == "__main__":
    print("Testing Kyma API Provider...")

    test_config = TestKymaProviderConfig()

    print("\n1. Testing provider in list...")
    test_config.test_kyma_in_provider_list()
    print("   OK kyma in provider list")

    print("\n2. Testing JSON config...")
    test_config.test_kyma_json_config_exists()
    print("   OK kyma JSON config loaded")

    print("\n3. Testing provider resolution...")
    test_config.test_kyma_provider_resolution()
    print("   OK Provider resolution works")

    print("\n4. Testing router configuration...")
    test_config.test_kyma_router_config()
    print("   OK Router configuration works")

    print("\n" + "=" * 50)
    print("All configuration tests passed!")
    print("=" * 50)
