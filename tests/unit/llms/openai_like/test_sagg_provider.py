"""
Tests for SAGG provider configuration and integration.

SAGG (https://api.privatedeskai.com) is an OpenAI-compatible LLM
inference gateway with automatic multi-provider failover.
"""

import os
import sys
from unittest.mock import MagicMock, patch

try:
    import pytest
except ImportError:
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestSaggProviderConfig:
    """Test SAGG provider configuration"""

    def test_sagg_in_provider_list(self):
        """Test that sagg is in the provider list"""
        from litellm import LlmProviders

        # Verify sagg is in the enum
        assert hasattr(LlmProviders, "SAGG")
        assert LlmProviders.SAGG.value == "sagg"

        # Verify it's in the provider list
        assert "sagg" in litellm.provider_list

    def test_sagg_json_config_exists(self):
        """Test that sagg is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify sagg is loaded
        assert JSONProviderRegistry.exists("sagg")

        # Get sagg config
        sagg = JSONProviderRegistry.get("sagg")
        assert sagg is not None
        assert sagg.base_url == "https://api.privatedeskai.com/v1"
        assert sagg.api_key_env == "SAGG_API_KEY"
        assert sagg.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_sagg_provider_resolution(self):
        """Test that provider resolution finds sagg - the real model id
        contains its own slash (deepseek-ai/DeepSeek-V4-Flash-0731), so
        this also confirms provider-prefix splitting only splits on the
        FIRST slash, same as e.g. pinstripes/ps/glm-4.5-air elsewhere in
        this same registry."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sagg/deepseek-ai/DeepSeek-V4-Flash-0731",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-ai/DeepSeek-V4-Flash-0731"
        assert provider == "sagg"
        assert api_base == "https://api.privatedeskai.com/v1"

    def test_sagg_router_config(self):
        """Test that sagg can be used in Router configuration"""
        from litellm import Router

        # This should not raise "Unsupported provider - sagg"
        router = Router(
            model_list=[
                {
                    "model_name": "sagg-deepseek",
                    "litellm_params": {
                        "model": "sagg/deepseek-ai/DeepSeek-V4-Flash-0731",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        # Verify the deployment was created successfully
        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "sagg-deepseek"

    def test_sagg_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens for
        sagg - SAGG's own request parser only recognizes max_tokens
        (cmd/gateway/billing.go's chatRequestForBilling struct has no
        max_completion_tokens field at all), so a caller sending the
        newer OpenAI param name needs this mapping to actually take
        effect server-side, not be silently dropped."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("sagg")
        config_class = create_config_class(provider)
        config = config_class()

        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(
            non_default_params, optional_params, "deepseek-ai/DeepSeek-V4-Flash-0731", False
        )

        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result
        assert result["temperature"] == 0.7


class TestSaggIntegration:
    """Integration tests for SAGG provider"""

    def test_sagg_completion_basic(self):
        """Test basic completion call to SAGG"""
        # Skip test if API key not set in environment
        if not os.environ.get("SAGG_API_KEY"):
            if pytest:
                pytest.skip("SAGG_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="sagg/deepseek-ai/DeepSeek-V4-Flash-0731",
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'test successful' and nothing else",
                    }
                ],
                max_tokens=10,
            )

            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert hasattr(response.choices[0].message, "content")
            assert response.choices[0].message.content is not None

            content = response.choices[0].message.content.lower()
            assert len(content) > 0

            print(f"✓ SAGG completion successful: {response.choices[0].message.content}")

        except Exception as e:
            if pytest:
                pytest.fail(f"SAGG completion failed: {str(e)}")
            else:
                raise


if __name__ == "__main__":
    print("Testing SAGG Provider...")

    test_config = TestSaggProviderConfig()

    print("\n1. Testing provider in list...")
    test_config.test_sagg_in_provider_list()
    print("   ✓ sagg in provider list")

    print("\n2. Testing JSON config...")
    test_config.test_sagg_json_config_exists()
    print("   ✓ sagg JSON config loaded")

    print("\n3. Testing provider resolution...")
    test_config.test_sagg_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n4. Testing router configuration...")
    test_config.test_sagg_router_config()
    print("   ✓ Router configuration works")

    print("\n5. Testing parameter mapping...")
    test_config.test_sagg_parameter_mapping()
    print("   ✓ Parameter mapping works")

    print("\n" + "=" * 50)
    print("✓ All configuration tests passed!")
    print("=" * 50)
