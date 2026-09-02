"""
Tests for SAGG provider configuration and integration.

SAGG (https://api.privatedeskai.com) is an OpenAI-compatible LLM
inference gateway with automatic multi-provider failover.
"""

import os
import sys

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestSaggProviderConfig:
    def test_sagg_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "SAGG")
        assert LlmProviders.SAGG.value == "sagg"
        assert "sagg" in litellm.provider_list

    def test_sagg_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("sagg")

        sagg = JSONProviderRegistry.get("sagg")
        assert sagg is not None
        assert sagg.base_url == "https://api.privatedeskai.com/v1"
        assert sagg.api_key_env == "SAGG_API_KEY"
        assert sagg.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_sagg_provider_resolution(self):
        """The real model id contains its own slash
        (deepseek-ai/DeepSeek-V4-Flash-0731), so this also confirms
        provider-prefix splitting only splits on the FIRST slash, same
        as e.g. pinstripes/ps/glm-4.5-air elsewhere in this registry."""
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
        from litellm import Router

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

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "sagg-deepseek"

    def test_sagg_parameter_mapping(self):
        """SAGG's own request parser only recognizes max_tokens, so a
        caller sending max_completion_tokens needs this mapping to
        actually take effect server-side rather than being dropped."""
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
