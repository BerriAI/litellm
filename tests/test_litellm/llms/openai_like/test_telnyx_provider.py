"""
Tests for Telnyx provider configuration via JSON providers system.
"""

import os
import sys

import pytest

from unittest.mock import MagicMock, patch

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestTelnyxProviderLoader:
    """Test Telnyx provider loading and configuration"""

    def test_telnyx_provider_exists(self):
        """Test that Telnyx is registered as a JSON provider"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("telnyx")

    def test_telnyx_provider_config(self):
        """Test that Telnyx provider config has correct values"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        telnyx = JSONProviderRegistry.get("telnyx")
        assert telnyx is not None
        assert telnyx.base_url == "https://api.telnyx.com/v2/ai"
        assert telnyx.api_key_env == "TELNYX_API_KEY"

    def test_telnyx_dynamic_config_generation(self):
        """Test dynamic config class creation for Telnyx"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("telnyx")
        config_class = create_config_class(provider)
        config = config_class()

        # Test API info resolution
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.telnyx.com/v2/ai"

        # Test with custom base
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://custom.api.com", "test-key"
        )
        assert api_base == "https://custom.api.com"
        assert api_key == "test-key"

    def test_telnyx_provider_resolution(self):
        """Test that provider resolution finds Telnyx"""
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        model, provider, api_key, api_base = get_llm_provider(
            model="telnyx/meta-llama/Meta-Llama-3.1-8B-Instruct",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert provider == "telnyx"
        assert api_base == "https://api.telnyx.com/v2/ai"

    def test_telnyx_supported_params(self):
        """Test that Telnyx config returns supported params"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("telnyx")
        config_class = create_config_class(provider)
        config = config_class()

        supported = config.get_supported_openai_params("meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_telnyx_provider_config_manager(self):
        """Test that ProviderConfigManager returns Telnyx configs"""
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct", provider=LlmProviders.TELNYX
        )

        assert config is not None
        assert config.custom_llm_provider == "telnyx"

    def test_telnyx_llm_providers_enum(self):
        """Test that TELNYX exists in LlmProviders enum"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "TELNYX")
        assert LlmProviders.TELNYX.value == "telnyx"


if __name__ == "__main__":
    print("Testing Telnyx Provider System...")

    test = TestTelnyxProviderLoader()

    print("\n1. Testing provider exists...")
    test.test_telnyx_provider_exists()
    print("   ✓ Telnyx provider registered")

    print("\n2. Testing provider config...")
    test.test_telnyx_provider_config()
    print("   ✓ Config values correct")

    print("\n3. Testing dynamic config generation...")
    test.test_telnyx_dynamic_config_generation()
    print("   ✓ Dynamic config works")

    print("\n4. Testing provider resolution...")
    test.test_telnyx_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n5. Testing supported params...")
    test.test_telnyx_supported_params()
    print("   ✓ Supported params work")

    print("\n6. Testing config manager...")
    test.test_telnyx_provider_config_manager()
    print("   ✓ Config manager works")

    print("\n7. Testing LlmProviders enum...")
    test.test_telnyx_llm_providers_enum()
    print("   ✓ LlmProviders enum works")

    print("\n" + "=" * 50)
    print("✓ All Telnyx provider tests passed!")
    print("=" * 50)
