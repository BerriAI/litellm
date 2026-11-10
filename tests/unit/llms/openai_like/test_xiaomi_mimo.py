"""
Tests for Xiaomi MiMo provider configuration and integration.
Related to issue #18794
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


class TestXiaomiMiMoProviderConfig:
    """Test Xiaomi MiMo provider configuration"""

    def test_xiaomi_mimo_in_provider_list(self):
        """Test that xiaomi_mimo is in the provider list (fixes #18794)"""
        from litellm import LlmProviders

        # Verify xiaomi_mimo is in the enum
        assert hasattr(LlmProviders, "XIAOMI_MIMO")
        assert LlmProviders.XIAOMI_MIMO.value == "xiaomi_mimo"

        # Verify it's in the provider list
        assert "xiaomi_mimo" in litellm.provider_list

    def test_xiaomi_mimo_json_config_exists(self):
        """Test that xiaomi_mimo is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify xiaomi_mimo is loaded
        assert JSONProviderRegistry.exists("xiaomi_mimo")

        # Get xiaomi_mimo config
        xiaomi_mimo = JSONProviderRegistry.get("xiaomi_mimo")
        assert xiaomi_mimo is not None
        assert xiaomi_mimo.base_url == "https://api.xiaomimimo.com/v1"
        assert xiaomi_mimo.api_key_env == "XIAOMI_MIMO_API_KEY"
        assert xiaomi_mimo.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_xiaomi_mimo_provider_resolution(self):
        """Test that provider resolution finds xiaomi_mimo"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="xiaomi_mimo/mimo-v2-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "mimo-v2-flash"
        assert provider == "xiaomi_mimo"
        assert api_base == "https://api.xiaomimimo.com/v1"

    def test_xiaomi_mimo_router_config(self):
        """Test that xiaomi_mimo can be used in Router configuration (fixes #18794)"""
        from litellm import Router

        # This should not raise "Unsupported provider - xiaomi_mimo"
        router = Router(
            model_list=[
                {
                    "model_name": "mimo-v2-flash",
                    "litellm_params": {
                        "model": "xiaomi_mimo/mimo-v2-flash",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        # Verify the deployment was created successfully
        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "mimo-v2-flash"
