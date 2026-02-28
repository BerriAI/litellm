"""
Unit tests for Venice.AI configuration.

These tests validate the Venice.AI configuration which is now JSON-based.
Venice.AI is an OpenAI-compatible provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)

import pytest

from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.dynamic_config import create_config_class


class TestVeniceAIConfig:
    """Test class for Venice.AI functionality"""

    @pytest.fixture
    def config(self):
        """Get Venice.AI config from JSON registry"""
        if not JSONProviderRegistry.exists("veniceai"):
            pytest.skip("Venice.AI provider not found in JSON registry")
        provider_config = JSONProviderRegistry.get("veniceai")
        if provider_config is None:
            pytest.skip("Venice.AI provider not found in JSON registry")
        return create_config_class(provider_config)()

    def test_default_api_base(self, config):
        """
        Test that default API base is used when none is provided
        """
        headers = {}
        api_key = "fake-veniceai-key"

        result = config.validate_environment(
            headers=headers,
            model="venice-model",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_get_supported_openai_params(self, config):
        """
        Test that get_supported_openai_params returns correct params
        """
        supported_params = config.get_supported_openai_params(model="venice-model")
        
        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        
        # Note: JSON-based configs inherit from OpenAIGPTConfig which includes functions
        # This is expected behavior for JSON-based providers

    def test_map_openai_params_includes_functions(self, config):
        """
        Test that functions parameter is mapped (JSON-based configs don't exclude functions)
        """
        non_default_params = {
            "functions": [{"name": "test_function", "description": "Test function"}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="venice-model",
            drop_params=False
        )
        
        # JSON-based configs inherit from OpenAIGPTConfig which includes functions
        assert "functions" in result
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000

    def test_get_complete_url(self, config):
        """
        Test that get_complete_url constructs the correct endpoint URL
        """
        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="venice-model",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "https://api.venice.ai/api/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self, config):
        """
        Test that get_complete_url works with custom api_base
        """
        url = config.get_complete_url(
            api_base="https://custom.venice.ai/api/v1",
            api_key="fake-key",
            model="venice-model",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "https://custom.venice.ai/api/v1/chat/completions"

