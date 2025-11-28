"""
Unit tests for PublicAI configuration.

These tests validate the PublicAIChatConfig class which extends OpenAIGPTConfig.
PublicAI is an OpenAI-compatible provider with minor customizations.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)

import pytest

import litellm
import litellm.utils
from litellm import completion
from litellm.llms.publicai.chat.transformation import PublicAIChatConfig


class TestPublicAIConfig:
    """Test class for PublicAI functionality"""

    def test_default_api_base(self):
        """
        Test that default API base is used when none is provided
        """
        config = PublicAIChatConfig()
        headers = {}
        api_key = "fake-publicai-key"

        result = config.validate_environment(
            headers=headers,
            model="swiss-ai-apertus",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_get_supported_openai_params(self):
        """
        Test that get_supported_openai_params returns correct params
        """
        config = PublicAIChatConfig()
        
        supported_params = config.get_supported_openai_params(model="swiss-ai-apertus")
        
        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        
        assert "functions" not in supported_params

    def test_map_openai_params_excludes_functions(self):
        """
        Test that functions parameter is not mapped
        """
        config = PublicAIChatConfig()
        
        non_default_params = {
            "functions": [{"name": "test_function", "description": "Test function"}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai-apertus",
            drop_params=False
        )
        
        assert "functions" not in result
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000

    def test_map_openai_params_max_completion_tokens_mapping(self):
        """
        Test that max_completion_tokens is mapped to max_tokens
        """
        config = PublicAIChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 1000,
            "temperature": 0.7
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai-apertus",
            drop_params=False
        )
        
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_get_complete_url(self):
        """
        Test that get_complete_url constructs the correct endpoint URL
        """
        config = PublicAIChatConfig()
        
        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="swiss-ai-apertus",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "https://platform.publicai.co/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self):
        """
        Test that get_complete_url works with custom api_base
        """
        config = PublicAIChatConfig()
        
        url = config.get_complete_url(
            api_base="https://custom.publicai.co/v1",
            api_key="fake-key",
            model="swiss-ai-apertus",
            optional_params={},
            litellm_params={},
            stream=False
        )
        
        assert url == "https://custom.publicai.co/v1/chat/completions"

