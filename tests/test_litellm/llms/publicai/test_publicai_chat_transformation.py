"""
Unit tests for Public AI configuration.

These tests validate the PublicAIChatConfig class which extends OpenAIGPTConfig.
Public AI is an OpenAI-compatible provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
import litellm.utils
from litellm import completion
from litellm.llms.publicai.chat.transformation import PublicAIChatConfig


class TestPublicAIConfig:
    """Test class for Public AI functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = PublicAIChatConfig()
        headers = {}
        api_key = "fake-publicai-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # We can't directly test the api_base value here since validate_environment
        # only returns the headers, but we can verify it doesn't raise an exception
        # which would happen if api_base handling was incorrect

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct params"""
        config = PublicAIChatConfig()
        
        supported_params = config.get_supported_openai_params("swiss-ai/Apertus-17B-v1-Instruct-Q6_K")
        
        # Should include standard OpenAI params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        assert "top_p" in supported_params

    def test_map_openai_params_basic(self):
        """Test that basic parameters are mapped correctly"""
        config = PublicAIChatConfig()
        
        non_default_params = {
            "temperature": 0.8,
            "max_tokens": 8192,
            "top_p": 0.9
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            drop_params=False
        )
        
        # All supported params should be included
        assert result.get("temperature") == 0.8
        assert result.get("max_tokens") == 8192
        assert result.get("top_p") == 0.9

    def test_map_openai_params_max_completion_tokens_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        config = PublicAIChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 1000,
            "temperature": 0.7
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            drop_params=False
        )
        
        # max_completion_tokens should be mapped to max_tokens
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_get_openai_compatible_provider_info(self):
        """Test that provider info returns correct defaults"""
        config = PublicAIChatConfig()
        
        # Test with no api_base or api_key
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        
        # Should return default api_base
        assert api_base == "https://platform.publicai.co/v1"

    def test_get_complete_url(self):
        """Test that complete URL is constructed correctly"""
        config = PublicAIChatConfig()
        
        # Test with no api_base
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            optional_params={},
            litellm_params={}
        )
        
        assert url == "https://platform.publicai.co/v1/chat/completions"

    def test_get_complete_url_custom_base(self):
        """Test that custom api_base is handled correctly"""
        config = PublicAIChatConfig()
        
        # Test with custom api_base without /chat/completions
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1",
            api_key="test-key",
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            optional_params={},
            litellm_params={}
        )
        
        assert url == "https://custom.api.com/v1/chat/completions"
        
        # Test with custom api_base with /chat/completions
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1/chat/completions",
            api_key="test-key",
            model="swiss-ai/Apertus-17B-v1-Instruct-Q6_K",
            optional_params={},
            litellm_params={}
        )
        
        assert url == "https://custom.api.com/v1/chat/completions"
