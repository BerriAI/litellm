"""
Unit tests for Parasail AI configuration.

These tests validate the ParasailChatConfig class which extends OpenAILikeChatConfig.
Parasail AI is an OpenAI-compatible provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.llms.parasail.chat.transformation import ParasailChatConfig


class TestParasailConfig:
    """Test class for Parasail AI functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = ParasailChatConfig()
        
        # Test with provided values
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="https://custom.parasail.io/v1",
            api_key="test-key"
        )
        assert api_base == "https://custom.parasail.io/v1"
        assert api_key == "test-key"
        
        # Test with default values
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None,
            api_key=None
        )
        assert api_base == "https://api.parasail.io/v1"
        assert api_key is None

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct params"""
        config = ParasailChatConfig()
        
        supported_params = config.get_supported_openai_params("parasail-model")
        
        # Should include these standard OpenAI params
        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        assert "frequency_penalty" in supported_params
        assert "presence_penalty" in supported_params
        assert "top_p" in supported_params
        assert "stop" in supported_params
        assert "n" in supported_params
        assert "functions" in supported_params
        assert "function_call" in supported_params
        assert "response_format" in supported_params
        assert "logit_bias" in supported_params

    def test_map_openai_params_basic(self):
        """Test that basic parameters are mapped correctly"""
        config = ParasailChatConfig()
        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True,
            "top_p": 0.9
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="parasail-model",
            drop_params=False
        )
        
        # Standard parameters should be included
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000
        assert result.get("stream") is True
        assert result.get("top_p") == 0.9

    def test_map_openai_params_max_completion_tokens_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        config = ParasailChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 1500,
            "temperature": 0.5
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="parasail-model",
            drop_params=False
        )
        
        # max_completion_tokens should be mapped to max_tokens
        assert result.get("max_tokens") == 1500
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.5

    def test_config_initialization(self):
        """Test that ParasailChatConfig can be initialized without errors"""
        config = ParasailChatConfig()
        assert config is not None
        assert isinstance(config, ParasailChatConfig)

    def test_inheritance_from_openai_like(self):
        """Test that ParasailChatConfig properly inherits from OpenAILikeChatConfig"""
        config = ParasailChatConfig()
        
        # Test that it has the expected methods from parent class
        assert hasattr(config, '_get_openai_compatible_provider_info')
        assert hasattr(config, 'get_supported_openai_params')
        assert hasattr(config, 'map_openai_params')
        
        # Test that methods are callable
        assert callable(config._get_openai_compatible_provider_info)
        assert callable(config.get_supported_openai_params)
        assert callable(config.map_openai_params)
