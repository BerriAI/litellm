"""
Unit tests for Moonshot AI configuration.

These tests validate the MoonshotChatConfig class which extends OpenAIGPTConfig.
Moonshot AI is an OpenAI-compatible provider with minor customizations.
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
from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig


class TestMoonshotConfig:
    """Test class for Moonshot AI functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = MoonshotChatConfig()
        headers = {}
        api_key = "fake-moonshot-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="moonshot-v1-8k",
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
        config = MoonshotChatConfig()
        
        supported_params = config.get_supported_openai_params("moonshot-v1-8k")
        
        # Should include these params
        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        
        # Should NOT include functions (not supported by Moonshot AI)
        assert "functions" not in supported_params

    def test_map_openai_params_excludes_functions(self):
        """Test that functions parameter is not mapped"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "functions": [{"name": "test_function", "description": "Test function"}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # Functions should not be in result (not in supported params)
        assert "functions" not in result
        # Other supported params should be included
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000




    def test_map_openai_params_allows_other_tool_choice_values(self):
        """Test that other tool_choice values are allowed"""
        config = MoonshotChatConfig()
        
        for tool_choice_value in ["auto", "none", {"type": "function", "function": {"name": "test"}}]:
            non_default_params = {
                "tool_choice": tool_choice_value,
                "tools": [{"type": "function", "function": {"name": "test"}}]
            }
            
            result = config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model="moonshot-v1-8k",
                drop_params=False
            )
            
            # tool_choice should be included for non-"required" values
            assert result.get("tool_choice") == tool_choice_value


    def test_map_openai_params_max_completion_tokens_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "max_completion_tokens": 1000,
            "temperature": 0.7
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # max_completion_tokens should be mapped to max_tokens
        assert result.get("max_tokens") == 1000
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_temperature_handling_clamps_to_max_1(self):
        """Test that temperature > 1 is clamped to 1 (Moonshot limitation)"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "temperature": 1.5  # OpenAI allows up to 2, but Moonshot only allows up to 1
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # Temperature should be clamped to 1
        assert result.get("temperature") == 1

    def test_temperature_handling_low_temp_with_multiple_n(self):
        """Test that temperature < 0.3 with n > 1 is adjusted to 0.3"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "temperature": 0.1,  # Less than 0.3
            "n": 3  # Multiple completions
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # Temperature should be adjusted to 0.3 to avoid Moonshot API exceptions
        assert result.get("temperature") == 0.3
        assert result.get("n") == 3

    def test_temperature_handling_low_temp_single_n(self):
        """Test that temperature < 0.3 with n = 1 is preserved"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "temperature": 0.1,  # Less than 0.3
            "n": 1  # Single completion
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # Temperature should be preserved when n = 1
        assert result.get("temperature") == 0.1
        assert result.get("n") == 1

    def test_temperature_handling_valid_range(self):
        """Test that temperatures in valid range [0.3, 1] are preserved"""
        config = MoonshotChatConfig()
        
        test_temps = [0.3, 0.5, 0.7, 1.0]
        
        for temp in test_temps:
            non_default_params = {
                "temperature": temp,
                "n": 2
            }
            
            result = config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model="moonshot-v1-8k",
                drop_params=False
            )
            
            # Temperature should be preserved
            assert result.get("temperature") == temp

    def test_tool_choice_required_adds_message(self):
        """Test that tool_choice='required' adds a special message and removes tool_choice"""
        config = MoonshotChatConfig()
        
        messages = [
            {"role": "user", "content": "What's the weather like?"}
        ]
        
        optional_params = {
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        }
        
        result = config.transform_request(
            model="moonshot-v1-8k",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Check that the special message was added
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What's the weather like?"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Please select a tool to handle the current issue."
        
        # Check that tool_choice was removed but tools are preserved
        assert "tool_choice" not in result
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_tool_choice_required_preserves_other_params(self):
        """Test that tool_choice='required' handling preserves other parameters"""
        config = MoonshotChatConfig()
        
        messages = [
            {"role": "user", "content": "Calculate 2+2"}
        ]
        
        optional_params = {
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        result = config.transform_request(
            model="moonshot-v1-8k",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Check that other parameters are preserved
        assert result.get("temperature") == 0.7
        assert result.get("max_tokens") == 1000
        assert "tools" in result
        
        # Check that tool_choice was removed
        assert "tool_choice" not in result
        
        # Check that the message was added
        assert len(result["messages"]) == 2
        assert result["messages"][1]["content"] == "Please select a tool to handle the current issue."

    def test_tool_choice_non_required_preserved(self):
        """Test that non-'required' tool_choice values are preserved"""
        config = MoonshotChatConfig()
        
        messages = [
            {"role": "user", "content": "What's the weather?"}
        ]
        
        test_values = ["auto", "none", {"type": "function", "function": {"name": "get_weather"}}]
        
        for tool_choice_value in test_values:
            optional_params = {
                "tool_choice": tool_choice_value,
                "tools": [{"type": "function", "function": {"name": "get_weather"}}]
            }
            
            result = config.transform_request(
                model="moonshot-v1-8k",
                messages=messages,
                optional_params=optional_params,
                litellm_params={},
                headers={}
            )
            
            # Check that tool_choice is preserved for non-"required" values
            assert result.get("tool_choice") == tool_choice_value
            
            # Check that no extra message was added
            assert len(result["messages"]) == 1
            assert result["messages"][0]["content"] == "What's the weather?"