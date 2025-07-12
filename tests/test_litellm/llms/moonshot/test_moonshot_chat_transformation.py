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

    def test_map_openai_params_handles_tool_choice_required_with_drop_params(self):
        """Test that tool_choice 'required' is dropped when drop_params=True"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "temperature": 0.7
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=True
        )
        
        # tool_choice should be dropped when it's "required" and drop_params=True
        assert "tool_choice" not in result
        # Tools should remain
        assert result.get("tools") is not None
        assert result.get("temperature") == 0.7

    def test_map_openai_params_raises_error_for_tool_choice_required(self):
        """Test that tool_choice 'required' raises error when drop_params=False"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "test"}}]
        }
        
        with pytest.raises(litellm.utils.UnsupportedParamsError) as exc_info:
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model="moonshot-v1-8k",
                drop_params=False
            )
        
        assert "tool_choice='required'" in str(exc_info.value)

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

    def test_map_openai_params_temperature_n_limitation(self):
        """Test that n is set to 1 when temperature is low"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "temperature": 0.2,  # Low temperature
            "n": 3  # Multiple results requested
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # n should be set to 1 when temperature is low
        assert result.get("n") == 1
        assert result.get("temperature") == 0.2

    def test_map_openai_params_temperature_n_not_limited_high_temp(self):
        """Test that n is not limited when temperature is high"""
        config = MoonshotChatConfig()
        
        non_default_params = {
            "temperature": 0.8,  # High temperature
            "n": 3  # Multiple results requested
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="moonshot-v1-8k",
            drop_params=False
        )
        
        # n should remain as 3 when temperature is high
        assert result.get("n") == 3
        assert result.get("temperature") == 0.8

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

    @pytest.mark.respx()
    def test_moonshot_completion_mock(self, respx_mock):
        """
        Mock test for Moonshot completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-moonshot-key"
        api_base = "https://api.moonshot.ai/v1"
        model = "moonshot/moonshot-v1-8k"
        model_name = "moonshot-v1-8k"  # The actual model name without provider prefix

        # Mock the HTTP request to the moonshot API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\nprint("Hey from LiteLLM!")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "```python" in response.choices[0].message.content
        assert "Hey from LiteLLM" in response.choices[0].message.content