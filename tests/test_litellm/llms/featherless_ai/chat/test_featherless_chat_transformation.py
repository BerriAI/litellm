"""
Unit tests for Featherless AI configuration.

These tests validate the FeatherlessAIConfig class which extends OpenAIGPTConfig.
Featherless AI is an OpenAI-compatible provider with a few customizations.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.featherless_ai.chat.transformation import FeatherlessAIConfig


class TestFeatherlessAIConfig:
    """Test class for FeatherlessAIConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = FeatherlessAIConfig()
        headers = {}
        api_key = "fake-featherless-key"

        result = config.validate_environment(
            headers=headers,
            model="featherless-ai/Qwerky-72B",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.featherless.ai/v1/",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = FeatherlessAIConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="featherless-ai/Qwerky-72B",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.featherless.ai/v1/",
            )

        assert "Missing Featherless AI API Key" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = FeatherlessAIConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

    def test_map_openai_params_with_tool_choice(self):
        """Test map_openai_params handles tool_choice parameter correctly"""
        config = FeatherlessAIConfig()

        # Test with auto value (supported)
        non_default_params = {"tool_choice": "auto"}
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="featherless-ai/Qwerky-72B",
            drop_params=False,
        )
        assert "tool_choice" in result
        assert result["tool_choice"] == "auto"

        # Test with none value (supported)
        non_default_params = {"tool_choice": "none"}
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="featherless-ai/Qwerky-72B",
            drop_params=False,
        )
        assert "tool_choice" in result
        assert result["tool_choice"] == "none"

        # Test with unsupported value and drop_params=True
        non_default_params = {
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="featherless-ai/Qwerky-72B",
            drop_params=True,
        )
        assert "tool_choice" not in result

        # Test with unsupported value and drop_params=False
        non_default_params = {
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }
        optional_params = {}
        with pytest.raises(Exception) as excinfo:
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="featherless-ai/Qwerky-72B",
                drop_params=False,
            )
        assert "Featherless AI doesn't support tool_choice=" in str(excinfo.value)

    def test_map_openai_params_with_tools(self):
        """Test map_openai_params handles tools parameter correctly"""
        config = FeatherlessAIConfig()

        # Test with tools and drop_params=True
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        non_default_params = {"tools": tools}
        optional_params = {}
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="featherless-ai/Qwerky-72B",
            drop_params=True,
        )
        assert "tools" not in result

        # Test with tools and drop_params=False
        with pytest.raises(Exception) as excinfo:
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="featherless-ai/Qwerky-72B",
                drop_params=False,
            )
        assert "Featherless AI doesn't support tools=" in str(excinfo.value)

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = FeatherlessAIConfig()
        headers = {}
        api_key = "fake-featherless-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="featherless-ai/Qwerky-72B",
            messages=[{"role": "user", "content": "Hello"}],
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

    def test_featherless_ai_completion_mock(self, respx_mock):
        """
        Mock test for Featherless AI completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import litellm

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )
        from litellm import completion

        # Set up environment variables for the test
        api_key = "fake-featherless-key"
        api_base = "https://api.featherless.ai/v1"
        model = "featherless_ai/featherless-ai/Qwerky-72B"
        model_name = "Qwerky-72B"  # The actual model name without provider prefix

        # Mock the HTTP request to the Featherless AI API
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
                            "content": '```python\nprint("Hi from LiteLLM!")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.',
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
                {"role": "user", "content": "write code for saying hi from LiteLLM"}
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
        assert "Hi from LiteLLM" in response.choices[0].message.content
