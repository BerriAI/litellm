"""
Unit tests for Morph configuration.

These tests validate the MorphChatConfig class which extends OpenAILikeChatConfig.
Morph is an OpenAI-compatible provider with a few customizations.
"""

import os
import sys
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.morph.chat.transformation import MorphChatConfig


class TestMorphChatConfig:
    """Test class for MorphChatConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = MorphChatConfig()
        headers = {}
        api_key = "fake-morph-key"

        result = config.validate_environment(
            headers=headers,
            model="morph/apply-v1",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.morphllm.com/v1",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_get_openai_compatible_provider_info(self):
        """Test the _get_openai_compatible_provider_info method"""
        config = MorphChatConfig()
        api_key = "fake-morph-key"

        result = config._get_openai_compatible_provider_info(
            api_base=None,
            api_key=api_key,
        )

        # Verify correct API base is returned
        assert result[0] == "https://api.morphllm.com/v1"
        assert result[1] == api_key

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = MorphChatConfig()
        
        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="morph/apply-v1",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.morphllm.com/v1",
            )

        assert "Morph API key is required" in str(excinfo.value)

    def test_inheritance(self):
        """Test proper inheritance from OpenAILikeChatConfig"""
        config = MorphChatConfig()

        from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

        assert isinstance(config, OpenAILikeChatConfig)
        assert hasattr(config, "_get_openai_compatible_provider_info")

    def test_morph_completion_mock(self, respx_mock):
        """
        Mock test for Morph completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import respx
        from litellm import completion
        
        # Set up environment variables for the test
        api_key = "fake-morph-key"
        api_base = "https://api.morphllm.com/v1"
        model = "morph/apply-v1"
        
        # Mock the HTTP request to the Morph API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "apply-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```python\nprint(\"Hi from LiteLLM!\")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            },
            status_code=200
        )
        
        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
            api_key=api_key,
            api_base=api_base
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
        
    def test_morph_apply_code_updates(self, respx_mock):
        """
        Test Morph's Apply Code Updates functionality which uses special tags
        for code and updates as per https://docs.morphllm.com/api-reference/endpoint/apply
        """
        import respx
        from litellm import completion
        
        # Set up environment variables for the test
        api_key = "fake-morph-key"
        api_base = "https://api.morphllm.com/v1"
        model = "morph/apply-v1"
        
        # Original code and update with Morph's special tags
        original_code = """def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total"""
        
        update_snippet = """def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total * 1.1  # Add 10% tax"""
        
        user_message = f"<code>{original_code}</code>\n<update>{update_snippet}</update>"
        
        # Expected response after applying the update
        expected_updated_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total * 1.1  # Add 10% tax
"""
        
        # Mock the HTTP request to the Morph API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "apply-v1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": expected_updated_code,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 25, "completion_tokens": 32, "total_tokens": 57},
            },
            status_code=200
        )
        
        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            api_key=api_key,
            api_base=api_base
        )
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        
        # Check that the response contains the expected updated code
        assert response.choices[0].message.content.strip() == expected_updated_code.strip() 