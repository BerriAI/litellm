"""
Unit tests for Nebius AI Studio configuration.

These tests validate the NebiusConfig class which extends OpenAIGPTConfig.
Nebius AI Studio is an OpenAI-compatible provider with minor customizations.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion
from litellm.llms.nebius.chat.transformation import NebiusConfig


class TestNebiusConfig:
    """Test class for Nebius AI Studio functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = NebiusConfig()
        headers = {}
        api_key = "fake-nebius-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="nebius/Qwen/Qwen3-4B",
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

    @pytest.mark.respx()
    def test_nebius_completion_mock(self, respx_mock):
        """
        Mock test for Nebius AI Studio completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.use_aiohttp_transport = False  # since this uses respx, we need to set use_aiohttp_transport to False

        # Set up environment variables for the test
        api_key = "fake-nebius-key"
        api_base = "https://api.studio.nebius.ai/v1"
        model = "nebius/Qwen/Qwen3-4B"
        model_name = "Qwen3-4B"  # The actual model name without provider prefix

        # Mock the HTTP request to the Nebius AI Studio API
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
