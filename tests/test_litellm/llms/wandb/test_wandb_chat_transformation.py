"""
Unit tests for WandB Inference configuration.

These tests validate the WandbInferenceConfig class which extends OpenAIGPTConfig.
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
from litellm.llms.wandb.chat.transformation import WandbConfig


class TestWandbConfig:
    """Test class for WandB Inference functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = WandbConfig()
        headers = {}
        api_key = "fake-wandb-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="wandb/openai/gpt-oss-20b",
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
    def test_wandb_completion_mock(self, respx_mock):
        """
        Mock test for WandB Inference completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-wandb-key"
        api_base = "https://api.inference.wandb.ai/v1"
        model = "wandb/openai/gpt-oss-20b"
        model_name = "gpt-oss-20b"  # The actual model name without provider prefix

        # Mock the HTTP request to the WandB Inference API
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

        # If response is a streaming wrapper, extract the first chunk for assertions
        # This handles both streaming and non-streaming responses
        # For streaming, response is typically an iterator yielding (event, data) tuples
        if hasattr(response, "__iter__") and not hasattr(response, "choices"):
            # Streaming response: get the first chunk
            first_chunk = next(iter(response))
            # first_chunk is likely a tuple: (event, data)
            # Try to extract the data part
            if isinstance(first_chunk, tuple) and len(first_chunk) == 2:
                data = first_chunk[1]
            else:
                data = first_chunk

            # The data object should have .choices[0] with .delta or .message
            choices = getattr(data, "choices", None)
            assert choices is not None
            assert len(choices) > 0
            choice = choices[0]
            # For streaming, content may be in .delta or .message
            content = None
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
            elif hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
            assert content is not None
            assert "```python" in content
            assert "Hey from LiteLLM" in content
        else:
            # Non-streaming response
            choices = getattr(response, "choices", None)
            assert choices is not None
            assert len(choices) > 0
            choice = choices[0]
            message = getattr(choice, "message", None)
            assert message is not None
            content = getattr(message, "content", None)
            assert content is not None

            # Check for specific content in the response
            assert "```python" in content
            assert "Hey from LiteLLM" in content
