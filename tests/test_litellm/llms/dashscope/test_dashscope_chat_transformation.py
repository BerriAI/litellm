"""
Unit tests for DashScope configuration.

These tests validate the DashScopeConfig class which extends OpenAIGPTConfig.
DashScope is an OpenAI-compatible provider with minor customizations.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig


class TestDashScopeConfig:
    """Test class for DashScope functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = DashScopeChatConfig()
        headers = {}
        api_key = "fake-dashscope-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="qwen-turbo",
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
    def test_dashscope_completion_mock(self, respx_mock):
        """
        Mock test for Dashscope completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-dashscope-key"
        api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        model = "dashscope/qwen-turbo"
        model_name = "qwen-turbo"  # The actual model name without provider prefix

        # Mock the HTTP request to the dashscope API
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

    def test_dashscope_transform_messages_preserves_multimodal_content(self):
        """
        Test that DashScopeChatConfig preserves multimodal content (image_url) in messages.
        This is critical for vision models like qwen3-vl-plus.
        """
        config = DashScopeChatConfig()
        
        # Test multimodal message with image_url
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image"
                    }
                ]
            }
        ]
        
        # Transform messages synchronously
        transformed = config._transform_messages(messages=messages, model="dashscope/qwen3-vl-plus", is_async=False)
        
        # Verify that image_url is preserved
        assert isinstance(transformed, list)
        assert len(transformed) == 1
        assert transformed[0]["role"] == "user"
        
        content = transformed[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        
        # Check image_url item
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/image.jpg"
        
        # Check text item
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Describe this image"
        
        # Test text-only message should also work
        text_messages = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        
        text_transformed = config._transform_messages(messages=text_messages, model="dashscope/qwen-turbo", is_async=False)
        assert len(text_transformed) == 1
        assert text_transformed[0]["content"] == "Hello, how are you?"
