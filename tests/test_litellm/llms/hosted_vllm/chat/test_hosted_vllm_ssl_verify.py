"""
Test SSL verification for hosted_vllm provider.

This test ensures that the ssl_verify parameter is properly passed through
to the HTTP client when using the hosted_vllm provider.

Issue: ssl_verify parameter was being ignored because hosted_vllm fell through
to the OpenAI catch-all path in main.py, which doesn't pass ssl_verify to the HTTP client.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm


class TestHostedVLLMSSLVerify:
    """Test suite for SSL verification in hosted_vllm provider."""

    @patch("litellm.llms.custom_httpx.llm_http_handler._get_httpx_client")
    def test_hosted_vllm_ssl_verify_false_sync(self, mock_get_httpx_client):
        """Test that ssl_verify=False is passed to the HTTP client for sync calls."""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response.text = '{"id": "chatcmpl-test", "object": "chat.completion", "created": 1234567890, "model": "test-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'
        mock_client.post.return_value = mock_response
        mock_get_httpx_client.return_value = mock_client

        try:
            litellm.completion(
                model="hosted_vllm/test-model",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://test-vllm.example.com/v1",
                ssl_verify=False,
            )
        except Exception:
            # Even if the response parsing fails, we just need to verify
            # that the mock was called with the correct ssl_verify parameter
            pass

        # Verify _get_httpx_client was called with ssl_verify=False
        mock_get_httpx_client.assert_called()
        call_args = mock_get_httpx_client.call_args

        # Check that params contains ssl_verify=False
        if call_args[0]:
            # Positional argument
            params = call_args[0][0]
        else:
            # Keyword argument
            params = call_args[1].get("params", {})

        assert (
            params.get("ssl_verify") is False
        ), f"Expected ssl_verify=False in params, got {params}"

    @patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_hosted_vllm_ssl_verify_false_async(
        self, mock_get_async_httpx_client
    ):
        """Test that ssl_verify=False is passed to the HTTP client for async calls."""
        # Setup mock async client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_response.text = '{"id": "chatcmpl-test", "object": "chat.completion", "created": 1234567890, "model": "test-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mock_get_async_httpx_client.return_value = mock_client

        try:
            await litellm.acompletion(
                model="hosted_vllm/test-model",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://test-vllm.example.com/v1",
                ssl_verify=False,
            )
        except Exception:
            # Even if the response parsing fails, we just need to verify
            # that the mock was called with the correct ssl_verify parameter
            pass

        # Verify get_async_httpx_client was called with ssl_verify=False
        mock_get_async_httpx_client.assert_called()
        call_kwargs = mock_get_async_httpx_client.call_args[1]

        # Check that params contains ssl_verify=False
        params = call_kwargs.get("params", {})
        assert (
            params.get("ssl_verify") is False
        ), f"Expected ssl_verify=False in params, got {params}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
