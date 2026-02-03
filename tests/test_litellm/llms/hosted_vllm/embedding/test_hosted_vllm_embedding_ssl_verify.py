"""
Test SSL verification for hosted_vllm provider embeddings.

This test ensures that the ssl_verify parameter is properly passed through
to the HTTP client when using the hosted_vllm provider for embeddings.

Issue: ssl_verify parameter was being ignored because hosted_vllm fell through
to the openai_like catch-all path in main.py, which doesn't pass ssl_verify to the HTTP client.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm


class TestHostedVLLMEmbeddingSSLVerify:
    """Test suite for SSL verification in hosted_vllm provider embeddings."""

    @patch("litellm.llms.custom_httpx.llm_http_handler._get_httpx_client")
    def test_hosted_vllm_embedding_ssl_verify_false_sync(self, mock_get_httpx_client):
        """Test that ssl_verify=False is passed to the HTTP client for sync embedding calls."""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ],
            "model": "text-embedding-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5,
            },
        }
        mock_response.text = '{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}], "model": "text-embedding-model", "usage": {"prompt_tokens": 5, "total_tokens": 5}}'
        mock_client.post.return_value = mock_response
        mock_get_httpx_client.return_value = mock_client

        try:
            litellm.embedding(
                model="hosted_vllm/text-embedding-model",
                input=["hello world"],
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
    async def test_hosted_vllm_embedding_ssl_verify_false_async(
        self, mock_get_async_httpx_client
    ):
        """Test that ssl_verify=False is passed to the HTTP client for async embedding calls."""
        # Setup mock async client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ],
            "model": "text-embedding-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5,
            },
        }
        mock_response.text = '{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}], "model": "text-embedding-model", "usage": {"prompt_tokens": 5, "total_tokens": 5}}'

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mock_get_async_httpx_client.return_value = mock_client

        try:
            await litellm.aembedding(
                model="hosted_vllm/text-embedding-model",
                input=["hello world"],
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
