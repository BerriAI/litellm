"""
Test Gemini batch embeddings with custom api_base and extra_headers.

This test ensures that:
1. Authentication headers are properly included when using custom api_base
2. The extra_headers parameter is correctly passed through
3. Both dict-based auth_header (Gemini) and Bearer token (Vertex AI) are handled
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest
import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_gemini_batch_embeddings_with_custom_api_base_and_auth_header():
    """
    Test that Gemini batch embeddings include auth_header when using custom api_base.
    
    This test verifies that when using Gemini embeddings with a custom api_base
    (e.g., Cloudflare AI Gateway), the x-goog-api-key header is properly included
    in the HTTP request.
    """
    client = HTTPHandler()
    
    def mock_auth_token(*args, **kwargs):
        return None, "test-project"
    
    with patch.object(client, "post") as mock_post, patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._ensure_access_token",
        side_effect=mock_auth_token
    ), patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._get_token_and_url"
    ) as mock_get_token:
        # Mock the _get_token_and_url to return auth_header dict and URL
        mock_get_token.return_value = (
            {"x-goog-api-key": "test-gemini-api-key"},
            "https://gateway.ai.cloudflare.com/v1/test/noauth/google-ai-studio/v1beta"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3, 0.4, 0.5]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="gemini/text-embedding-004",
            input=["Hello, world!"],
            api_key="test-gemini-api-key",
            api_base="https://gateway.ai.cloudflare.com/v1/test/noauth/google-ai-studio/v1beta",
            client=client
        )
        
        # Verify the POST was called
        mock_post.assert_called_once()
        
        # Get the headers that were passed to the POST request
        call_args = mock_post.call_args
        kwargs = call_args.kwargs if hasattr(call_args, 'kwargs') else call_args[1]
        headers = kwargs.get("headers", {})
        
        # Verify auth_header is included
        assert "x-goog-api-key" in headers, f"x-goog-api-key not in headers: {headers}"
        assert headers["x-goog-api-key"] == "test-gemini-api-key"
        
        # Verify Content-Type is still present
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json; charset=utf-8"


def test_gemini_batch_embeddings_with_extra_headers():
    """
    Test that extra_headers parameter is properly included in the request.
    
    This test verifies that custom headers passed via extra_headers are
    properly merged into the request headers.
    """
    client = HTTPHandler()
    
    def mock_auth_token(*args, **kwargs):
        return None, "test-project"
    
    with patch.object(client, "post") as mock_post, patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._ensure_access_token",
        side_effect=mock_auth_token
    ), patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._get_token_and_url"
    ) as mock_get_token:
        # Mock the _get_token_and_url to return auth_header dict and URL
        mock_get_token.return_value = (
            {"x-goog-api-key": "test-gemini-api-key"},
            "https://gateway.ai.cloudflare.com/v1/test/google-ai-studio/v1beta"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="gemini/text-embedding-004",
            input=["Test"],
            api_key="test-gemini-api-key",
            api_base="https://gateway.ai.cloudflare.com/v1/test/google-ai-studio/v1beta",
            headers={"Authorization": "Bearer test-token", "X-Custom": "custom-value"},
            client=client
        )
        
        # Verify the POST was called
        mock_post.assert_called_once()
        
        # Get the headers that were passed to the POST request
        call_args = mock_post.call_args
        kwargs = call_args.kwargs if hasattr(call_args, 'kwargs') else call_args[1]
        headers = kwargs.get("headers", {})
        
        # Verify all headers are included
        assert "x-goog-api-key" in headers
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"
        assert "X-Custom" in headers
        assert headers["X-Custom"] == "custom-value"

