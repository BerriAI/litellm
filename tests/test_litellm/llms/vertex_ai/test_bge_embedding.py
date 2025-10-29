"""
Test BGE embeddings with Vertex AI using custom api_base.

This test ensures that BGE embeddings work correctly with Vertex AI
and that the request body is properly formatted.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../../..")
)

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def test_vertex_ai_bge_embedding_with_custom_api_base():
    """
    Test Vertex AI BGE embeddings with custom api_base.
    
    This test verifies that when using a BGE model with Vertex AI and
    a custom api_base, the request is properly formatted and sent to
    the correct endpoint.
    """
    client = HTTPHandler()
    
    def mock_auth_token(*args, **kwargs):
        return "fake-token", "fake-project"
    
    with patch.object(client, "post") as mock_post, patch(
        "litellm.llms.vertex_ai.vertex_embeddings.embedding_handler.VertexEmbedding._ensure_access_token",
        side_effect=mock_auth_token
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {
                    "embeddings": {
                        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "statistics": {"token_count": 2}
                    }
                },
                {
                    "embeddings": {
                        "values": [0.6, 0.7, 0.8, 0.9, 1.0],
                        "statistics": {"token_count": 2}
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="vertex_ai/bge-small-en-v1.5",
            input=["Hello", "World"],
            api_base="http://10.96.32.8",
            client=client
        )
        
        mock_post.assert_called_once()
        
        call_args = mock_post.call_args
        kwargs = call_args.kwargs if hasattr(call_args, 'kwargs') else call_args[1]
        
        if "url" in kwargs:
            api_url_called = kwargs["url"]
        elif len(call_args[0]) > 0:
            api_url_called = call_args[0][0]
        else:
            api_url_called = "Unknown"
        
        # Vertex AI may use 'json' or 'data' parameter
        if "json" in kwargs:
            request_data = kwargs["json"]
        elif "data" in kwargs:
            request_data = json.loads(kwargs["data"])
        else:
            request_data = {}
        
        print("\n" + "="*50)
        print("Mock Request Body Received:")
        print("="*50)
        print(json.dumps(request_data, indent=2))
        print("="*50)
        print(f"API Base: {api_url_called}")
        print("="*50 + "\n")
        
        assert "instances" in request_data
        assert len(request_data["instances"]) == 2
        # BGE models should use "prompt" instead of "content"
        assert "prompt" in request_data["instances"][0]
        assert request_data["instances"][0]["prompt"] == "Hello"
        assert "prompt" in request_data["instances"][1]
        assert request_data["instances"][1]["prompt"] == "World"
        
        assert isinstance(response.data, list)
        assert len(response.data) == 2
        assert "embedding" in response.data[0]

