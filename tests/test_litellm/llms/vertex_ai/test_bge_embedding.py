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
        # BGE models return embeddings directly as arrays, not wrapped in objects
        mock_response.json.return_value = {
            "predictions": [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0]
            ],
            "deployedModelId": "849506872875548672",
            "model": "projects/1060139831167/locations/us-central1/models/baai_bge-small-en-v1.5",
            "modelDisplayName": "baai_bge-small-en-v1.5",
            "modelVersionId": "1"
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


def test_vertex_ai_bge_with_endpoint_id_pattern():
    """
    Test BGE with vertex_ai/bge/endpoint_id pattern.
    
    This test verifies that the pattern vertex_ai/bge/204379420394258432
    correctly triggers BGE transformations and routes to the endpoint.
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
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0]
            ],
            "deployedModelId": "204379420394258432",
            "model": "projects/1060139831167/locations/europe-west4/models/baai_bge-base-en",
            "modelDisplayName": "baai_bge-base-en",
            "modelVersionId": "1"
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="vertex_ai/bge/204379420394258432",
            input=["Hello", "World"],
            vertex_project="1060139831167",
            vertex_location="europe-west4",
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
        print("BGE Endpoint Pattern Test:")
        print("="*50)
        print(f"Model: vertex_ai/bge/204379420394258432")
        print(f"API URL: {api_url_called}")
        print("Request Body:")
        print(json.dumps(request_data, indent=2))
        print("="*50 + "\n")
        
        # Verify URL contains the endpoint ID and uses endpoints/ path
        assert "204379420394258432" in api_url_called, f"Endpoint ID not in URL: {api_url_called}"
        assert "endpoints" in api_url_called, f"Expected 'endpoints' in URL, got: {api_url_called}"
        
        # Verify BGE-specific request format (uses "prompt" not "content")
        assert "instances" in request_data
        assert "prompt" in request_data["instances"][0]
        assert request_data["instances"][0]["prompt"] == "Hello"
        
        # Verify response
        assert isinstance(response.data, list)
        assert len(response.data) == 2


def test_vertex_ai_bge_psc_endpoint_url_construction():
    """
    Test that BGE models with PSC endpoints construct correct URL without bge/ prefix.
    
    Verifies that vertex_ai/bge/378943383978115072 with api_base http://10.128.16.2
    constructs URL: http://10.128.16.2/v1/projects/{project}/locations/{location}/endpoints/378943383978115072:predict
    
    The bge/ prefix should be stripped from the endpoint URL.
    """
    client = HTTPHandler()
    
    def mock_auth_token(*args, **kwargs):
        return "test-token-123", "test-gcp-project-id-123"
    
    with patch.object(client, "post") as mock_post, patch(
        "litellm.llms.vertex_ai.vertex_embeddings.embedding_handler.VertexEmbedding._ensure_access_token",
        side_effect=mock_auth_token
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="vertex_ai/bge/378943383978115072",
            input=["The food was delicious and the waiter.."],
            api_base="http://10.128.16.2",
            vertex_project="test-gcp-project-id-123",
            vertex_location="us-central1",
            client=client,
            use_psc_endpoint_format=True  # Enable PSC endpoint format for this test
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
        
        print("\n" + "="*50)
        print("PSC Endpoint URL Construction Test:")
        print("="*50)
        print(f"Model: vertex_ai/bge/378943383978115072")
        print(f"API Base: http://10.128.16.2")
        print(f"Constructed URL: {api_url_called}")
        print("="*50 + "\n")
        
        # Verify the URL is constructed correctly
        expected_url = "http://10.128.16.2/v1/projects/test-gcp-project-id-123/locations/us-central1/endpoints/378943383978115072:predict"
        assert api_url_called == expected_url, f"Expected URL: {expected_url}, Got: {api_url_called}"
        
        # Verify bge/ prefix is NOT in the URL
        assert "bge/" not in api_url_called, f"URL should not contain 'bge/' prefix: {api_url_called}"
        
        # Verify response works
        assert isinstance(response.data, list)
        assert len(response.data) == 1


