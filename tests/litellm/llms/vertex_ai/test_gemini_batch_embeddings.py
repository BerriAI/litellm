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
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation import (
    _is_multimodal_input,
    _parse_data_url,
    process_embed_content_response,
    transform_openai_input_gemini_embed_content,
)
from litellm.types.utils import EmbeddingResponse


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
            "embeddings": [
                {
                    "values": [0.1, 0.2, 0.3, 0.4, 0.5]
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
            "embeddings": [
                {
                    "values": [0.1, 0.2, 0.3]
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


def test_is_multimodal_input_detection():
    """Test that _is_multimodal_input correctly detects multimodal inputs."""
    assert _is_multimodal_input("plain text") is False
    assert _is_multimodal_input(["text1", "text2"]) is False
    
    assert _is_multimodal_input("data:image/png;base64,iVBORw0KGgo=") is True
    assert _is_multimodal_input(["text", "data:image/png;base64,abc"]) is True
    
    assert _is_multimodal_input("files/abc123") is True
    assert _is_multimodal_input(["text", "files/myfile"]) is True


def test_parse_data_url():
    """Test that _parse_data_url correctly extracts MIME type and base64 data."""
    mime_type, base64_data = _parse_data_url("data:image/png;base64,iVBORw0KGgo=")
    assert mime_type == "image/png"
    assert base64_data == "iVBORw0KGgo="
    
    mime_type, base64_data = _parse_data_url("data:audio/mpeg;base64,SUQzBAA=")
    assert mime_type == "audio/mpeg"
    assert base64_data == "SUQzBAA="
    
    mime_type, base64_data = _parse_data_url("data:video/mp4;base64,AAAAIGZ0eXA=")
    assert mime_type == "video/mp4"
    assert base64_data == "AAAAIGZ0eXA="
    
    mime_type, base64_data = _parse_data_url("data:application/pdf;base64,JVBERi0=")
    assert mime_type == "application/pdf"
    assert base64_data == "JVBERi0="


def test_mime_type_validation():
    """Test that unsupported MIME types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported MIME type"):
        _parse_data_url("data:text/plain;base64,SGVsbG8=")
    
    with pytest.raises(ValueError, match="Unsupported MIME type"):
        _parse_data_url("data:application/json;base64,e30=")


def test_parse_data_url_invalid_format():
    """Test that invalid data URL formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid data URL format"):
        _parse_data_url("not-a-data-url")
    
    with pytest.raises(ValueError, match="missing comma"):
        _parse_data_url("data:image/png;base64")


def test_transform_multimodal_text_and_image():
    """Test transformation of mixed text and image input."""
    input_data = [
        "The food was delicious",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    ]
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={},
        resolved_files=None,
    )
    
    assert "content" in result
    assert "parts" in result["content"]
    parts = result["content"]["parts"]
    
    assert len(parts) == 2
    assert parts[0]["text"] == "The food was delicious"
    assert "inline_data" in parts[1]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert "data" in parts[1]["inline_data"]


def test_transform_multimodal_with_file_reference():
    """Test transformation with Gemini file reference."""
    input_data = ["Some text", "files/abc123"]
    
    resolved_files = {
        "files/abc123": {
            "mime_type": "image/jpeg",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123"
        }
    }
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={},
        resolved_files=resolved_files,
    )
    
    assert "content" in result
    parts = result["content"]["parts"]
    
    assert len(parts) == 2
    assert parts[0]["text"] == "Some text"
    assert "file_data" in parts[1]
    assert parts[1]["file_data"]["mime_type"] == "image/jpeg"
    assert parts[1]["file_data"]["file_uri"] == "https://generativelanguage.googleapis.com/v1beta/files/abc123"


def test_embed_content_response_processing():
    """Test processing of embedContent response (single embedding)."""
    response_json = {
        "embedding": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }
    
    model_response = EmbeddingResponse()
    result = process_embed_content_response(
        input=["test input"],
        model_response=model_response,
        model="gemini-embedding-2-preview",
        response_json=response_json,
    )
    
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert result.model == "gemini-embedding-2-preview"


def test_gemini_multimodal_embedding_e2e():
    """Test end-to-end multimodal embedding call through litellm.embedding()."""
    client = HTTPHandler()
    
    def mock_auth_token(*args, **kwargs):
        return None, "test-project"
    
    with patch.object(client, "post") as mock_post, patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._ensure_access_token",
        side_effect=mock_auth_token
    ), patch(
        "litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler.GoogleBatchEmbeddings._get_token_and_url"
    ) as mock_get_token:
        mock_get_token.return_value = (
            {"x-goog-api-key": "test-key"},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent?key=test-key"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        mock_post.return_value = mock_response
        
        response = litellm.embedding(
            model="gemini/gemini-embedding-2-preview",
            input=["The food was delicious", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="],
            api_key="test-key",
            client=client
        )
        
        mock_post.assert_called_once()
        
        call_args = mock_post.call_args
        kwargs = call_args.kwargs if hasattr(call_args, 'kwargs') else call_args[1]
        
        request_body = json.loads(kwargs.get("data", "{}"))
        
        assert "content" in request_body
        assert "parts" in request_body["content"]
        parts = request_body["content"]["parts"]
        
        assert len(parts) == 2
        assert parts[0]["text"] == "The food was delicious"
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_gemini_multimodal_embedding_with_audio():
    """Test multimodal embedding with audio input."""
    input_data = ["Audio description", "data:audio/mpeg;base64,SUQzBAAAAAA="]
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={},
        resolved_files=None,
    )
    
    parts = result["content"]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "Audio description"
    assert parts[1]["inline_data"]["mime_type"] == "audio/mpeg"


def test_gemini_multimodal_embedding_with_video():
    """Test multimodal embedding with video input."""
    input_data = ["data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA"]
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={},
        resolved_files=None,
    )
    
    parts = result["content"]["parts"]
    assert len(parts) == 1
    assert parts[0]["inline_data"]["mime_type"] == "video/mp4"



def test_transform_with_optional_params():
    """Test that optional params like outputDimensionality are passed through."""
    input_data = ["test text"]
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={"outputDimensionality": 768, "taskType": "SEMANTIC_SIMILARITY"},
        resolved_files=None,
    )
    
    assert result["outputDimensionality"] == 768
    assert result["taskType"] == "SEMANTIC_SIMILARITY"


def test_dimensions_mapped_to_output_dimensionality():
    """Test that OpenAI 'dimensions' param is mapped to Gemini 'outputDimensionality'."""
    input_data = ["test text"]
    
    result = transform_openai_input_gemini_embed_content(
        input=input_data,
        model="gemini-embedding-2-preview",
        optional_params={"dimensions": 768},
        resolved_files=None,
    )
    
    assert "outputDimensionality" in result
    assert result["outputDimensionality"] == 768
    assert "dimensions" not in result

