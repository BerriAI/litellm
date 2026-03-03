import pytest
from unittest.mock import patch, MagicMock
from litellm import embedding

def test_vertex_ai_embedding_extra_headers():
    """Test that extra_headers are forwarded to vertex_ai embedding provider via mocks."""
    with patch("litellm.main.vertex_embedding.embedding") as mock_vertex_embedding:
        mock_vertex_embedding.return_value = MagicMock()
        try:
            embedding(
                model="vertex_ai/text-embedding-004",
                input=["hello"],
                extra_headers={"X-Custom-Header": "test-value"},
            )
        except Exception:
            pass
        mock_vertex_embedding.assert_called_once()
        call_kwargs = mock_vertex_embedding.call_args.kwargs
        assert call_kwargs.get("extra_headers") == {"X-Custom-Header": "test-value"}

def test_vertex_multimodal_embedding_headers():
    """Test that headers are forwarded to vertex_ai multimodal embedding provider."""
    with patch("litellm.main.vertex_multimodal_embedding.multimodal_embedding") as mock_vertex_multi:
        mock_vertex_multi.return_value = MagicMock()
        try:
            embedding(
                model="vertex_ai/multimodalembedding@001",
                input=["hello"],
                extra_headers={"X-Custom-Header": "multi-test"}
            )
        except Exception:
            pass
        mock_vertex_multi.assert_called_once()
        call_kwargs = mock_vertex_multi.call_args.kwargs
        assert call_kwargs.get("headers") == {"X-Custom-Header": "multi-test"}