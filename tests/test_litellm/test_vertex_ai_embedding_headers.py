import pytest
from unittest.mock import patch, MagicMock
from litellm import embedding

def test_vertex_ai_embedding_extra_headers():
    """Test standard vertex embedding header forwarding."""
    with patch("litellm.main.vertex_embedding.embedding") as mock_vertex:
        mock_vertex.return_value = MagicMock()
        try:
            embedding(
                model="vertex_ai/text-embedding-004",
                input=["hello"],
                extra_headers={"X-Custom-Header": "test-value"},
            )
        except Exception:
            pass
        mock_vertex.assert_called_once()
        assert mock_vertex.call_args.kwargs.get("extra_headers") == {"X-Custom-Header": "test-value"}

def test_vertex_multimodal_embedding_headers():
    """Test multimodal vertex embedding header forwarding."""
    with patch("litellm.main.vertex_multimodal_embedding.multimodal_embedding") as mock_multimodal:
        mock_multimodal.return_value = MagicMock()
        try:
            # Using a multimodal model triggers the different provider path
            embedding(
                model="vertex_ai/multimodalembedding@001",
                input=["hello"],
                extra_headers={"X-Custom-Header": "multi-value"},
            )
        except Exception:
            pass
        mock_multimodal.assert_called_once()
        # NOTE: The multimodal handler uses 'headers' as the parameter name
        assert mock_multimodal.call_args.kwargs.get("headers") == {"X-Custom-Header": "multi-value"}
        print("\n✅ Success: Headers forwarded for both standard and multimodal Vertex AI!")
