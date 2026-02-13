import pytest
from unittest.mock import patch, MagicMock
from litellm import embedding

def test_vertex_ai_embedding_extra_headers():
    """
    Test that extra_headers are correctly forwarded to the 
    vertex_embedding.embedding function.
    """
    # We patch the exact location where main.py calls the vertex provider
    with patch("litellm.main.vertex_embedding.embedding") as mock_vertex_embedding:
        # Mock a successful return so the call doesn't fail
        mock_vertex_embedding.return_value = MagicMock()
        
        # Trigger the embedding call
        try:
            embedding(
                model="vertex_ai/text-embedding-004",
                input=["hello"],
                extra_headers={"X-Custom-Header": "test-value"},
            )
        except Exception:
            # We don't care about subsequent errors, only the forwarding
            pass
        
        # VERIFICATION: This is the important part
        mock_vertex_embedding.assert_called_once()
        call_kwargs = mock_vertex_embedding.call_args.kwargs
        
        # Check that the headers we passed actually reached the provider
        assert call_kwargs.get("extra_headers") == {"X-Custom-Header": "test-value"}
        print("\n✅ Success: extra_headers correctly forwarded to Vertex AI provider!")
