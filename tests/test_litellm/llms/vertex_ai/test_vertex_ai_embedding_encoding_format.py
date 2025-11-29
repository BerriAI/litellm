"""
Test that encoding_format parameter is accepted for Vertex AI / Gemini embeddings.

This test verifies the fix for the issue where the OpenAI client sends
encoding_format: base64 automatically, which was causing UnsupportedParamsError
for Gemini embeddings.

Reference: https://github.com/BerriAI/litellm/issues/XXXX
"""

import pytest

import litellm
from litellm.utils import get_optional_params_embeddings, get_llm_provider


class TestVertexAIEmbeddingEncodingFormat:
    """Test encoding_format parameter handling for Vertex AI embeddings."""

    def test_vertex_ai_embedding_accepts_encoding_format_without_drop_params(self):
        """
        Test that encoding_format is accepted without needing drop_params=True.
        
        The OpenAI Python client automatically sends encoding_format: base64.
        This should not raise an UnsupportedParamsError for Vertex AI embeddings.
        """
        litellm.drop_params = False
        
        model, custom_llm_provider, _, _ = get_llm_provider(
            model="vertex_ai/textembedding-gecko"
        )
        
        optional_params = get_optional_params_embeddings(
            model=model,
            encoding_format="base64",
            custom_llm_provider=custom_llm_provider,
        )
        
        assert "encoding_format" not in optional_params

    def test_gemini_embedding_accepts_encoding_format_without_drop_params(self):
        """
        Test that encoding_format is accepted for Gemini embeddings.
        
        The OpenAI Python client automatically sends encoding_format: base64.
        This should not raise an UnsupportedParamsError for Gemini embeddings.
        """
        litellm.drop_params = False
        
        model, custom_llm_provider, _, _ = get_llm_provider(
            model="gemini/text-embedding-004"
        )
        
        optional_params = get_optional_params_embeddings(
            model=model,
            encoding_format="base64",
            custom_llm_provider=custom_llm_provider,
        )
        
        assert "encoding_format" not in optional_params

    def test_vertex_ai_embedding_with_dimensions_and_encoding_format(self):
        """
        Test that both dimensions and encoding_format work together.
        
        dimensions should be passed through, encoding_format should be ignored.
        """
        litellm.drop_params = False
        
        model, custom_llm_provider, _, _ = get_llm_provider(
            model="vertex_ai/text-embedding-005"
        )
        
        optional_params = get_optional_params_embeddings(
            model=model,
            dimensions=256,
            encoding_format="base64",
            custom_llm_provider=custom_llm_provider,
        )
        
        assert optional_params.get("outputDimensionality") == 256
        assert "encoding_format" not in optional_params

    def test_vertex_ai_embedding_encoding_format_float(self):
        """
        Test that encoding_format: float is also accepted and ignored.
        """
        litellm.drop_params = False
        
        model, custom_llm_provider, _, _ = get_llm_provider(
            model="vertex_ai/textembedding-gecko"
        )
        
        optional_params = get_optional_params_embeddings(
            model=model,
            encoding_format="float",
            custom_llm_provider=custom_llm_provider,
        )
        
        assert "encoding_format" not in optional_params

