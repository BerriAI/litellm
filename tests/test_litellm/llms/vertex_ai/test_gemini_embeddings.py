"""
Test Gemini Embedding Models on Vertex AI

Tests the transformation logic for Gemini embedding models that use
the :embedContent endpoint instead of :predict.
"""

from litellm.llms.vertex_ai.vertex_embeddings.gemini_embeddings import (
    GEMINI_EMBEDDING_MODELS,
    VertexGeminiEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


class TestVertexGeminiEmbeddingConfig:
    """Test VertexGeminiEmbeddingConfig class"""

    def test_is_gemini_embedding_model(self):
        """Test model detection for Gemini embedding models"""
        # Should detect Gemini embedding models
        assert VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "gemini-embedding-001"
        )
        assert VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "gemini-embedding-2-exp-11-2025"
        )
        assert VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "text-embedding-005"
        )
        assert VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "text-multilingual-embedding-002"
        )

        # Should handle routing prefixes
        assert VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "vertex_ai/gemini-embedding-001"
        )

        # Should not detect non-Gemini models
        assert not VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "textembedding-gecko"
        )
        assert not VertexGeminiEmbeddingConfig.is_gemini_embedding_model(
            "text-embedding-ada-002"
        )

    def test_transform_request_single_input(self):
        """Test request transformation for single input"""
        input_text = "Hello, world!"
        optional_params = {
            "task_type": "RETRIEVAL_QUERY",
            "outputDimensionality": 768,
        }

        result = VertexGeminiEmbeddingConfig.transform_request(
            input=input_text,
            optional_params=optional_params,
            model="gemini-embedding-001",
        )

        # Verify structure
        assert "content" in result
        assert "parts" in result["content"]
        assert len(result["content"]["parts"]) == 1
        assert result["content"]["parts"][0]["text"] == input_text

        # Verify optional params
        assert result["taskType"] == "RETRIEVAL_QUERY"
        assert result["outputDimensionality"] == 768

        # Should not have batch flag for single input
        assert "_batch_inputs" not in result

    def test_transform_request_multiple_inputs(self):
        """Test request transformation for multiple inputs"""
        input_texts = ["Hello, world!", "Goodbye, world!"]
        optional_params = {"task_type": "SEMANTIC_SIMILARITY"}

        result = VertexGeminiEmbeddingConfig.transform_request(
            input=input_texts,
            optional_params=optional_params,
            model="gemini-embedding-001",
        )

        # Should have batch flag for multiple inputs
        assert "_batch_inputs" in result
        assert result["_batch_inputs"] == input_texts

        # First input should be in content
        assert "content" in result
        assert result["content"]["parts"][0]["text"] == input_texts[0]

        # Verify optional params
        assert result["taskType"] == "SEMANTIC_SIMILARITY"

    def test_transform_response_single_embedding(self):
        """Test response transformation for single embedding"""
        response = {"embedding": {"values": [0.1, 0.2, 0.3]}}

        model_response = EmbeddingResponse()
        result = VertexGeminiEmbeddingConfig.transform_response(
            response=response,
            model="gemini-embedding-001",
            model_response=model_response,
        )

        # Verify structure
        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.model == "gemini-embedding-001"

        # Verify usage
        assert hasattr(result, "usage")
        assert result.usage.prompt_tokens == 0  # Not provided in response
        assert result.usage.total_tokens == 0

    def test_transform_response_multiple_embeddings(self):
        """Test response transformation for multiple embeddings"""
        responses = [
            {"embedding": {"values": [0.1, 0.2, 0.3]}},
            {"embedding": {"values": [0.4, 0.5, 0.6]}},
        ]

        model_response = EmbeddingResponse()
        result = VertexGeminiEmbeddingConfig.transform_response(
            response=responses,
            model="gemini-embedding-001",
            model_response=model_response,
        )

        # Verify structure
        assert result.object == "list"
        assert len(result.data) == 2

        # First embedding
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]

        # Second embedding
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]

        assert result.model == "gemini-embedding-001"

    def test_gemini_embedding_models_list(self):
        """Test that GEMINI_EMBEDDING_MODELS contains expected models"""
        assert "gemini-embedding-001" in GEMINI_EMBEDDING_MODELS
        assert "gemini-embedding-2-exp-11-2025" in GEMINI_EMBEDDING_MODELS
        assert "text-embedding-005" in GEMINI_EMBEDDING_MODELS
        assert "text-multilingual-embedding-002" in GEMINI_EMBEDDING_MODELS
