"""
Tests for Router embedding chunking functionality.

This tests the enforce_embedding_context_limit feature which automatically
chunks large embedding inputs that exceed the model's context limit.
"""

from typing import List
from unittest.mock import patch

import pytest

from litellm import Router
from litellm.types.utils import Usage
from litellm.utils import EmbeddingResponse


class TestEmbeddingChunkingHelpers:
    """Test the helper methods for embedding chunking."""

    def setup_method(self):
        """Set up a router instance for testing."""
        self.router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=True,
            embedding_chunk_size=512,
        )

    def test_approx_token_count_short_text(self):
        """Test token count estimation for short text."""
        text = "Hello world"  # 11 chars -> ~2-3 tokens
        count = self.router._approx_token_count(text)
        assert count == 2  # 11 // 4 = 2

    def test_approx_token_count_long_text(self):
        """Test token count estimation for long text."""
        text = "a" * 4000  # 4000 chars -> ~1000 tokens
        count = self.router._approx_token_count(text)
        assert count == 1000

    def test_chunk_text_small_input(self):
        """Test that small inputs are not chunked."""
        text = "This is a short text"  # Well under 512 tokens
        chunks = self.router._chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_large_input(self):
        """Test that large inputs are properly chunked."""
        # Create text that's ~1000 tokens (4000 chars)
        text = "word " * 800  # 4000 chars = 1000 tokens
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Each chunk should be under the limit (512 tokens * 4 chars = 2048 chars)
        for chunk in chunks:
            assert len(chunk) <= 512 * 4 + 100  # Allow some buffer for word boundary

    def test_chunk_text_preserves_content(self):
        """Test that chunking doesn't lose content."""
        text = "word " * 800
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Rejoined chunks should contain all words
        rejoined = "".join(chunks)
        # Account for potential whitespace differences
        assert rejoined.replace(" ", "").replace("\n", "") == text.replace(" ", "")

    def test_chunk_text_breaks_on_word_boundary(self):
        """Test that chunks break on word boundaries when possible."""
        text = "hello world " * 500  # Long text with clear word boundaries
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Each chunk (except possibly last) should end with space or be at word boundary
        for chunk in chunks[:-1]:
            # Should not break in middle of "hello" or "world"
            stripped = chunk.rstrip()
            assert stripped.endswith("hello") or stripped.endswith("world")

    def test_merge_embeddings_single(self):
        """Test merging a single embedding returns it unchanged."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = self.router._merge_embeddings([embedding])
        assert result == embedding

    def test_merge_embeddings_multiple(self):
        """Test merging multiple embeddings averages them."""
        embeddings = [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]
        result = self.router._merge_embeddings(embeddings)

        # Average of [1,3], [2,4], [3,5] = [2, 3, 4]
        assert result == [2.0, 3.0, 4.0]

    def test_merge_embeddings_empty(self):
        """Test merging empty list returns empty list."""
        result = self.router._merge_embeddings([])
        assert result == []


class TestEmbeddingChunkingIntegration:
    """Integration tests for embedding chunking with mocked API calls."""

    def _create_mock_embedding_response(
        self, embedding: List[float]
    ) -> EmbeddingResponse:
        """Helper to create a mock embedding response."""
        return EmbeddingResponse(
            model="text-embedding-ada-002",
            data=[{"object": "embedding", "embedding": embedding, "index": 0}],
            usage=Usage(prompt_tokens=10, total_tokens=10),
        )

    @patch.object(Router, "function_with_fallbacks")
    def test_embedding_chunking_triggered_for_large_input(self, mock_fallbacks):
        """Test that chunking is triggered for inputs exceeding chunk size."""
        # Setup
        router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=True,
            embedding_chunk_size=100,  # Small chunk size to trigger chunking
        )

        # Use a callable that returns incrementing embeddings
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            # Return different embedding each call
            base = call_count[0] * 0.1
            return self._create_mock_embedding_response([base, base + 0.1, base + 0.2])

        mock_fallbacks.side_effect = mock_response

        # Create input that exceeds 100 tokens (~400 chars)
        # 1000 chars / 4 = 250 tokens, with chunk_size=100, this creates 3 chunks
        large_input = "test " * 200  # 1000 chars = ~250 tokens

        # Execute
        response = router.embedding(model="test-embedding", input=large_input)

        # Verify chunking happened (multiple API calls)
        assert call_count[0] >= 2, f"Expected at least 2 chunks, got {call_count[0]}"

        # Verify response structure
        assert len(response.data) == 1
        assert "embedding" in response.data[0]

        # Verify we got a merged embedding (list of floats)
        merged = response.data[0]["embedding"]
        assert isinstance(merged, list)
        assert len(merged) == 3
        assert all(isinstance(v, float) for v in merged)

    @patch.object(Router, "function_with_fallbacks")
    def test_embedding_no_chunking_for_small_input(self, mock_fallbacks):
        """Test that small inputs are not chunked."""
        router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=True,
            embedding_chunk_size=512,
        )

        mock_fallbacks.return_value = self._create_mock_embedding_response(
            [0.1, 0.2, 0.3]
        )

        # Small input
        small_input = "hello world"

        response = router.embedding(model="test-embedding", input=small_input)

        # Should only make one call
        assert mock_fallbacks.call_count == 1
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]

    @patch.object(Router, "function_with_fallbacks")
    def test_embedding_chunking_disabled(self, mock_fallbacks):
        """Test that chunking can be disabled."""
        router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=False,  # Disabled
            embedding_chunk_size=100,
        )

        mock_fallbacks.return_value = self._create_mock_embedding_response(
            [0.1, 0.2, 0.3]
        )

        # Large input that would normally be chunked
        large_input = "test " * 200

        _ = router.embedding(model="test-embedding", input=large_input)

        # Should only make one call (no chunking)
        assert mock_fallbacks.call_count == 1

    @patch.object(Router, "function_with_fallbacks")
    def test_embedding_list_input_with_chunking(self, mock_fallbacks):
        """Test chunking with list of inputs."""
        router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=True,
            embedding_chunk_size=100,
        )

        # Use callable to handle variable number of calls
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            base = call_count[0] * 0.1
            return self._create_mock_embedding_response([base, base + 0.1])

        mock_fallbacks.side_effect = mock_response

        inputs = [
            "test " * 200,  # Large - needs chunking (~250 tokens)
            "small text",  # Small - no chunking
        ]

        response = router.embedding(model="test-embedding", input=inputs)

        # Should have 2 embeddings in response (one per input)
        assert len(response.data) == 2

        # Verify both are valid embeddings
        assert isinstance(response.data[0]["embedding"], list)
        assert isinstance(response.data[1]["embedding"], list)

        # Multiple calls should have been made (at least 2 for large + 1 for small)
        assert call_count[0] >= 3, f"Expected at least 3 calls, got {call_count[0]}"


class TestEmbeddingChunkingAsync:
    """Test async embedding chunking."""

    def _create_mock_embedding_response(
        self, embedding: List[float]
    ) -> EmbeddingResponse:
        """Helper to create a mock embedding response."""
        return EmbeddingResponse(
            model="text-embedding-ada-002",
            data=[{"object": "embedding", "embedding": embedding, "index": 0}],
            usage=Usage(prompt_tokens=10, total_tokens=10),
        )

    @pytest.mark.asyncio
    @patch.object(Router, "async_function_with_fallbacks")
    async def test_aembedding_chunking(self, mock_async_fallbacks):
        """Test async embedding with chunking."""
        router = Router(
            model_list=[
                {
                    "model_name": "test-embedding",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            enforce_embedding_context_limit=True,
            embedding_chunk_size=100,
        )

        # Use async callable that returns incrementing embeddings
        call_count = [0]

        async def mock_response(*args, **kwargs):
            call_count[0] += 1
            base = call_count[0] * 0.1
            return self._create_mock_embedding_response([base, base + 0.1, base + 0.2])

        mock_async_fallbacks.side_effect = mock_response

        large_input = "test " * 200

        response = await router.aembedding(model="test-embedding", input=large_input)

        # Verify chunking happened
        assert call_count[0] >= 2, f"Expected at least 2 chunks, got {call_count[0]}"

        # Verify response structure
        assert len(response.data) == 1
        merged = response.data[0]["embedding"]
        assert isinstance(merged, list)
        assert len(merged) == 3


class TestRouterEmbeddingDefaults:
    """Test default configuration for embedding chunking."""

    def test_default_enforce_embedding_context_limit_is_false(self):
        """Test that enforce_embedding_context_limit defaults to False (opt-in feature)."""
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ]
        )
        # Default is False - users must opt-in to chunking
        assert router.enforce_embedding_context_limit is False

    def test_default_embedding_chunk_size_is_512(self):
        """Test that embedding_chunk_size defaults to 512."""
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ]
        )
        assert router.embedding_chunk_size == 512

    def test_custom_embedding_chunk_size(self):
        """Test that embedding_chunk_size can be customized."""
        router = Router(
            model_list=[
                {
                    "model_name": "test",
                    "litellm_params": {"model": "text-embedding-ada-002"},
                }
            ],
            embedding_chunk_size=2048,
        )
        assert router.embedding_chunk_size == 2048
