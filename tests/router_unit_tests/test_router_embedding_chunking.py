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
        """Test token count estimation for short text.

        With safety factor of 1.1, the estimate is conservative (higher):
        11 chars / 4 * 1.1 = 3 (rounded)
        """
        text = "Hello world"  # 11 chars
        count = self.router._approx_token_count(text)
        assert count == 3  # int((11 / 4) * 1.1) = 3

    def test_approx_token_count_long_text(self):
        """Test token count estimation for long text.

        With safety factor of 1.1, the estimate is conservative (higher):
        4000 chars / 4 * 1.1 = 1100
        """
        text = "a" * 4000  # 4000 chars
        count = self.router._approx_token_count(text)
        assert count == 1100  # int((4000 / 4) * 1.1) = 1100

    def test_chunk_text_small_input(self):
        """Test that small inputs are not chunked."""
        text = "This is a short text"  # Well under 512 tokens
        chunks = self.router._chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_large_input(self):
        """Test that large inputs are properly chunked."""
        # Create text that's ~1100 tokens (4000 chars)
        text = "word " * 800  # 4000 chars = 1100 tokens (with 1.1 safty margin) 
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Each chunk should be under the limit.
        # Base limit: 512 tokens * 4 chars/token * 0.9 safety margin = 1843 chars
        # Buffer of 200 chars accounts for word boundary search which looks back
        # up to 10% of chunk size (max 200 chars) to find a space.
        max_chunk_chars = int(512 * 4 * 0.9) + 200
        for chunk in chunks:
            assert (
                len(chunk) <= max_chunk_chars
            ), f"Chunk too large: {len(chunk)} > {max_chunk_chars}"

    def test_chunk_text_preserves_content(self):
        """Test that chunking doesn't lose content."""
        text = "word " * 800
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Verify no words are lost by checking word count
        original_words = text.split()
        rejoined_words = "".join(chunks).split()
        assert len(rejoined_words) == len(
            original_words
        ), f"Word count mismatch: original={len(original_words)}, rejoined={len(rejoined_words)}"

        # Also verify the actual content matches (ignoring whitespace normalization)
        # Chunking may adjust whitespace at boundaries, but words must be identical
        assert rejoined_words == original_words, "Words in chunks don't match original"

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

        # Create input that exceeds 100 tokens
        # With 1000 chars and the token estimation formula: int((chars/4) * 1.1)
        # Estimated tokens = int((1000/4) * 1.1) = 275 tokens
        # With chunk_size=100 and 0.9 safety margin, effective chunk = 90 tokens
        # Expected chunks = ceil(275/90) = 4 chunks (approximately)
        large_input = "test " * 200  # 1000 chars

        # Execute
        response = router.embedding(model="test-embedding", input=large_input)

        # Verify chunking happened - with 275 estimated tokens and ~90 token chunks,
        # we expect 3-4 chunks depending on word boundary adjustments
        assert call_count[0] >= 3, f"Expected at least 3 chunks, got {call_count[0]}"
        assert call_count[0] <= 5, f"Expected at most 5 chunks, got {call_count[0]}"

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


class TestEmbeddingInputValidation:
    """Test input validation for embedding methods."""

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

    def test_validate_embedding_input_string(self):
        """Test that string input is wrapped in a list."""
        result = self.router._validate_embedding_input("hello world")
        assert result == ["hello world"]

    def test_validate_embedding_input_list_of_strings(self):
        """Test that valid list of strings passes through."""
        inputs = ["hello", "world"]
        result = self.router._validate_embedding_input(inputs)
        assert result == ["hello", "world"]

    def test_validate_embedding_input_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input([])
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_embedding_input_non_string_elements_raises(self):
        """Test that list with non-string elements raises ValueError."""
        # Integer in list
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input(["hello", 123, "world"])
        assert "non-string items" in str(exc_info.value)
        assert "int" in str(exc_info.value)

        # None in list
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input(["hello", None])
        assert "non-string items" in str(exc_info.value)
        assert "NoneType" in str(exc_info.value)

    def test_validate_embedding_input_invalid_type_raises(self):
        """Test that invalid types raise ValueError."""
        # Integer
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input(123)  # type: ignore[arg-type]
        assert "must be a string or list" in str(exc_info.value)

        # Dictionary
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input({"text": "hello"})  # type: ignore[arg-type]
        assert "must be a string or list" in str(exc_info.value)

    def test_validate_embedding_input_mixed_types_shows_indices(self):
        """Test that error message shows indices of non-string items."""
        with pytest.raises(ValueError) as exc_info:
            self.router._validate_embedding_input(["a", 1, "b", 2.0, "c"])
        error_msg = str(exc_info.value)
        # Should mention indices 1 and 3
        assert "1" in error_msg or "3" in error_msg


class TestEmbeddingChunkingEdgeCases:
    """Test edge cases for embedding chunking."""

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

    def test_chunk_text_empty_string(self):
        """Test chunking behavior with empty string input."""
        chunks = self.router._chunk_text("", chunk_size=512)
        # Empty string should return empty list (no chunks)
        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking behavior with whitespace-only input."""
        # Single space
        chunks = self.router._chunk_text("   ", chunk_size=512)
        assert chunks == []  # Whitespace-only chunks are skipped

        # Newlines only
        chunks = self.router._chunk_text("\n\n\n", chunk_size=512)
        assert chunks == []

        # Mixed whitespace
        chunks = self.router._chunk_text("  \n  \t  ", chunk_size=512)
        assert chunks == []

    def test_chunk_text_no_spaces_long_string(self):
        """Test chunking text with no word boundaries (continuous characters).

        When there are no spaces to break on, the chunker should still
        produce chunks that don't exceed the size limit, even if it means
        breaking mid-"word".
        """
        # Create a long string with no spaces - 10000 chars = ~2500 tokens
        text = "a" * 10000
        chunks = self.router._chunk_text(text, chunk_size=512)

        # Should create multiple chunks
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

        # Each chunk should respect the size limit
        # 512 tokens * 4 chars * 0.9 safety = 1843 chars max
        max_chunk_chars = int(512 * 4 * 0.9)
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_chunk_chars + 50, (
                f"Chunk {i} too large: {len(chunk)} > {max_chunk_chars}"
            )

        # All content should be preserved
        assert "".join(chunks) == text

    def test_merge_embeddings_different_dimensions(self):
        """Test that merging embeddings with different dimensions raises an error or handles gracefully.

        Note: The current implementation uses zip() which will silently truncate
        to the shortest vector. This test documents the current behavior.
        """
        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],  # Different dimension
        ]
        # Current behavior: zip truncates to shortest
        result = self.router._merge_embeddings(embeddings)
        # Result will have length of shortest vector (2)
        assert len(result) == 2
        assert result == [2.5, 3.5]  # Average of [1,4] and [2,5]

    def test_chunk_text_single_word_exceeds_limit(self):
        """Test chunking when a single 'word' exceeds the chunk limit."""
        # A single very long word (no spaces)
        long_word = "supercalifragilisticexpialidocious" * 100  # ~3400 chars
        chunks = self.router._chunk_text(long_word, chunk_size=100)

        # Should still chunk despite no word boundaries
        assert len(chunks) >= 2

        # Content preserved
        assert "".join(chunks) == long_word

    def test_chunk_text_unicode_content(self):
        """Test chunking with unicode/non-ASCII content."""
        # Chinese text (characters typically use more tokens)
        chinese_text = "这是一个测试文本。" * 200
        chunks = self.router._chunk_text(chinese_text, chunk_size=512)

        # Should produce chunks
        assert len(chunks) >= 1

        # Content preserved
        assert "".join(chunks) == chinese_text

    def test_chunk_text_mixed_content(self):
        """Test chunking with mixed ASCII and unicode content."""
        mixed_text = "Hello 世界! " * 500
        chunks = self.router._chunk_text(mixed_text, chunk_size=512)

        # Should produce multiple chunks
        assert len(chunks) >= 1

        # Content preserved
        assert "".join(chunks) == mixed_text
