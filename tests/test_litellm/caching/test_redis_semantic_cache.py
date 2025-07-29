import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


# Tests for RedisSemanticCache
def test_redis_semantic_cache_initialization(monkeypatch):
    # Mock the redisvl import
    semantic_cache_mock = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
            "redisvl.utils.vectorize": MagicMock(CustomTextVectorizer=MagicMock()),
        },
    ):
        from litellm.caching.redis_semantic_cache import RedisSemanticCache

        # Set environment variables
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_PASSWORD", "test_password")

        # Initialize the cache with a similarity threshold
        redis_semantic_cache = RedisSemanticCache(similarity_threshold=0.8)

        # Verify the semantic cache was initialized with correct parameters
        assert redis_semantic_cache.similarity_threshold == 0.8

        # Use pytest.approx for floating point comparison to handle precision issues
        assert redis_semantic_cache.distance_threshold == pytest.approx(0.2, abs=1e-10)
        assert redis_semantic_cache.embedding_model == "text-embedding-ada-002"

        # Test initialization with missing similarity_threshold
        with pytest.raises(ValueError, match="similarity_threshold must be provided"):
            RedisSemanticCache()


def test_redis_semantic_cache_get_cache(monkeypatch):
    # Mock the redisvl import and embedding function
    semantic_cache_mock = MagicMock()
    custom_vectorizer_mock = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
            "redisvl.utils.vectorize": MagicMock(
                CustomTextVectorizer=custom_vectorizer_mock
            ),
        },
    ):
        from litellm.caching.redis_semantic_cache import RedisSemanticCache

        # Set environment variables
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_PASSWORD", "test_password")

        # Initialize cache
        redis_semantic_cache = RedisSemanticCache(similarity_threshold=0.8)

        # Mock the llmcache.check method to return a result
        mock_result = [
            {
                "prompt": "What is the capital of France?",
                "response": '{"content": "Paris is the capital of France."}',
                "vector_distance": 0.1,  # Distance of 0.1 means similarity of 0.9
            }
        ]
        redis_semantic_cache.llmcache.check = MagicMock(return_value=mock_result)

        # Mock the embedding function
        with patch(
            "litellm.embedding", return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        ):
            # Test get_cache with a message
            result = redis_semantic_cache.get_cache(
                key="test_key", messages=[{"content": "What is the capital of France?"}]
            )

            # Verify result is properly parsed
            assert result == {"content": "Paris is the capital of France."}

            # Verify llmcache.check was called
            redis_semantic_cache.llmcache.check.assert_called_once()


@pytest.mark.asyncio
async def test_redis_semantic_cache_async_get_cache(monkeypatch):
    # Mock the redisvl import
    semantic_cache_mock = MagicMock()
    custom_vectorizer_mock = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
            "redisvl.utils.vectorize": MagicMock(
                CustomTextVectorizer=custom_vectorizer_mock
            ),
        },
    ):
        from litellm.caching.redis_semantic_cache import RedisSemanticCache

        # Set environment variables
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("REDIS_PASSWORD", "test_password")

        # Initialize cache
        redis_semantic_cache = RedisSemanticCache(similarity_threshold=0.8)

        # Mock the async methods
        mock_result = [
            {
                "prompt": "What is the capital of France?",
                "response": '{"content": "Paris is the capital of France."}',
                "vector_distance": 0.1,  # Distance of 0.1 means similarity of 0.9
            }
        ]

        redis_semantic_cache.llmcache.acheck = AsyncMock(return_value=mock_result)
        redis_semantic_cache._get_async_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        # Test async_get_cache with a message
        result = await redis_semantic_cache.async_get_cache(
            key="test_key",
            messages=[{"content": "What is the capital of France?"}],
            metadata={},
        )

        # Verify result is properly parsed
        assert result == {"content": "Paris is the capital of France."}

        # Verify methods were called
        redis_semantic_cache._get_async_embedding.assert_called_once()
        redis_semantic_cache.llmcache.acheck.assert_called_once()
