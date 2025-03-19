import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch('asyncio.get_running_loop') as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.parametrize("namespace", [None, "test"])
@pytest.mark.asyncio
async def test_redis_cache_async_increment(namespace, monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    assert redis_cache is not None

    expected_key = "test:test" if namespace else "test"

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_set_cache
        await redis_cache.async_increment(key=expected_key, value=1)

        # Verify that the set method was called on the mock Redis instance
        mock_redis_instance.incrbyfloat.assert_called_once_with(
            name=expected_key, amount=1
        )


@pytest.mark.asyncio
async def test_redis_client_init_with_socket_timeout(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "my-fake-host")
    redis_cache = RedisCache(socket_timeout=1.0)
    assert redis_cache.redis_kwargs["socket_timeout"] == 1.0
    client = redis_cache.init_async_client()
    assert client is not None
    assert client.connection_pool.connection_kwargs["socket_timeout"] == 1.0


@pytest.mark.asyncio
async def test_redis_cache_async_batch_get_cache(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()
    
    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None
    
    # Setup the return value for mget
    mock_redis_instance.mget.return_value = [
        b'{"key1": "value1"}',
        None,
        b'{"key3": "value3"}'
    ]
    
    test_keys = ["key1", "key2", "key3"]
    
    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_batch_get_cache
        result = await redis_cache.async_batch_get_cache(key_list=test_keys)
        
        # Verify mget was called with the correct keys
        mock_redis_instance.mget.assert_called_once()
        
        # Check that results were properly decoded
        assert result["key1"] == {"key1": "value1"}
        assert result["key2"] is None
        assert result["key3"] == {"key3": "value3"}


# Tests for RedisSemanticCache
@pytest.mark.asyncio
async def test_redis_semantic_cache_initialization(monkeypatch):
    # Mock the redisvl import
    semantic_cache_mock = MagicMock()
    with patch.dict("sys.modules", {
        "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
        "redisvl.utils.vectorize": MagicMock(CustomTextVectorizer=MagicMock())
    }):
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


@pytest.mark.asyncio
async def test_redis_semantic_cache_get_cache(monkeypatch):
    # Mock the redisvl import and embedding function
    semantic_cache_mock = MagicMock()
    custom_vectorizer_mock = MagicMock()
    
    with patch.dict("sys.modules", {
        "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
        "redisvl.utils.vectorize": MagicMock(CustomTextVectorizer=custom_vectorizer_mock)
    }):
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
                "vector_distance": 0.1  # Distance of 0.1 means similarity of 0.9
            }
        ]
        redis_semantic_cache.llmcache.check = MagicMock(return_value=mock_result)
        
        # Mock the embedding function
        with patch("litellm.embedding", return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}):
            # Test get_cache with a message
            result = redis_semantic_cache.get_cache(
                key="test_key",
                messages=[{"content": "What is the capital of France?"}]
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
    
    with patch.dict("sys.modules", {
        "redisvl.extensions.llmcache": MagicMock(SemanticCache=semantic_cache_mock),
        "redisvl.utils.vectorize": MagicMock(CustomTextVectorizer=custom_vectorizer_mock)
    }):
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
                "vector_distance": 0.1  # Distance of 0.1 means similarity of 0.9
            }
        ]
        
        redis_semantic_cache.llmcache.acheck = AsyncMock(return_value=mock_result)
        redis_semantic_cache._get_async_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        # Test async_get_cache with a message
        result = await redis_semantic_cache.async_get_cache(
            key="test_key",
            messages=[{"content": "What is the capital of France?"}],
            metadata={}
        )
        
        # Verify result is properly parsed
        assert result == {"content": "Paris is the capital of France."}
        
        # Verify methods were called
        redis_semantic_cache._get_async_embedding.assert_called_once()
        redis_semantic_cache.llmcache.acheck.assert_called_once()
