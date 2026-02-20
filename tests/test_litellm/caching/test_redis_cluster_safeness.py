import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from litellm.caching.redis_cache import RedisCache

@pytest.fixture
def mock_redis():
    with patch("litellm._redis.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # Mock _setup_health_pings to avoid connection attempts
        with patch("litellm.caching.redis_cache.RedisCache._setup_health_pings"):
            yield mock_client

def test_redis_cache_batch_get_uses_pipeline(mock_redis):
    cache = RedisCache(host="localhost", port=6379)
    
    # Mock pipeline for sync Redis
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value.__enter__.return_value = mock_pipe
    
    # Simulate Redis returning JSON strings
    mock_pipe.execute.return_value = [
        json.dumps("val1").encode("utf-8"),
        json.dumps("val2").encode("utf-8")
    ]
    
    keys = ["key1", "key2"]
    results = cache.batch_get_cache(keys)
    
    mock_redis.pipeline.assert_called_with(transaction=False)
    assert mock_pipe.get.call_count == 2
    mock_pipe.execute.assert_called_once()
    assert results == {"key1": "val1", "key2": "val2"}

@pytest.mark.asyncio
async def test_redis_cache_async_batch_get_uses_pipeline():
    with patch("litellm.caching.redis_cache.RedisCache.init_async_client") as mock_init:
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_init.return_value = mock_client
        
        # In redis-py, pipeline() itself is not async, but its context manager and execute() are.
        # But for simplicity in mock:
        mock_client.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[
            json.dumps("val1").encode("utf-8"),
            json.dumps("val2").encode("utf-8")
        ])
        
        with patch("litellm.caching.redis_cache.RedisCache._setup_health_pings"):
            cache = RedisCache(host="localhost", port=6379)
            keys = ["key1", "key2"]
            results = await cache.async_batch_get_cache(keys)
            
            mock_client.pipeline.assert_called_with(transaction=False)
            assert mock_pipe.get.call_count == 2
            assert mock_pipe.execute.call_count == 1
            assert results == {"key1": "val1", "key2": "val2"}

@pytest.mark.asyncio
async def test_redis_cache_delete_cache_keys_uses_pipeline():
    with patch("litellm.caching.redis_cache.RedisCache.init_async_client") as mock_init:
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_init.return_value = mock_client
        
        mock_client.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock()
        
        with patch("litellm.caching.redis_cache.RedisCache._setup_health_pings"):
            cache = RedisCache(host="localhost", port=6379)
            keys = ["key1", "key2"]
            await cache.delete_cache_keys(keys)
            
            mock_client.pipeline.assert_called_with(transaction=False)
            assert mock_pipe.delete.call_count == 2
            assert mock_pipe.execute.call_count == 1
