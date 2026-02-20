import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.caching.redis_cluster_cache import RedisClusterCache


@patch("litellm._redis.get_redis_client")
def test_redis_cluster_batch_get(mock_get_redis_client):
    """
    Test that RedisClusterCache uses pipeline instead of mget for batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value.__enter__.return_value = mock_pipe
    mock_pipe.execute.return_value = [None, None]
    
    mock_get_redis_client.return_value = mock_redis

    # Create RedisClusterCache instance with mock client
    cache = RedisClusterCache(
        startup_nodes=[{"host": "localhost", "port": 6379}],
        password="hello",
    )

    # Test batch_get_cache
    keys = ["key1", "key2"]
    cache.batch_get_cache(keys)

    # Verify pipeline was called
    mock_redis.pipeline.assert_called_once()
    assert mock_pipe.get.call_count == 2
    mock_pipe.execute.assert_called_once()


@pytest.mark.asyncio
@patch("litellm._redis.get_redis_client")
async def test_redis_cluster_async_batch_get(mock_get_redis_client):
    """
    Test that RedisClusterCache uses pipeline instead of mget for async batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    
    # Mock the async context manager
    mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipe
    mock_pipe.execute.return_value = [None, None]

    # Create RedisClusterCache instance with mock client
    cache = RedisClusterCache(
        startup_nodes=[{"host": "localhost", "port": 6379}],
        password="hello",
    )

    # Mock the init_async_client to return our mock redis client
    cache.init_async_client = MagicMock(return_value=mock_redis)

    # Test async_batch_get_cache
    keys = ["key1", "key2"]
    await cache.async_batch_get_cache(keys)

    # Verify pipeline was called
    mock_redis.pipeline.assert_called_once()
    assert mock_pipe.get.call_count == 2
    mock_pipe.execute.assert_called_once()
