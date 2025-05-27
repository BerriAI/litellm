import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.redis_cluster_cache import RedisClusterCache


@patch("litellm._redis.init_redis_cluster")
def test_redis_cluster_batch_get(mock_init_redis_cluster):
    """
    Test that RedisClusterCache uses mget_nonatomic instead of mget for batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.mget_nonatomic.return_value = [None, None]  # Simulate no cache hits
    mock_init_redis_cluster.return_value = mock_redis

    # Create RedisClusterCache instance with mock client
    cache = RedisClusterCache(
        startup_nodes=[{"host": "localhost", "port": 6379}],
        password="hello",
    )

    # Test batch_get_cache
    keys = ["key1", "key2"]
    cache.batch_get_cache(keys)

    # Verify mget_nonatomic was called instead of mget
    mock_redis.mget_nonatomic.assert_called_once()
    assert not mock_redis.mget.called


@pytest.mark.asyncio
@patch("litellm._redis.init_redis_cluster")
async def test_redis_cluster_async_batch_get(mock_init_redis_cluster):
    """
    Test that RedisClusterCache uses mget_nonatomic instead of mget for async batch operations
    """
    # Create a mock Redis client
    mock_redis = MagicMock()
    mock_redis.mget_nonatomic.return_value = [None, None]  # Simulate no cache hits

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

    # Verify mget_nonatomic was called instead of mget
    mock_redis.mget_nonatomic.assert_called_once()
    assert not mock_redis.mget.called
