import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ResponseError

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch('asyncio.get_running_loop') as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.asyncio
async def test_redis_moved_error_handling(redis_no_ping):
    """
    Test that RedisCache properly handles MOVED errors from Redis Cluster
    by reinitializing the client and retrying the operation.
    """
    redis_cache = RedisCache(host="localhost", port=6379)
    
    # Create a mock for the initial Redis client that will raise a MOVED error
    mock_initial_client = AsyncMock()
    mock_initial_client.incrbyfloat.side_effect = ResponseError("MOVED 8116 172.16.7.254:6379")
    
    # Create a mock for the reinitialized Redis client that will succeed
    mock_new_client = AsyncMock()
    mock_new_client.incrbyfloat.return_value = 42.0
    mock_new_client.ttl.return_value = -1  # Simulate no TTL set
    mock_new_client.expire.return_value = True
    
    # Setup the init_async_client method to return the mock clients in sequence
    redis_cache.init_async_client = MagicMock()
    redis_cache.init_async_client.side_effect = [mock_initial_client, mock_new_client]
    
    # Call the method being tested
    result = await redis_cache.async_increment(key="test_key", value=1.0, ttl=60)
    
    # Verify that init_async_client was called twice (initial + after MOVED error)
    assert redis_cache.init_async_client.call_count == 2
    
    # Verify the initial client received the incrbyfloat call
    mock_initial_client.incrbyfloat.assert_called_once_with(name="test_key", amount=1.0)
    
    # Verify the new client received the incrbyfloat call after reinitialization
    mock_new_client.incrbyfloat.assert_called_once_with(name="test_key", amount=1.0)
    
    # Verify TTL was checked and set on the new client
    mock_new_client.ttl.assert_called_once_with("test_key")
    mock_new_client.expire.assert_called_once_with("test_key", 60)
    
    # Verify the correct result was returned
    assert result == 42.0


@pytest.mark.asyncio
async def test_redis_moved_error_retry_fails(redis_no_ping):
    """
    Test that RedisCache properly handles the case where the retry after a MOVED error
    also fails with another exception.
    """
    redis_cache = RedisCache(host="localhost", port=6379)
    
    # Create a mock for the initial Redis client that will raise a MOVED error
    mock_initial_client = AsyncMock()
    mock_initial_client.incrbyfloat.side_effect = ResponseError("MOVED 8116 172.16.7.254:6379")
    
    # Create a mock for the reinitialized Redis client that will also fail
    mock_new_client = AsyncMock()
    mock_new_client.incrbyfloat.side_effect = Exception("Connection refused")
    
    # Setup the init_async_client method to return the mock clients in sequence
    redis_cache.init_async_client = MagicMock()
    redis_cache.init_async_client.side_effect = [mock_initial_client, mock_new_client]
    
    # Call the method being tested and expect it to raise an exception
    with pytest.raises(Exception, match="Connection refused"):
        await redis_cache.async_increment(key="test_key", value=1.0)
    
    # Verify that init_async_client was called twice (initial + after MOVED error)
    assert redis_cache.init_async_client.call_count == 2
    
    # Verify the initial client received the incrbyfloat call
    mock_initial_client.incrbyfloat.assert_called_once_with(name="test_key", amount=1.0)
    
    # Verify the new client received the incrbyfloat call after reinitialization
    mock_new_client.incrbyfloat.assert_called_once_with(name="test_key", amount=1.0)