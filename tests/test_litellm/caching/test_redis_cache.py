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
    with patch("asyncio.get_running_loop") as mock_get_loop:
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
        b'{"key3": "value3"}',
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


@pytest.mark.asyncio
async def test_handle_lpop_count_for_older_redis_versions(monkeypatch):
    """Test the helper method that handles LPOP with count for Redis versions < 7.0"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    # Create RedisCache instance
    redis_cache = RedisCache()

    # Create a mock pipeline
    mock_pipeline = AsyncMock()
    # Set up execute to return different values each time
    mock_pipeline.execute.side_effect = [
        [b"value1"],  # First execute returns first value
        [b"value2"],  # Second execute returns second value
    ]

    # Test the helper method
    result = await redis_cache.handle_lpop_count_for_older_redis_versions(
        pipe=mock_pipeline, key="test_key", count=2
    )

    # Verify results
    assert result == [b"value1", b"value2"]
    assert mock_pipeline.lpop.call_count == 2
    assert mock_pipeline.execute.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redis_version",
    [
        # Standard cases
        "7.0.0",  # Standard Redis string version
        7.0,  # Valkey/ElastiCache float version (THE BUG this fix addresses)
        7,  # Integer version (e.g., from some Redis forks)
        
        # Version < 7
        "6",  # String without dots, version < 7
        
        # Malformed versions (fallback to 7)
        "latest",  # Non-numeric version
        "",  # Empty string
        -7.0,  # Negative float
        
        # Format variations
        " 7.0.0 ",  # Whitespace (should be stripped)
        "7.0.0-rc1",  # Version with suffix
        "10.0.0",  # Double digit major version
    ],
)
async def test_async_lpop_with_float_redis_version(
    monkeypatch, redis_no_ping, redis_version
):
    """
    Test async_lpop with various Redis version formats (especially float).
    
    This test specifically addresses the issue where AWS ElastiCache Valkey 
    returns redis_version as a float (e.g., 7.0) instead of a string (e.g., "7.0.0"),
    which caused a 'float' object has no attribute 'split' error when trying to 
    use the Redis transaction buffer feature.
    
    The fix converts the version to a string and handles edge cases like:
    - Floats (7.0) and integers (7)
    - Strings with/without dots ("7" vs "7.0.0")
    - Malformed versions ("v7.0.0", "latest") - fallback to version 7
    - Whitespace (" 7.0.0 ")
    - Negative versions (fallback to version 7)
    
    Related: Database deadlock issues when use_redis_transaction_buffer is enabled.
    """
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    
    # Create RedisCache instance
    redis_cache = RedisCache()
    redis_cache.redis_version = redis_version  # Set the version to test
    
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None
    
    # Mock lpop to return a test value (Redis >= 7.0 behavior)
    mock_redis_instance.lpop.return_value = [b"value1", b"value2"]
    
    # Mock pipeline for Redis < 7.0 (used when major_version < 7)
    mock_pipeline = MagicMock()
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    # Make pipeline() a regular method (not async) that returns the mock
    mock_redis_instance.pipeline = MagicMock(return_value=mock_pipeline)
    
    # Mock handle_lpop_count_for_older_redis_versions for Redis < 7
    with patch.object(
        redis_cache, "handle_lpop_count_for_older_redis_versions",
        return_value=[b"value1", b"value2"]
    ):
        with patch.object(
            redis_cache, "init_async_client", return_value=mock_redis_instance
        ):
            # Call async_lpop with count - this should not raise AttributeError
            result = await redis_cache.async_lpop(key="test_key", count=2)
            
            # Verify the method completed without error
            assert result is not None
