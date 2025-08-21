import os
import sys
from unittest.mock import MagicMock, patch
import json
import datetime
import asyncio

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.s3_cache import S3Cache


@pytest.fixture
def mock_s3_dependencies():
    mock_s3_client = MagicMock()
    
    with patch("boto3.client", return_value=mock_s3_client):
        yield {"s3_client": mock_s3_client}


@pytest.mark.asyncio
async def test_s3_cache_async_set_cache(mock_s3_dependencies):
    """Test async_set_cache functionality using run_in_executor"""
    cache = S3Cache("test-bucket")
    test_value = {"key": "value", "number": 42}
    
    await cache.async_set_cache("test_key", test_value)
    
    cache.s3_client.put_object.assert_called_once()
    call_args = cache.s3_client.put_object.call_args
    
    assert call_args[1]["Bucket"] == "test-bucket"
    assert call_args[1]["Key"] == "test_key"
    assert call_args[1]["Body"] == json.dumps(test_value)
    assert call_args[1]["ContentType"] == "application/json"
    assert call_args[1]["ContentLanguage"] == "en"
    assert call_args[1]["ContentDisposition"] == 'inline; filename="test_key.json"'


@pytest.mark.asyncio
async def test_s3_cache_async_set_cache_with_ttl(mock_s3_dependencies):
    """Test async_set_cache with TTL functionality"""
    cache = S3Cache("test-bucket")
    test_value = {"key": "value"}
    ttl = datetime.timedelta(seconds=3600)  # 1 hour

    await cache.async_set_cache("test_key", test_value, ttl=ttl)

    cache.s3_client.put_object.assert_called_once()
    call_args = cache.s3_client.put_object.call_args

    assert "Expires" in call_args[1]
    assert "CacheControl" in call_args[1]
    assert "max-age=1:00:00" in call_args[1]["CacheControl"]


@pytest.mark.asyncio
async def test_s3_cache_async_get_cache(mock_s3_dependencies):
    """Test async_get_cache functionality using run_in_executor"""
    cache = S3Cache("test-bucket")
    
    mock_response = {
        "Body": MagicMock()
    }
    mock_response["Body"].read.return_value = b'{"key": "value", "number": 42}'
    cache.s3_client.get_object.return_value = mock_response
    
    result = await cache.async_get_cache("test_key")
    
    cache.s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", 
        Key="test_key"
    )
    
    assert result == {"key": "value", "number": 42}


@pytest.mark.asyncio
async def test_s3_cache_async_get_cache_not_found(mock_s3_dependencies):
    """Test async_get_cache when key is not found"""
    import botocore.exceptions
    
    cache = S3Cache("test-bucket")
    
    error_response = {"Error": {"Code": "NoSuchKey"}}
    cache.s3_client.get_object.side_effect = botocore.exceptions.ClientError(
        error_response, "GetObject"
    )
    
    result = await cache.async_get_cache("nonexistent_key")
    
    cache.s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", 
        Key="nonexistent_key"
    )
    assert result is None


@pytest.mark.asyncio
async def test_s3_cache_async_set_cache_pipeline(mock_s3_dependencies):
    """Test async_set_cache_pipeline functionality"""
    cache = S3Cache("test-bucket")
    
    cache_list = [
        ("key1", {"data": "value1"}),
        ("key2", {"data": "value2"}),
        ("key3", {"data": "value3"}),
    ]
    
    await cache.async_set_cache_pipeline(cache_list)
    
    # Should have called put_object 3 times
    assert cache.s3_client.put_object.call_count == 3
    
    # Verify each call
    calls = cache.s3_client.put_object.call_args_list
    for i, (key, value) in enumerate(cache_list):
        call_args = calls[i][1]
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == key
        assert call_args["Body"] == json.dumps(value)


@pytest.mark.asyncio
async def test_s3_cache_concurrent_async_operations(mock_s3_dependencies):
    """Test concurrent async operations to ensure they don't block each other"""
    cache = S3Cache("test-bucket")
    
    # Create multiple concurrent set operations
    tasks = []
    for i in range(5):
        key = f"concurrent_key_{i}"
        value = {"id": i, "data": f"test_data_{i}"}
        tasks.append(cache.async_set_cache(key, value))
    
    # Execute all tasks concurrently
    await asyncio.gather(*tasks)
    
    # Verify all operations were called
    assert cache.s3_client.put_object.call_count == 5
    
    # Verify each call had correct parameters
    calls = cache.s3_client.put_object.call_args_list
    for i, call in enumerate(calls):
        call_args = call[1]
        assert call_args["Bucket"] == "test-bucket"
        assert f"concurrent_key_{i}" == call_args["Key"]


@pytest.mark.asyncio
async def test_s3_cache_async_error_handling(mock_s3_dependencies):
    """Test that async methods handle errors gracefully"""
    cache = S3Cache("test-bucket")
    
    # Test async_set_cache error handling
    cache.s3_client.put_object.side_effect = Exception("S3 Error")
    
    # Should not raise exception, just log it
    await cache.async_set_cache("error_key", {"data": "value"})
    
    # Test async_get_cache error handling
    cache.s3_client.get_object.side_effect = Exception("S3 Error")
    
    result = await cache.async_get_cache("error_key")
    assert result is None


@pytest.mark.asyncio
async def test_s3_cache_async_with_key_prefix(mock_s3_dependencies):
    """Test async operations with s3_path prefix"""
    cache = S3Cache("test-bucket", s3_path="cache/data")
    test_value = {"key": "value"}
    
    await cache.async_set_cache("namespace:key", test_value)
    
    cache.s3_client.put_object.assert_called_once()
    call_args = cache.s3_client.put_object.call_args
    
    # Should transform key with prefix and colon replacement
    assert call_args[1]["Key"] == "cache/data/namespace/key"


def test_s3_cache_supports_async():
    """Test that S3Cache now supports async operations"""
    from litellm.caching.caching import Cache, LiteLLMCacheType
    
    cache = Cache(type=LiteLLMCacheType.S3, s3_bucket_name="test-bucket")
    
    # Should now return True for async support
    assert cache._supports_async() is True


@pytest.mark.asyncio
async def test_s3_cache_async_disconnect(mock_s3_dependencies):
    """Test async disconnect method"""
    cache = S3Cache("test-bucket")
    
    # Should not raise any exceptions
    await cache.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
