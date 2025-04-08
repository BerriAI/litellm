import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from litellm.caching.s3_cache import S3Cache
import botocore

@pytest.fixture
def bucket_name():
    return "test-bucket"

@pytest.fixture
def s3_path():
    return "test/path"

@pytest.fixture
def mock_s3_client():
    return AsyncMock()

@pytest.fixture
async def s3_cache(bucket_name, s3_path):
    with patch('aioboto3.Session'):
        cache = S3Cache(
            s3_bucket_name=bucket_name,
            s3_path=s3_path
        )
        cache._s3_client = AsyncMock()
        yield cache
        await cache.disconnect()

@pytest.mark.asyncio
async def test_s3_client_initialization():
    """Test S3 client initialization"""
    mock_client = AsyncMock()
    mock_session = MagicMock()
    mock_session.client.return_value.__aenter__.return_value = mock_client
    
    with patch('aioboto3.Session', return_value=mock_session):
        cache = S3Cache(s3_bucket_name="test-bucket")
        client = await cache.s3_client
        
    assert client == mock_client
    mock_session.client.assert_called_once()

@pytest.mark.asyncio
async def test_async_set_cache_with_ttl(s3_cache, bucket_name, s3_path):
    """Test setting cache with TTL"""
    s3_cache = await s3_cache.__anext__()
    key = "test_key"
    value = {"data": "test_value"}
    ttl = 3600

    await s3_cache.async_set_cache(key, value, ttl=ttl)

    expected_key = f"{s3_path}/{key}"
    s3_cache._s3_client.put_object.assert_called_once()
    call_kwargs = s3_cache._s3_client.put_object.call_args.kwargs
    
    assert call_kwargs['Bucket'] == bucket_name
    assert call_kwargs['Key'] == expected_key
    assert call_kwargs['Body'] == json.dumps(value)
    assert f'max-age={ttl}' in call_kwargs['CacheControl']

@pytest.mark.asyncio
async def test_async_set_cache_without_ttl(s3_cache, bucket_name, s3_path):
    """Test setting cache without TTL"""
    s3_cache = await s3_cache.__anext__()
    key = "test_key"
    value = {"data": "test_value"}

    await s3_cache.async_set_cache(key, value)

    expected_key = f"{s3_path}/{key}"
    s3_cache._s3_client.put_object.assert_called_once()
    call_kwargs = s3_cache._s3_client.put_object.call_args.kwargs
    
    assert call_kwargs['Bucket'] == bucket_name
    assert call_kwargs['Key'] == expected_key
    assert call_kwargs['Body'] == json.dumps(value)
    assert 'max-age=31536000' in call_kwargs['CacheControl']

@pytest.mark.asyncio
async def test_async_get_cache_success(s3_cache, bucket_name, s3_path):
    """Test getting cache successfully"""
    s3_cache = await s3_cache.__anext__()
    key = "test_key"
    expected_value = {"data": "test_value"}
    
    mock_body = AsyncMock()
    mock_body.read.return_value = json.dumps(expected_value).encode('utf-8')
    
    s3_cache._s3_client.get_object.return_value = {
        "Body": mock_body
    }

    result = await s3_cache.async_get_cache(key)
    
    expected_key = f"{s3_path}/{key}"
    s3_cache._s3_client.get_object.assert_called_once_with(
        Bucket=bucket_name,
        Key=expected_key
    )
    assert result == expected_value

@pytest.mark.asyncio
async def test_async_get_cache_no_key(s3_cache):
    """Test getting cache with non-existent key"""
    s3_cache = await s3_cache.__anext__()
    key = "nonexistent_key"
    error = botocore.exceptions.ClientError(
        error_response={"Error": {"Code": "NoSuchKey"}},
        operation_name="GetObject"
    )
    s3_cache._s3_client.get_object.side_effect = error

    result = await s3_cache.async_get_cache(key)
    assert result is None

@pytest.mark.asyncio
async def test_async_get_cache_error(s3_cache):
    """Test getting cache with unexpected error"""
    s3_cache = await s3_cache.__anext__()
    key = "error_key"
    s3_cache._s3_client.get_object.side_effect = Exception("Unexpected error")

    result = await s3_cache.async_get_cache(key)
    assert result is None


@pytest.mark.asyncio
async def test_disconnect(s3_cache):
    """Test disconnecting from S3"""
    s3_cache = await s3_cache.__anext__()
    tmp_s3_client = s3_cache._s3_client
    await s3_cache.disconnect()
    tmp_s3_client.__aexit__.assert_called_once()
    assert s3_cache._s3_client is None

@pytest.mark.asyncio
async def test_sync_methods_raise_error(s3_cache):
    """Test that sync methods raise NotImplementedError"""
    s3_cache = await s3_cache.__anext__()
    with pytest.raises(NotImplementedError):
        s3_cache.set_cache("key", "value")
        
    with pytest.raises(NotImplementedError):
        s3_cache.get_cache("key")
