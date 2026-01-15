import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.azure_blob_cache import AzureBlobCache


@pytest.fixture
def mock_azure_dependencies():
    """Mock all Azure dependencies to avoid requiring actual Azure credentials"""
    
    # Create mock container clients that will be assigned to the cache instance
    mock_container_client = MagicMock()
    mock_async_container_client = AsyncMock()
    
    # Mock credentials
    mock_credential = MagicMock()
    mock_async_credential = AsyncMock()
    
    # Create mock blob service clients that return the container clients
    mock_blob_service_client = MagicMock()
    mock_blob_service_client.get_container_client.return_value = mock_container_client
    
    mock_async_blob_service_client = AsyncMock()
    # For AsyncMock, we need to make get_container_client return the mock directly, not a coroutine
    mock_async_blob_service_client.get_container_client = MagicMock(return_value=mock_async_container_client)
    
    # Patch Azure dependencies at their source locations
    with patch("azure.identity.DefaultAzureCredential", return_value=mock_credential), \
         patch("azure.identity.aio.DefaultAzureCredential", return_value=mock_async_credential), \
         patch("azure.storage.blob.BlobServiceClient", return_value=mock_blob_service_client), \
         patch("azure.storage.blob.aio.BlobServiceClient", return_value=mock_async_blob_service_client), \
         patch("azure.core.exceptions.ResourceExistsError"):
        
        yield {
            "container_client": mock_container_client,
            "async_container_client": mock_async_container_client,
            "blob_service_client": mock_blob_service_client,
            "async_blob_service_client": mock_async_blob_service_client,
            "credential": mock_credential,
            "async_credential": mock_async_credential,
        }


@pytest.mark.asyncio
async def test_blob_cache_async_get_cache(mock_azure_dependencies):
    """Test async_get_cache method with mocked Azure dependencies"""
    
    # Create cache instance (this will use the mocked dependencies)
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    # Mock the download_blob response
    mock_blob = AsyncMock()
    mock_blob.readall.return_value = b'{"test_key": "test_value"}'
    
    # Set up the mock for download_blob on the actual container client instance
    cache.async_container_client.download_blob.return_value = mock_blob
    
    # Test successful cache retrieval
    result = await cache.async_get_cache("test_key")
    
    # Verify the call was made correctly
    cache.async_container_client.download_blob.assert_called_once_with("test_key")
    mock_blob.readall.assert_called_once()
    
    # Check the result
    assert result == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_blob_cache_async_get_cache_not_found(mock_azure_dependencies):
    """Test async_get_cache method when blob is not found"""
    
    # Import the exception inside the test to avoid import issues
    from azure.core.exceptions import ResourceNotFoundError
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    # Mock ResourceNotFoundError
    cache.async_container_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")
    
    # Test cache miss
    result = await cache.async_get_cache("nonexistent_key")
    
    # Verify the call was made and result is None
    cache.async_container_client.download_blob.assert_called_once_with("nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_blob_cache_async_set_cache(mock_azure_dependencies):
    """Test async_set_cache method with mocked Azure dependencies"""
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    test_value = {"key": "value", "number": 42}
    
    # Test setting cache
    await cache.async_set_cache("test_key", test_value)
    
    # Verify the call was made correctly
    cache.async_container_client.upload_blob.assert_called_once_with(
        "test_key", 
        '{"key": "value", "number": 42}',
        overwrite=True
    )


def test_blob_cache_sync_get_cache(mock_azure_dependencies):
    """Test sync get_cache method with mocked Azure dependencies"""
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    # Mock the download_blob response
    mock_blob = MagicMock()
    mock_blob.readall.return_value = b'{"sync_key": "sync_value"}'
    
    cache.container_client.download_blob.return_value = mock_blob
    
    # Test successful cache retrieval
    result = cache.get_cache("sync_key")
    
    # Verify the call was made correctly
    cache.container_client.download_blob.assert_called_once_with("sync_key")
    mock_blob.readall.assert_called_once()
    
    # Check the result
    assert result == {"sync_key": "sync_value"}


def test_blob_cache_sync_set_cache(mock_azure_dependencies):
    """Test sync set_cache method with mocked Azure dependencies"""
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    test_value = {"sync_key": "sync_value", "number": 123}
    
    # Test setting cache
    cache.set_cache("sync_test_key", test_value)
    
    # Verify the call was made correctly
    cache.container_client.upload_blob.assert_called_once_with(
        "sync_test_key", 
        '{"sync_key": "sync_value", "number": 123}'
    )


def test_blob_cache_sync_get_cache_not_found(mock_azure_dependencies):
    """Test sync get_cache method when blob is not found"""
    
    from azure.core.exceptions import ResourceNotFoundError
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    # Mock ResourceNotFoundError
    cache.container_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")
    
    # Test cache miss
    result = cache.get_cache("nonexistent_key")
    
    # Verify the call was made and result is None
    cache.container_client.download_blob.assert_called_once_with("nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_blob_cache_async_set_cache_pipeline(mock_azure_dependencies):
    """Test async_set_cache_pipeline method with mocked Azure dependencies"""
    
    cache = AzureBlobCache("https://my-test-host", "test-container")
    
    # Test data for pipeline
    cache_list = [
        ("key1", {"value": "data1"}),
        ("key2", {"value": "data2"}),
        ("key3", {"value": "data3"}),
    ]
    
    # Test pipeline cache setting
    await cache.async_set_cache_pipeline(cache_list)
    
    # Verify all calls were made correctly
    expected_calls = [
        (("key1", '{"value": "data1"}'), {"overwrite": True}),
        (("key2", '{"value": "data2"}'), {"overwrite": True}),
        (("key3", '{"value": "data3"}'), {"overwrite": True}),
    ]
    
    assert cache.async_container_client.upload_blob.call_count == 3
    for expected_call in expected_calls:
        cache.async_container_client.upload_blob.assert_any_call(*expected_call[0], **expected_call[1])
