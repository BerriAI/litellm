"""
Test timedelta serialization in cache implementations.

This test ensures that timedelta objects can be properly serialized
to JSON in all cache implementations without causing serialization errors.
"""

import json
import unittest
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.caching.s3_cache import S3Cache
from litellm.caching.gcs_cache import GCSCache
from litellm.caching.azure_blob_cache import AzureBlobCache
from litellm.caching.json_utils import TimedeltaJSONEncoder


class TestTimedeltaJSONEncoder(unittest.TestCase):
    """Test the TimedeltaJSONEncoder class."""

    def test_timedelta_serialization(self):
        """Test that timedelta objects are properly serialized to seconds."""
        test_data = {
            "latency": [timedelta(seconds=1.5), timedelta(milliseconds=500)],
            "time_to_first_token": [timedelta(milliseconds=200)],
            "normal_value": 42,
            "nested": {
                "response_time": timedelta(seconds=2.3),
                "other_data": "test"
            }
        }
        
        # This should not raise an exception
        serialized = json.dumps(test_data, cls=TimedeltaJSONEncoder)
        deserialized = json.loads(serialized)
        
        # Verify timedelta objects were converted to seconds
        assert deserialized["latency"] == [1.5, 0.5]
        assert deserialized["time_to_first_token"] == [0.2]
        assert deserialized["normal_value"] == 42
        assert deserialized["nested"]["response_time"] == 2.3
        assert deserialized["nested"]["other_data"] == "test"

    def test_reproduces_original_error(self):
        """Test that reproduces the original error before the fix."""
        test_data = {
            "latency": [timedelta(seconds=1.5)],
            "normal_value": 42
        }
        
        # This would have raised: "Object of type timedelta is not JSON serializable"
        with pytest.raises(TypeError, match="timedelta is not JSON serializable"):
            json.dumps(test_data)
        
        # But with our custom encoder, it should work
        serialized = json.dumps(test_data, cls=TimedeltaJSONEncoder)
        deserialized = json.loads(serialized)
        assert deserialized["latency"] == [1.5]
        assert deserialized["normal_value"] == 42


class TestRedisCacheTimedeltaHandling:
    """Test Redis cache handling of timedelta objects."""

    def test_redis_cache_sync_with_timedelta(self):
        """Test that RedisCache can handle timedelta objects without errors."""
        # Mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.set = Mock()
        mock_redis_client.get = Mock(return_value=None)
        mock_redis_client.ping = Mock(return_value=True)
        
        # Create RedisCache instance with mocked client
        with patch('litellm._redis.get_redis_client', return_value=mock_redis_client):
            cache = RedisCache(host="localhost", port=6379)
            
            # Test data with timedelta objects (similar to what's stored in latency routing)
            test_data = {
                "deployment_1": {
                    "latency": [timedelta(seconds=1.2), timedelta(milliseconds=800)],
                    "time_to_first_token": [timedelta(milliseconds=150)],
                    "2024-01-01-10-30": {
                        "tpm": 1000,
                        "rpm": 10
                    }
                }
            }
            
            # This should not raise an exception
            cache.set_cache("test_key", test_data)
            
            # Verify that set was called with JSON-serialized data
            mock_redis_client.set.assert_called_once()
            call_args = mock_redis_client.set.call_args
            serialized_value = call_args[1]['value']
            
            # Verify the serialized value is valid JSON
            deserialized = json.loads(serialized_value)
            assert deserialized["deployment_1"]["latency"] == [1.2, 0.8]
            assert deserialized["deployment_1"]["time_to_first_token"] == [0.15]

    @pytest.mark.asyncio
    async def test_redis_cache_async_with_timedelta(self):
        """Test that async RedisCache can handle timedelta objects without errors."""
        # Mock async Redis client
        mock_async_redis_client = Mock()
        mock_async_redis_client.set = Mock(return_value=True)
        mock_async_redis_client.get = Mock(return_value=None)
        mock_async_redis_client.ping = Mock(return_value=True)
        
        # Create RedisCache instance with mocked client
        with patch('litellm._redis.get_redis_async_client', return_value=mock_async_redis_client):
            cache = RedisCache(host="localhost", port=6379)
            
            # Test data with timedelta objects
            test_data = {
                "deployment_1": {
                    "latency": [timedelta(seconds=1.2), timedelta(milliseconds=800)],
                    "time_to_first_token": [timedelta(milliseconds=150)],
                    "2024-01-01-10-30": {
                        "tpm": 1000,
                        "rpm": 10
                    }
                }
            }
            
            # This should not raise an exception
            await cache.async_set_cache("test_key", test_data)
            
            # Verify that set was called with JSON-serialized data
            mock_async_redis_client.set.assert_called_once()
            call_args = mock_async_redis_client.set.call_args
            serialized_value = call_args[1]['value']
            
            # Verify the serialized value is valid JSON
            deserialized = json.loads(serialized_value)
            assert deserialized["deployment_1"]["latency"] == [1.2, 0.8]
            assert deserialized["deployment_1"]["time_to_first_token"] == [0.15]


class TestS3CacheTimedeltaHandling:
    """Test S3 cache handling of timedelta objects."""

    def test_s3_cache_with_timedelta(self):
        """Test that S3Cache can handle timedelta objects without errors."""
        # Mock S3 client
        mock_s3_client = Mock()
        mock_s3_client.put_object = Mock()
        
        # Create S3Cache instance with mocked client
        with patch('boto3.client', return_value=mock_s3_client):
            cache = S3Cache(s3_bucket_name="test-bucket")
            
            # Test data with timedelta objects
            test_data = {
                "latency": [timedelta(seconds=1.2), timedelta(milliseconds=800)],
                "time_to_first_token": [timedelta(milliseconds=150)]
            }
            
            # This should not raise an exception
            cache.set_cache("test_key", test_data)
            
            # Verify that put_object was called with JSON-serialized data
            mock_s3_client.put_object.assert_called_once()
            call_args = mock_s3_client.put_object.call_args
            serialized_value = call_args[1]['Body']
            
            # Verify the serialized value is valid JSON
            deserialized = json.loads(serialized_value)
            assert deserialized["latency"] == [1.2, 0.8]
            assert deserialized["time_to_first_token"] == [0.15]


class TestGCSCacheTimedeltaHandling:
    """Test GCS cache handling of timedelta objects."""

    def test_gcs_cache_with_timedelta(self):
        """Test that GCSCache can handle timedelta objects without errors."""
        # Mock HTTP client
        mock_client = Mock()
        mock_client.post = Mock()
        
        # Create GCSCache instance and then mock its client and headers
        cache = GCSCache(bucket_name="test-bucket")
        cache.sync_client = mock_client
        cache._construct_headers = Mock(return_value={"Authorization": "Bearer test-token"})
        
        # Test data with timedelta objects
        test_data = {
            "latency": [timedelta(seconds=1.2), timedelta(milliseconds=800)],
            "time_to_first_token": [timedelta(milliseconds=150)]
        }
        
        # This should not raise an exception
        cache.set_cache("test_key", test_data)
        
        # Verify that post was called with JSON-serialized data
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        serialized_value = call_args[1]['data']
        
        # Verify the serialized value is valid JSON
        deserialized = json.loads(serialized_value)
        assert deserialized["latency"] == [1.2, 0.8]
        assert deserialized["time_to_first_token"] == [0.15]


class TestAzureBlobCacheTimedeltaHandling:
    """Test Azure Blob cache handling of timedelta objects."""

    def test_azure_blob_cache_with_timedelta(self):
        """Test that AzureBlobCache can handle timedelta objects without errors."""
        # Mock Azure Blob client
        mock_container_client = Mock()
        mock_container_client.upload_blob = Mock()
        
        # Mock the BlobServiceClient import
        with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            mock_blob_service.return_value.get_container_client.return_value = mock_container_client
            
            # Create AzureBlobCache instance with mocked client
            cache = AzureBlobCache(account_url="https://test.blob.core.windows.net", container="test-container")
            
            # Test data with timedelta objects
            test_data = {
                "latency": [timedelta(seconds=1.2), timedelta(milliseconds=800)],
                "time_to_first_token": [timedelta(milliseconds=150)]
            }
            
            # This should not raise an exception
            cache.set_cache("test_key", test_data)
            
            # Verify that upload_blob was called with JSON-serialized data
            mock_container_client.upload_blob.assert_called_once()
            call_args = mock_container_client.upload_blob.call_args
            serialized_value = call_args[0][1]  # Second positional argument
            
            # Verify the serialized value is valid JSON
            deserialized = json.loads(serialized_value)
            assert deserialized["latency"] == [1.2, 0.8]
            assert deserialized["time_to_first_token"] == [0.15]
