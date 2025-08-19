import os
import sys
from unittest.mock import MagicMock, patch
import json
import datetime

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


def test_s3_cache_set_cache(mock_s3_dependencies):
    """Test basic set_cache functionality"""
    cache = S3Cache("test-bucket")
    test_value = {"key": "value", "number": 42}
    
    cache.set_cache("test_key", test_value)
    
    cache.s3_client.put_object.assert_called_once()
    call_args = cache.s3_client.put_object.call_args
    
    assert call_args[1]["Bucket"] == "test-bucket"
    assert call_args[1]["Key"] == "test_key"
    assert call_args[1]["Body"] == json.dumps(test_value)
    assert call_args[1]["ContentType"] == "application/json"
    assert call_args[1]["ContentLanguage"] == "en"
    assert call_args[1]["ContentDisposition"] == 'inline; filename="test_key.json"'


def test_s3_cache_set_cache_with_ttl(mock_s3_dependencies):
    """Test set_cache with TTL functionality"""
    cache = S3Cache("test-bucket")
    test_value = {"key": "value"}
    ttl = datetime.timedelta(seconds=3600)  # 1 hour

    cache.set_cache("test_key", test_value, ttl=ttl)

    cache.s3_client.put_object.assert_called_once()
    call_args = cache.s3_client.put_object.call_args

    assert "Expires" in call_args[1]
    assert "CacheControl" in call_args[1]
    assert "max-age=1:00:00" in call_args[1]["CacheControl"]


def test_s3_cache_get_cache(mock_s3_dependencies):
    """Test basic get_cache functionality"""
    cache = S3Cache("test-bucket")
    
    mock_response = {
        "Body": MagicMock()
    }
    mock_response["Body"].read.return_value = b'{"key": "value", "number": 42}'
    cache.s3_client.get_object.return_value = mock_response
    
    result = cache.get_cache("test_key")
    
    cache.s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", 
        Key="test_key"
    )
    
    assert result == {"key": "value", "number": 42}


def test_s3_cache_get_cache_not_found(mock_s3_dependencies):
    """Test get_cache when key is not found"""
    import botocore.exceptions
    
    cache = S3Cache("test-bucket")
    
    error_response = {"Error": {"Code": "NoSuchKey"}}
    cache.s3_client.get_object.side_effect = botocore.exceptions.ClientError(
        error_response, "GetObject"
    )
    
    result = cache.get_cache("nonexistent_key")
    
    cache.s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", 
        Key="nonexistent_key"
    )
    assert result is None


def test_s3_key_transformation():
    """Test the _to_s3_key method for key transformation"""
    cache = S3Cache("test-bucket")
    
    # Test basic key transformation (colon to slash)
    result = cache._to_s3_key("user:123:session:456")
    assert result == "user/123/session/456"
    
    # Test with s3_path prefix
    cache_with_prefix = S3Cache("test-bucket", s3_path="cache/data")
    result = cache_with_prefix._to_s3_key("namespace:key")
    assert result == "cache/data/namespace/key"
    
    # Test with s3_path that has trailing slash
    cache_with_slash = S3Cache("test-bucket", s3_path="cache/data/")
    result = cache_with_slash._to_s3_key("namespace:key")
    assert result == "cache/data/namespace/key"


def test_s3_cache_initialization():
    """Test S3Cache initialization with various parameters"""
    # Test basic initialization
    cache = S3Cache("test-bucket")
    assert cache.bucket_name == "test-bucket"
    assert cache.key_prefix == ""
    
    # Test with s3_path
    cache_with_path = S3Cache("test-bucket", s3_path="my/cache/path")
    assert cache_with_path.key_prefix == "my/cache/path/"