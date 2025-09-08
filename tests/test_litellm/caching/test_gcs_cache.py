import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.gcs_cache import GCSCache


@pytest.fixture
def mock_gcs_dependencies():
    """Mock httpx clients and GCS auth"""
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()

    with patch("litellm.caching.gcs_cache._get_httpx_client", return_value=mock_sync_client), \
         patch("litellm.caching.gcs_cache.get_async_httpx_client", return_value=mock_async_client), \
         patch("litellm.caching.gcs_cache.GCSBucketBase.sync_construct_request_headers", return_value={}):
        yield {
            "sync_client": mock_sync_client,
            "async_client": mock_async_client,
        }


@pytest.mark.asyncio
async def test_gcs_cache_async_set_and_get(mock_gcs_dependencies):
    cache = GCSCache(bucket_name="test-bucket")
    await cache.async_set_cache("key", {"foo": "bar"})
    mock_gcs_dependencies["async_client"].post.assert_called_once()

    mock_gcs_dependencies["async_client"].get.return_value.status_code = 200
    mock_gcs_dependencies["async_client"].get.return_value.text = "{\"foo\": \"bar\"}"
    result = await cache.async_get_cache("key")
    assert result == {"foo": "bar"}
