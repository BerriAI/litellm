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

    with (
        patch(
            "litellm.caching.gcs_cache._get_httpx_client", return_value=mock_sync_client
        ),
        patch(
            "litellm.caching.gcs_cache.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch(
            "litellm.caching.gcs_cache.GCSBucketBase.sync_construct_request_headers",
            return_value={},
        ),
    ):
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
    mock_gcs_dependencies["async_client"].get.return_value.text = '{"foo": "bar"}'
    result = await cache.async_get_cache("key")
    assert result == {"foo": "bar"}


@pytest.mark.asyncio
async def test_gcs_cache_async_get_encodes_object_name_in_path(mock_gcs_dependencies):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/30377

    When gcs_path is set, the object name contains a '/' (e.g. "my_cache/<hash>").
    The GCS JSON API requires the object name in the GET path to be URL-encoded,
    so the '/' must be sent as '%2F'. Otherwise GCS returns 404 and every read
    silently misses.
    """
    cache = GCSCache(bucket_name="test-bucket", gcs_path="my_cache/")

    mock_gcs_dependencies["async_client"].get.return_value.status_code = 200
    mock_gcs_dependencies["async_client"].get.return_value.text = '{"foo": "bar"}'

    result = await cache.async_get_cache("abc123")
    assert result == {"foo": "bar"}

    called_url = mock_gcs_dependencies["async_client"].get.call_args.kwargs["url"]
    # The slash from gcs_path must be percent-encoded in the path segment.
    assert "/o/my_cache%2Fabc123?alt=media" in called_url
    assert "/o/my_cache/abc123" not in called_url


def test_gcs_cache_get_encodes_object_name_in_path(mock_gcs_dependencies):
    """Sync counterpart of the regression test for issue #30377."""
    cache = GCSCache(bucket_name="test-bucket", gcs_path="my_cache/")

    mock_gcs_dependencies["sync_client"].get.return_value.status_code = 200
    mock_gcs_dependencies["sync_client"].get.return_value.text = '{"foo": "bar"}'

    result = cache.get_cache("abc123")
    assert result == {"foo": "bar"}

    called_url = mock_gcs_dependencies["sync_client"].get.call_args.kwargs["url"]
    assert "/o/my_cache%2Fabc123?alt=media" in called_url
    assert "/o/my_cache/abc123" not in called_url


def test_gcs_cache_set_encodes_object_name_in_query(mock_gcs_dependencies):
    """
    The set path uses the object name as a query parameter. Encoding it keeps
    both sides symmetric so the key written matches the key read back.
    """
    cache = GCSCache(bucket_name="test-bucket", gcs_path="my_cache/")
    cache.set_cache("abc123", {"foo": "bar"})

    called_url = mock_gcs_dependencies["sync_client"].post.call_args.kwargs["url"]
    assert "name=my_cache%2Fabc123" in called_url


@pytest.mark.asyncio
async def test_gcs_cache_async_set_encodes_object_name_in_query(mock_gcs_dependencies):
    """Async counterpart of test_gcs_cache_set_encodes_object_name_in_query."""
    cache = GCSCache(bucket_name="test-bucket", gcs_path="my_cache/")
    await cache.async_set_cache("abc123", {"foo": "bar"})

    called_url = mock_gcs_dependencies["async_client"].post.call_args.kwargs["url"]
    assert "name=my_cache%2Fabc123" in called_url
