from importlib import import_module
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


from litellm.caching.gcs_cache import GCSCache


@pytest.fixture
def mock_gcs_dependencies():
    """Mock httpx clients and GCS auth"""
    mock_sync_client = MagicMock()
    mock_async_client = AsyncMock()

    with (
        patch.object(import_module("litellm.caching.gcs_cache"), "get_httpx_client", return_value=mock_sync_client),
        patch.object(
            import_module("litellm.caching.gcs_cache"),
            "get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch.object(
            import_module("litellm.caching.gcs_cache").GCSBucketBase,
            "sync_construct_request_headers",
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


@pytest.mark.asyncio
async def test_gcs_cache_completion_hit_and_param_sensitivity(mock_gcs_dependencies):
    """Same params hit the injected GCS cache client; different params do not."""
    import asyncio

    import litellm
    from litellm.caching import Cache

    store = {}

    import json as _json
    from urllib.parse import parse_qs, unquote, urlparse

    def _store(url, data=None, json=None, content=None, **kw):
        object_name = parse_qs(urlparse(url).query).get("name", [None])[0]
        if object_name is not None:
            store[unquote(object_name)] = data if isinstance(data, (str, bytes)) else json
        resp = AsyncMock()
        resp.status_code = 200
        resp.text = "{}"
        return resp

    def _get(url, **kw):
        object_name = unquote(url.split("/o/")[-1].split("?")[0])
        resp = AsyncMock()
        if object_name in store:
            stored = store[object_name]
            resp.status_code = 200
            resp.text = stored if isinstance(stored, str) else _json.dumps(stored)
        else:
            resp.status_code = 404
            resp.text = "not found"
        return resp

    mock_gcs_dependencies["async_client"].post.side_effect = _store
    mock_gcs_dependencies["async_client"].get.side_effect = _get
    mock_gcs_dependencies["sync_client"].post.side_effect = _store
    mock_gcs_dependencies["sync_client"].get.side_effect = _get

    messages = [{"role": "user", "content": "write a one sentence poem about: gcs-cache-unit-key"}]
    litellm.cache = Cache(type="gcs", gcs_bucket_name="test-bucket")
    try:
        response1 = await litellm.acompletion(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is so great!",
        )
        deadline = asyncio.get_event_loop().time() + 10
        while not store and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        assert store
        response2 = await litellm.acompletion(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            max_tokens=20,
            mock_response="This number is great!",
        )
        assert response1.choices[0].message.content == response2.choices[0].message.content

        response3 = await litellm.acompletion(
            "gpt-3.5-turbo",
            messages=messages,
            caching=True,
            temperature=0.5,
            mock_response="This number is awful!",
        )
        assert response1.choices[0].message.content != response3.choices[0].message.content
    finally:
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []


