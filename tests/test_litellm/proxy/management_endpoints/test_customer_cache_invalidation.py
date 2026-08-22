"""
Unit tests for end-user cache invalidation on customer CUD (issue #31838).

Budget enforcement reads the end-user from ``user_api_key_cache`` under
``end_user_id:{id}`` (TTL 300s). The /customer/new, /customer/update and
/customer/delete endpoints must drop that key so a create/update/delete is
reflected immediately instead of serving a stale budget for up to the TTL.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.management_endpoints.customer_endpoints import (
    _invalidate_end_user_cache,
)


@pytest.fixture
def patched_cache(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    cache = MagicMock()
    cache.async_delete_cache = AsyncMock()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache, raising=False)
    return cache


def _deleted_keys(cache):
    return [call.kwargs["key"] for call in cache.async_delete_cache.await_args_list]


@pytest.mark.asyncio
async def test_invalidates_each_user_id(patched_cache):
    await _invalidate_end_user_cache(["alice", "bob"])
    assert _deleted_keys(patched_cache) == ["end_user_id:alice", "end_user_id:bob"]


@pytest.mark.asyncio
async def test_empty_list_is_noop(patched_cache):
    await _invalidate_end_user_cache([])
    patched_cache.async_delete_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_empty_user_ids(patched_cache):
    await _invalidate_end_user_cache(["alice", "", None])
    assert _deleted_keys(patched_cache) == ["end_user_id:alice"]


@pytest.mark.asyncio
async def test_swallows_cache_errors(patched_cache):
    patched_cache.async_delete_cache.side_effect = RuntimeError("redis down")
    # Must not raise — cache invalidation failures should not break the endpoint.
    await _invalidate_end_user_cache(["alice"])
