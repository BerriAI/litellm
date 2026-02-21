"""
Regression tests for Redis connection pool leak fixes (RC1-RC5).

Tests are pure unit tests — no Redis server required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as async_redis

from litellm._redis import get_redis_async_client, get_redis_connection_pool
from litellm.caching.llm_caching_handler import LLMClientCache


def test_url_config_uses_passed_pool():
    """When connection_pool is provided with a URL config, the client
    should use the passed pool — not create a new one via from_url()."""
    mock_pool = MagicMock()

    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {"url": "redis://localhost:6379/0"}

        client = get_redis_async_client(connection_pool=mock_pool)

    assert client.connection_pool is mock_pool


def test_url_config_falls_back_to_from_url_without_pool():
    """When no connection_pool is provided, URL config should still
    use from_url() as before."""
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {"url": "redis://localhost:6379/0"}

        client = get_redis_async_client()

    # from_url creates its own pool — just verify it's not None
    assert client.connection_pool is not None


def test_max_connections_url_config(monkeypatch):
    """max_connections should be respected when using URL-based config."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "10")

    pool = get_redis_connection_pool()

    assert pool.max_connections == 10


def test_max_connections_url_config_string_value(monkeypatch):
    """max_connections provided as a string (from env var) should be
    cast to int."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "25")

    pool = get_redis_connection_pool()

    assert pool.max_connections == 25


def test_max_connections_url_config_invalid_value():
    """Invalid max_connections should be silently ignored, falling back
    to the pool default (50 for BlockingConnectionPool)."""
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379/0",
            "max_connections": "not_a_number",
        }

        pool = get_redis_connection_pool()

    # BlockingConnectionPool default is 50
    assert pool.max_connections == 50


def test_max_connections_url_config_none_value():
    """max_connections=None should be silently ignored."""
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379/0",
            "max_connections": None,
        }

        pool = get_redis_connection_pool()

    assert pool.max_connections == 50


def _make_redis_cache():
    """Create a RedisCache with all external I/O mocked out."""
    mock_sync_client = MagicMock()
    mock_async_pool = AsyncMock()
    patches = [
        patch("litellm._redis.get_redis_client", return_value=mock_sync_client),
        patch("litellm._redis.get_redis_connection_pool", return_value=mock_async_pool),
        patch("litellm.caching.redis_cache.RedisCache._setup_health_pings"),
    ]
    for p in patches:
        p.start()

    from litellm.caching.redis_cache import RedisCache
    cache = RedisCache(host="localhost", port=6379)

    for p in patches:
        p.stop()

    return cache, mock_sync_client, mock_async_pool


@pytest.mark.asyncio
async def test_disconnect_closes_sync_client():
    """disconnect() should close both the async pool and the sync client."""
    cache, mock_sync_client, mock_async_pool = _make_redis_cache()
    await cache.disconnect()

    mock_async_pool.disconnect.assert_awaited_once_with(inuse_connections=True)
    mock_sync_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_idempotent():
    """Calling disconnect() twice should not raise."""
    cache, mock_sync_client, mock_async_pool = _make_redis_cache()
    mock_sync_client.close.side_effect = [None, RuntimeError("already closed")]

    await cache.disconnect()
    await cache.disconnect()  # should not raise


@pytest.mark.asyncio
async def test_eviction_calls_aclose():
    """When an async client is evicted from LLMClientCache, its aclose()
    should be scheduled via create_task."""
    cache = LLMClientCache(max_size_in_memory=2, default_ttl=600)

    client = AsyncMock()
    client.aclose = AsyncMock()

    cache.set_cache(key="client-0", value=client)
    cache.set_cache(key="filler", value="x")
    # Third insert triggers eviction of client-0
    cache.set_cache(key="trigger", value="y")

    # Let the scheduled task run
    await asyncio.sleep(0.05)

    assert client.aclose.await_count > 0


@pytest.mark.asyncio
async def test_eviction_non_closeable_safe():
    """Evicting plain values (strings, dicts, ints) should not crash."""
    cache = LLMClientCache(max_size_in_memory=2, default_ttl=600)

    cache.set_cache(key="str-val", value="hello")
    cache.set_cache(key="dict-val", value={"foo": "bar"})
    # This evicts "str-val" — should not raise
    cache.set_cache(key="int-val", value=42)

    await asyncio.sleep(0.05)

    # If we got here without exception, the test passes
    assert cache.get_cache(key="int-val") == 42
