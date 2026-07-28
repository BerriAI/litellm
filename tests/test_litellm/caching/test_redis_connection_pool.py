"""
Regression tests for Redis connection pool leak fixes (RC1-RC5).

Tests are pure unit tests — no Redis server required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as async_redis

import litellm
from litellm._redis import get_redis_async_client, get_redis_connection_pool


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


def _make_url_redis_cache():
    """
    RedisCache over a URL config with a real (lazy) connection pool.

    Unlike _make_redis_cache above, the pool is NOT mocked — these tests assert on
    pool *identity*, so a single shared mock would hide the behaviour under test.
    BlockingConnectionPool opens no sockets until a command runs, so this stays
    server-free.
    """
    from litellm.caching.redis_cache import RedisCache

    with patch("litellm._redis.get_redis_client", return_value=MagicMock()), patch(
        "litellm.caching.redis_cache.RedisCache._setup_health_pings"
    ):
        return RedisCache(url="redis://localhost:6379/0")


@pytest.fixture
def clean_client_cache():
    """Isolate the process-wide async client cache from other tests."""
    litellm.in_memory_llm_clients_cache.cache_dict.clear()
    litellm.in_memory_llm_clients_cache.ttl_dict.clear()
    yield litellm.in_memory_llm_clients_cache
    litellm.in_memory_llm_clients_cache.cache_dict.clear()
    litellm.in_memory_llm_clients_cache.ttl_dict.clear()


@pytest.mark.asyncio
async def test_client_cache_expiry_reuses_connection_pool(clean_client_cache):
    """
    The cached async client expires on a TTL (default 600s). Building a fresh
    connection pool for the replacement client abandons a pool of established
    connections while the new pool opens its own, so a single RedisCache
    transiently holds ~2x max_connections against the server every rotation —
    which is how a proxy walks into "max number of clients reached".

    The replacement client must reuse the existing pool.
    """
    cache = _make_url_redis_cache()

    first_client = cache.init_async_client()
    pool = cache.async_redis_conn_pool
    assert pool is not None

    # what TTL expiry does to the entry, on the same event loop
    clean_client_cache.cache_dict.clear()
    clean_client_cache.ttl_dict.clear()

    second_client = cache.init_async_client()

    assert second_client is not first_client, "expected a new client once the cached entry expired"
    assert cache.async_redis_conn_pool is pool, "connection pool was rebuilt instead of reused"
    assert second_client.connection_pool is pool, "replacement client did not attach to the existing pool"


@pytest.mark.asyncio
async def test_shared_pool_survives_outgoing_client_close(clean_client_cache):
    """
    Reusing one pool across clients is only safe if closing the outgoing client
    leaves the pool alone. redis-py guarantees this by setting
    auto_close_connection_pool=False whenever a pool is passed in; pin it, because
    losing it would let an expiring client tear the pool out from under live traffic.
    """
    cache = _make_url_redis_cache()

    first_client = cache.init_async_client()
    pool = cache.async_redis_conn_pool

    assert first_client.auto_close_connection_pool is False

    clean_client_cache.cache_dict.clear()
    clean_client_cache.ttl_dict.clear()
    second_client = cache.init_async_client()

    with patch.object(pool, "disconnect", new=AsyncMock()) as pool_disconnect:
        await first_client.aclose()

    pool_disconnect.assert_not_awaited()
    assert second_client.connection_pool is pool


def test_new_event_loop_gets_its_own_connection_pool(clean_client_cache):
    """
    A pool's connections are bound to the loop that created them, so a different
    event loop must never be handed the previous loop's pool. Guards the reuse
    above from over-reaching.
    """
    cache = _make_url_redis_cache()
    loops = []  # held so a closed loop's id() cannot be recycled by the next one
    pools = []

    async def init_in_current_loop():
        cache.init_async_client()
        return cache.async_redis_conn_pool

    for _ in range(3):
        loop = asyncio.new_event_loop()
        loops.append(loop)
        pools.append(loop.run_until_complete(init_in_current_loop()))

    for loop in loops:
        loop.close()

    assert len({id(p) for p in pools}) == 3, "a connection pool was shared across event loops"
