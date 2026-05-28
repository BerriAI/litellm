"""
LIT-3263 — Redis async client cache TTL pinning.

The async Redis client cached in litellm.in_memory_llm_clients_cache must
not be evicted while the proxy is running. The default LLMClientCache TTL is
600s, which on sustained high-TPS deployments forces a fresh
BlockingConnectionPool every 10 minutes — the cold pool then triggers a
wave of concurrent connect attempts that exhaust the pool wait queue,
surfacing as Timeout connecting to server errors from
RedisCache.async_increment_pipeline.

These tests pin behaviour:

  * set_cache is called with an explicit, long TTL (matching the
    constant).
  * Cache hits short-circuit — no fresh BlockingConnectionPool is built
    on subsequent calls.
  * The constant is configurable via the REDIS_ASYNC_CLIENT_CACHE_TTL
    env var.
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm.constants as litellm_constants
from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_ASYNC_CLIENT_CACHE_TTL


@pytest.fixture
def redis_no_ping():
    """Suppress the background ping task during construction."""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.fixture
def redis_cache_instance(redis_no_ping):
    """Build a RedisCache without actually opening a sync connection."""
    with (
        patch("litellm._redis.get_redis_client") as gc,
        patch("litellm._redis.get_redis_connection_pool") as gp,
    ):
        gc.return_value = MagicMock(name="sync_redis_client")
        gp.return_value = MagicMock(name="sync_pool")
        yield RedisCache(host="lit-3263-host", port=6379)


def test_redis_async_client_cache_ttl_constant_default():
    """The TTL constant defaults to 86400 (1 day) — long enough that the
    pool is never recycled by the default LLMClientCache eviction."""
    assert REDIS_ASYNC_CLIENT_CACHE_TTL == 86400


def test_redis_async_client_cache_ttl_env_override(monkeypatch):
    """The TTL constant must be tunable via env var."""
    monkeypatch.setenv("REDIS_ASYNC_CLIENT_CACHE_TTL", "172800")
    try:
        importlib.reload(litellm_constants)
        assert litellm_constants.REDIS_ASYNC_CLIENT_CACHE_TTL == 172800
    finally:
        monkeypatch.delenv("REDIS_ASYNC_CLIENT_CACHE_TTL", raising=False)
        importlib.reload(litellm_constants)


def test_init_async_client_passes_ttl_to_in_memory_cache(redis_cache_instance):
    """init_async_client must set the cached client with the pinned TTL,
    not the LLMClientCache default of 600s."""
    cache = redis_cache_instance

    fake_async_client = MagicMock(name="fake_async_redis_client")
    fake_pool = MagicMock(name="fake_pool")

    with (
        patch("litellm._redis.get_redis_connection_pool", return_value=fake_pool),
        patch("litellm._redis.get_redis_async_client", return_value=fake_async_client),
        patch("litellm.in_memory_llm_clients_cache") as fake_in_memory,
    ):
        fake_in_memory.get_cache.return_value = None

        result = cache.init_async_client()

    assert result is fake_async_client
    # Pool was rebuilt for the current event loop and stored on the instance.
    assert cache.async_redis_conn_pool is fake_pool
    fake_in_memory.set_cache.assert_called_once()
    kwargs = fake_in_memory.set_cache.call_args.kwargs
    assert kwargs["value"] is fake_async_client
    assert kwargs["ttl"] == REDIS_ASYNC_CLIENT_CACHE_TTL
    assert kwargs["ttl"] >= 3600, (
        "TTL must be at least 1 hour — otherwise the connection pool gets "
        "recycled mid-traffic (LIT-3263)."
    )


def test_init_async_client_reuses_cached_client_no_pool_churn(redis_cache_instance):
    """Once the cached client is populated, init_async_client must reuse it
    and must NOT call get_redis_connection_pool again — building a fresh
    BlockingConnectionPool every call is the exact behaviour LIT-3263 is
    fixing."""
    cache = redis_cache_instance

    fake_async_client = MagicMock(name="cached_client")

    with (
        patch("litellm._redis.get_redis_connection_pool") as fake_get_pool,
        patch("litellm._redis.get_redis_async_client") as fake_get_client,
        patch("litellm.in_memory_llm_clients_cache") as fake_in_memory,
    ):
        fake_in_memory.get_cache.return_value = fake_async_client

        result = cache.init_async_client()

    assert result is fake_async_client
    fake_get_pool.assert_not_called()
    fake_get_client.assert_not_called()


def test_init_async_client_creates_fresh_pool_on_cache_miss(redis_cache_instance):
    """If the cache misses (because TTL did expire, the entry was explicitly
    evicted, or this is the first call) we still need to construct a fresh
    pool. Both cache-miss reinits must pin the same long TTL."""
    cache = redis_cache_instance

    first_client, second_client = MagicMock(name="c1"), MagicMock(name="c2")
    first_pool, second_pool = MagicMock(name="p1"), MagicMock(name="p2")

    with (
        patch(
            "litellm._redis.get_redis_connection_pool",
            side_effect=[first_pool, second_pool],
        ),
        patch(
            "litellm._redis.get_redis_async_client",
            side_effect=[first_client, second_client],
        ),
        patch("litellm.in_memory_llm_clients_cache") as fake_in_memory,
    ):
        fake_in_memory.get_cache.return_value = None
        c1 = cache.init_async_client()
        fake_in_memory.get_cache.return_value = None
        c2 = cache.init_async_client()

    assert c1 is first_client
    assert c2 is second_client
    assert fake_in_memory.set_cache.call_count == 2
    for call in fake_in_memory.set_cache.call_args_list:
        assert call.kwargs["ttl"] == REDIS_ASYNC_CLIENT_CACHE_TTL
