import asyncio
import os
from contextlib import AsyncExitStack
from datetime import datetime
from uuid import uuid4

import pytest
import pytest_asyncio

import litellm
from litellm.caching.redis_cache import RedisCache
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import _PROXY_MaxParallelRequestsHandler
from litellm.proxy.hooks.parallel_request_limiter_v3 import PARALLEL_REQUEST_SLOT_TTL_SECONDS
from litellm.proxy.utils import InternalUsageCache


@pytest_asyncio.fixture(loop_scope="function")
async def isolated_test_redis(monkeypatch):
    raw_port = os.environ.get("LITELLM_TEST_REDIS_PORT", "")
    if not raw_port.isdecimal() or not 1 <= int(raw_port) <= 65535:
        pytest.fail("Set LITELLM_TEST_REDIS_PORT to an isolated Redis server's loopback port")
    for name in tuple(os.environ):
        if name.startswith("REDIS_"):
            monkeypatch.delenv(name)
    namespace = f"litellm-lua-test-{uuid4().hex}"
    cache = RedisCache(
        host="127.0.0.1",
        port=int(raw_port),
        namespace=namespace,
        client_name=namespace,
        socket_timeout=2,
        socket_connect_timeout=2,
    )
    async with AsyncExitStack() as cleanup:
        cleanup.callback(cache.redis_client.close)
        cleanup.push_async_callback(cache.async_redis_conn_pool.disconnect)
        client = cache.init_async_client()
        cleanup.push_async_callback(cache.async_redis_conn_pool.disconnect)
        cleanup.push_async_callback(client.aclose)
        cleanup.callback(litellm.in_memory_llm_clients_cache.delete_cache, cache._get_async_client_cache_key())
        try:
            await client.ping()
            yield cache
        finally:
            async for key in client.scan_iter(match=f"{namespace}:*"):
                await client.delete(key)


@pytest.mark.asyncio
async def test_concurrent_realtime_releases_update_redis_without_lost_decrement(isolated_test_redis):
    remote = isolated_test_redis
    first_cache, second_cache = DualCache(redis_cache=remote), DualCache(redis_cache=remote)
    first, second = (_PROXY_MaxParallelRequestsHandler(InternalUsageCache(c)) for c in (first_cache, second_cache))
    auth = UserAPIKeyAuth(api_key="concurrent-key", max_parallel_requests=2)
    first_data, second_data = {"model": "test"}, {"model": "test"}
    first.begin_realtime_attachment(first_data)
    second.begin_realtime_attachment(second_data)
    await first.async_pre_call_hook(auth, first_cache, first_data, "_arealtime")
    await second.async_pre_call_hook(auth, second_cache, second_data, "_arealtime")
    key = f"concurrent-key::{datetime.now().strftime('%Y-%m-%d-%H-%M')}::request_count"
    counter = {"current_requests": 2, "current_rpm": 2, "current_tpm": 17}
    await first_cache.async_set_cache(key, counter)
    await second_cache.async_set_cache(key, counter, local_only=True)
    remote.redis_client.pexpire(remote.check_and_fix_namespace(key), 15000)
    await asyncio.gather(
        first.async_release_realtime_attachment(first_data, auth),
        second.async_release_realtime_attachment(second_data, auth),
    )
    expected = {"current_requests": 0, "current_rpm": 2, "current_tpm": 17}
    assert await remote.async_get_cache(key) == expected
    assert 0 < remote.redis_client.pttl(remote.check_and_fix_namespace(key)) <= 15000
    assert await first_cache.async_get_cache(key) == expected
    assert await second_cache.async_get_cache(key) == expected
    await first_cache.async_set_cache("missing", counter, local_only=True)
    await first._release_realtime_counter("missing")
    assert await remote.async_get_cache("missing") is None
    assert await first_cache.async_get_cache("missing", local_only=True) is None


@pytest.mark.asyncio
async def test_realtime_lease_redis_renewal_is_atomic_and_does_not_resurrect(isolated_test_redis):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PARALLEL_RENEW_SCRIPT

    client = isolated_test_redis.init_async_client()
    first_key = isolated_test_redis.check_and_fix_namespace("first")
    second_key = isolated_test_redis.check_and_fix_namespace("second")
    now = (await client.time())[0]
    await client.zadd(first_key, {"owner": now - 10, "other": now})
    await client.zadd(second_key, {"owner": now - PARALLEL_REQUEST_SLOT_TTL_SECONDS})
    renew = client.register_script(PARALLEL_RENEW_SCRIPT)
    assert await renew(keys=[first_key, second_key], args=["owner", PARALLEL_REQUEST_SLOT_TTL_SECONDS]) == [0]
    assert await client.zscore(first_key, "owner") == now - 10
    await client.zadd(second_key, {"owner": now - 10})
    assert await renew(keys=[first_key, second_key], args=["owner", PARALLEL_REQUEST_SLOT_TTL_SECONDS]) == [1]
    assert await client.zscore(first_key, "owner") >= now
    assert await client.ttl(first_key) > PARALLEL_REQUEST_SLOT_TTL_SECONDS - 10
    await client.zrem(second_key, "owner")
    assert await renew(keys=[first_key, second_key], args=["owner", PARALLEL_REQUEST_SLOT_TTL_SECONDS]) == [0]
    assert await client.zscore(second_key, "owner") is None
    assert await client.zscore(first_key, "other") == now
