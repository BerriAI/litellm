from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from redis.asyncio import Redis

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AuthCacheInvalidationSubscriber,
    auth_cache_invalidation_channel,
    evict_local,
    evict_shared,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.integration._support.redis_process import owned_redis

KEY: Final = hashlib.sha256(b"native-cache-test-key").hexdigest()


async def until(predicate: Callable[[], bool], seconds: float = 12) -> None:
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", (None, "phase3"))
@pytest.mark.parametrize("shared", (False, True))
async def test_native_subscriber_invalidates_and_recovers_without_serving_outage_fills(
    tmp_path: Path, namespace: str | None, shared: bool
) -> None:
    with owned_redis(tmp_path) as server:
        redis_cache: Final = RedisCache(
            host=server.host,
            port=server.port,
            namespace=namespace,
            socket_timeout=0.3,
            socket_connect_timeout=0.3,
        )
        cache: Final = UserApiKeyCache(redis_cache=redis_cache if shared else None)
        additional: Final = InMemoryCache()
        spend: Final = SpendCounterStore(connect(None).spend_counters())
        subscriber: Final = AuthCacheInvalidationSubscriber(redis_cache, cache, (additional,), spend)
        publisher: Final = Redis(host=server.host, port=server.port)
        channel: Final = auth_cache_invalidation_channel(redis_cache)
        subscriber.start()
        started: Final = cache.current_generation()
        try:
            assert not subscriber.is_ready
            await until(lambda: subscriber.is_ready)
            assert cache.current_generation() > started
            fill: Final = cache.begin_fill(KEY)
            assert cache.set_if_generation(KEY, {"token": KEY, "models": ["allowed"]}, fill)
            assert cache.get_local(KEY) == {"token": KEY, "models": ["allowed"]}
            spend.set_local("spend:key:k-1", 99.0)
            assert (
                await publisher.publish(
                    channel, json.dumps({"cache_key": "spend:key:k-1", "new_value": 0.0, "ttl": 60})
                )
                == 1
            )
            await until(lambda: spend.get_local("spend:key:k-1") == 0.0)
            assert cache.get_local(KEY) == {"token": KEY, "models": ["allowed"]}
            assert (
                await publisher.publish(
                    channel, json.dumps({"cache_key": "spend_db_floor:spend:key:k-1", "new_value": 3.0})
                )
                == 1
            )
            await until(lambda: additional.get_cache("spend_db_floor:spend:key:k-1") == 3.0)
            assert await publisher.publish(channel, json.dumps({"cache_key": "spend:key:k-1"})) == 1
            await until(lambda: spend.get_local("spend:key:k-1") is None)
            additional.set_cache("model-cache", "stale")
            assert await publisher.publish(channel, json.dumps({"cache_key": KEY})) == 1
            await until(lambda: cache.get_local(KEY) is None)
            assert not cache.set_if_generation(KEY, {"token": KEY}, fill)
            assert await publisher.publish(channel, json.dumps({"cache_key": "model-cache"})) == 1
            await until(lambda: additional.get_cache("model-cache") is None)

            cache.set_local(KEY, {"token": KEY}, None)
            before_stop: Final = cache.current_generation()
            server.stop()
            await until(lambda: cache.current_generation() > before_stop)
            assert not subscriber.is_ready
            assert cache.get_local(KEY) is None
            outage: Final = cache.begin_fill(KEY)
            assert not cache.set_if_generation(KEY, {"token": KEY}, outage)
            cache.set_local(KEY, {"token": KEY}, None)
            assert cache.get_local(KEY) is None
            server.start()
            await until(lambda: subscriber.is_ready)
            assert cache.current_generation() > outage
            assert not cache.set_if_generation(KEY, {"token": KEY}, outage)
            assert cache.set_if_generation(KEY, {"token": KEY}, cache.begin_fill(KEY))
            assert await publisher.publish(channel, json.dumps({"cache_key": KEY})) == 1
            await until(lambda: cache.get_local(KEY) is None)
        finally:
            await subscriber.stop()
            await publisher.aclose()
        assert cache.get_local(KEY) is None
        assert not cache.set_if_generation(KEY, {"token": KEY}, cache.begin_fill(KEY))


@pytest.mark.asyncio
async def test_management_eviction_reaches_native_memory_and_shared_redis(tmp_path: Path) -> None:
    with owned_redis(tmp_path) as server:
        redis_cache: Final = RedisCache(host=server.host, port=server.port, namespace="management")
        first: Final = UserApiKeyCache(redis_cache=redis_cache)
        second: Final = UserApiKeyCache(redis_cache=redis_cache)
        await first.async_set_cache(KEY, {"token": KEY})
        assert await second.async_get_cache(KEY) == {"token": KEY}
        evict_local((KEY,), first)
        assert first.get_local(KEY) is None
        await evict_shared((KEY,), first)
        second.delete_local(KEY)
        assert await second.async_get_cache(KEY) is None
