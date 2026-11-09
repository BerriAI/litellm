from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import psycopg
import pytest
from redis.asyncio import Redis

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.auth.auth_checks import get_team_object
from litellm.proxy.auth.auth_object_prefetch import AuthObjectRefs, prefetch_auth_objects, prefetch_identity_keys
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AuthCacheInvalidationSubscriber,
    auth_cache_invalidation_channel,
    evict_local,
    evict_shared,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.repositories.database_settings import NativeDatabaseSettings
from litellm.rust_bridge._native import NativeDatabase
from tests.integration._support.database_relay import HeldStatementRelay
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


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", (True, False))
async def test_prefetch_does_not_restore_an_invalidated_redis_reply(tmp_path: Path, identity: bool) -> None:
    key: Final = KEY if identity else "team_id:prefetch-race"
    value: Final = {"token": KEY, "blocked": False, "marker": "held-redis-prefetch"}
    with owned_redis(tmp_path) as server:
        relay: Final = HeldStatementRelay(server.host, server.port, b"held-redis-prefetch", response=True)
        relay.start()
        coordination: Final = RedisCache(host=server.host, port=server.port)
        cache: Final = UserApiKeyCache(redis_cache=RedisCache(host=server.host, port=relay.port, socket_timeout=30))
        writer: Final = UserApiKeyCache(redis_cache=coordination)
        subscriber: Final = AuthCacheInvalidationSubscriber(coordination, cache)
        redis: Final = Redis(host=server.host, port=server.port)
        subscriber.start()
        try:
            await until(lambda: subscriber.is_ready)
            await writer.async_set_cache(key, value)
            fill: Final = asyncio.create_task(
                prefetch_identity_keys((key,), cache)
                if identity
                else prefetch_auth_objects(AuthObjectRefs(team_id="prefetch-race"), cache, None)
            )
            await until(relay.held.is_set)
            generation: Final = cache.current_generation()
            await evict_shared((key,), writer)
            assert (
                await redis.publish(auth_cache_invalidation_channel(coordination), json.dumps({"cache_key": key})) == 1
            )
            await until(lambda: cache.current_generation() > generation)
            assert await redis.get(key) is None
            assert cache.get_local(key) is None
            relay.release()
            await fill
            assert cache.get_local(key) is None
            assert await cache.async_get_cache(key) is None
        finally:
            relay.release()
            await subscriber.stop()
            await redis.aclose()
            relay.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("prefetch", (True, False))
async def test_sqlx_prefetch_does_not_restore_local_or_shared_context_after_invalidation(
    tmp_path: Path, prefetch: bool
) -> None:
    team_id: Final = f"prefetch-{uuid.uuid4().hex}"
    key: Final = f"team_id:{team_id}"
    settings: Final = NativeDatabaseSettings.from_url(os.environ["DATABASE_URL"])
    relay: Final = HeldStatementRelay(settings.host, settings.port, team_id.encode(), response=True)
    relay.start()
    database: Final = await NativeDatabase.connect(
        replace(settings, host="127.0.0.1", port=relay.port, ssl_mode="disable")
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute('INSERT INTO "LiteLLM_TeamTable" (team_id, blocked) VALUES (%s, false)', (team_id,))
    with owned_redis(tmp_path) as server:
        coordination: Final = RedisCache(host=server.host, port=server.port)
        cache: Final = UserApiKeyCache(redis_cache=coordination)
        writer: Final = UserApiKeyCache(redis_cache=coordination)
        subscriber: Final = AuthCacheInvalidationSubscriber(coordination, cache)
        redis: Final = Redis(host=server.host, port=server.port)
        subscriber.start()
        try:
            await until(lambda: subscriber.is_ready)
            fill: Final = asyncio.create_task(
                prefetch_auth_objects(
                    AuthObjectRefs(team_id=team_id), cache, SimpleNamespace(native_auth_database=database)
                )
                if prefetch
                else get_team_object(team_id, SimpleNamespace(native_auth_database=database), cache)
            )
            await until(relay.held.is_set)
            with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
                connection.execute('UPDATE "LiteLLM_TeamTable" SET blocked=true WHERE team_id=%s', (team_id,))
            generation: Final = cache.current_generation()
            await evict_shared((key,), writer)
            assert (
                await redis.publish(auth_cache_invalidation_channel(coordination), json.dumps({"cache_key": key})) == 1
            )
            await until(lambda: cache.current_generation() > generation)
            assert await redis.get(key) is None
            assert cache.get_local(key) is None
            relay.release()
            await fill
            assert await redis.get(key) is None
            assert cache.get_local(key) is None
            await prefetch_auth_objects(
                AuthObjectRefs(team_id=team_id), cache, SimpleNamespace(native_auth_database=database)
            )
            assert cache.get_local(key)["blocked"] is True
        finally:
            relay.release()
            await subscriber.stop()
            await redis.aclose()
            await database.close()
            relay.stop()
            with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
                connection.execute('DELETE FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team_id,))
