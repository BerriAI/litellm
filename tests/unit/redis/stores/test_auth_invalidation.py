import asyncio
import json
import math
from unittest.mock import patch

import pytest
from redis.exceptions import ConnectionError

import litellm.proxy.common_utils.auth_cache_invalidation_pubsub as publishers
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy import proxy_server
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import AuthCacheInvalidationSubscriber
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_invalidation import (
    AuthInvalidation,
    TransportAuthInvalidationStore,
    decode_auth_invalidation,
)
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest.fixture(autouse=True)
def isolated_publish_tasks(monkeypatch):
    monkeypatch.setattr(publishers, "_pending_publishes", set())
    monkeypatch.setattr(publishers, "_in_flight_publishes", asyncio.Semaphore(16))


async def next_invalidation(subscription):
    for _ in range(100):
        message = await subscription.receive(timeout=0)
        if message is not None:
            return message
        await asyncio.sleep(0)
    pytest.fail("No auth invalidation received")


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", [None, "", "fleet"])
@pytest.mark.parametrize("value,ttl", [(None, None), (0.0, 60), (42, 1.25), (None, 0.0)])
async def test_old_and_new_publishers_preserve_channels_bytes_and_optional_fields(
    server, raw, monkeypatch, namespace, value, ttl
):
    cache = fake_redis_cache(server, namespace=namespace)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    store = TransportAuthInvalidationStore(cache.transport(), empty_namespace=namespace == "")
    channel = (
        "litellm_proxy.auth_cache_invalidation"
        if namespace is None
        else f"{namespace}:litellm_proxy.auth_cache_invalidation"
    )
    old = raw.pubsub()
    await old.subscribe(channel)
    await old.get_message(ignore_subscribe_messages=True, timeout=0)
    new = await store.subscribe()
    assert not isinstance(new, RedisFailure)
    expected = json.dumps(
        {
            key: item
            for key, item in {"cache_key": "counter:雪", "new_value": value, "ttl": ttl}.items()
            if item is not None
        }
    ).encode()
    try:
        await raw.publish(channel, expected)
        assert await next_invalidation(new) == AuthInvalidation("counter:雪", value, ttl)
        assert (await old.get_message(ignore_subscribe_messages=True, timeout=0))["data"] == expected
        await publishers.publish_auth_cache_invalidation("counter:雪", new_value=value, ttl=ttl)
        await asyncio.gather(*publishers._pending_publishes)
        delivered = await old.get_message(ignore_subscribe_messages=True, timeout=0)
        assert delivered["channel"] == channel.encode()
        assert delivered["data"] == expected
        assert await next_invalidation(new) == AuthInvalidation("counter:雪", value, ttl)
        assert await raw.pttl(channel) == -2
    finally:
        await new.close()
        await old.aclose()


@pytest.mark.asyncio
async def test_batch_evictions_publish_every_key_in_order_without_deleting_shared_data(server, raw, monkeypatch):
    cache = fake_redis_cache(server, namespace="fleet")
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    store = TransportAuthInvalidationStore(cache.transport())
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    keys = ("project_id:one", "user_id:two", "credentials:雪", "project_id:one")
    await raw.set("fleet:project_id:one", b"shared auth object")
    try:
        await publishers.broadcast_evictions(keys)
        assert tuple([await next_invalidation(subscription) for _ in keys]) == tuple(
            AuthInvalidation(key) for key in keys
        )
        assert await raw.get("fleet:project_id:one") == b"shared auth object"
        assert await subscription.receive(timeout=0) is None
    finally:
        await subscription.close()


@pytest.mark.parametrize(
    "payload,expected",
    [
        ('{"cache_key":"key","new_value":true,"ttl":"60"}', AuthInvalidation("key")),
        ('{"cache_key":"key","new_value":0,"ttl":1.5}', AuthInvalidation("key", 0.0, 1.5)),
        (b'{"cache_key":"\xff"}', AuthInvalidation("�")),
        ('{"cache_key":"key","new_value":Infinity}', AuthInvalidation("key", float("inf"))),
    ],
)
def test_legacy_numeric_and_utf8_replacement_rules(payload, expected):
    assert decode_auth_invalidation(payload) == expected


def test_legacy_nan_remains_a_numeric_replacement():
    value = decode_auth_invalidation('{"cache_key":"key","new_value":NaN}')
    assert isinstance(value, AuthInvalidation) and math.isnan(value.new_value)


@pytest.mark.asyncio
async def test_malformed_messages_are_consumed_before_the_next_valid_eviction(server, raw):
    store = TransportAuthInvalidationStore(fake_redis_cache(server).transport())
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    try:
        for payload in (b"garbage", b"{}", b"[]", b'{"cache_key":7}'):
            await raw.publish("litellm_proxy.auth_cache_invalidation", payload)
            result = await next_invalidation(subscription)
            assert isinstance(result, RedisFailure) and result.kind == "decode"
        await raw.publish("litellm_proxy.auth_cache_invalidation", '{"cache_key":"one"}')
        assert await next_invalidation(subscription) == AuthInvalidation("one")
    finally:
        await subscription.close()


@pytest.mark.asyncio
async def test_unguarded_publications_recover_after_outages(server, raw, monkeypatch):
    cache = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    store = TransportAuthInvalidationStore(cache.transport())
    server.connected = False
    for _ in range(5):
        cache._circuit_breaker.record_failure(is_timeout=True)
    failure = await store.publish_change("single")
    assert isinstance(failure, RedisFailure) and failure.kind == "transport"
    batch_failure = await store.publish_evictions(("one", "two"))
    assert isinstance(batch_failure, RedisFailure) and batch_failure.kind == "transport"
    await publishers.publish_auth_cache_invalidation("single")
    await publishers.broadcast_evictions(("one", "two"))
    server.connected = True
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    try:
        await publishers.publish_auth_cache_invalidation("recovered", 1.5, 60)
        await asyncio.gather(*publishers._pending_publishes)
        assert await next_invalidation(subscription) == AuthInvalidation("recovered", 1.5, 60)
    finally:
        await subscription.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_applied_but_unacknowledged_publications_are_not_replayed(server, raw, monkeypatch, batch):
    cache = fake_redis_cache(server)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", cache)
    store = TransportAuthInvalidationStore(cache.transport())
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    client = cache.init_async_client()
    pipeline = client.pipeline(transaction=False)
    original = pipeline.execute if batch else client.publish

    async def applied_without_reply(*args):
        await original(*args)
        raise ConnectionError("reply lost")

    try:
        with (
            patch.object(client, "pipeline", return_value=pipeline),
            patch.object(
                pipeline if batch else client, "execute" if batch else "publish", side_effect=applied_without_reply
            ),
        ):
            if batch:
                await publishers.broadcast_evictions(("once", "second"))
            else:
                await publishers.publish_auth_cache_invalidation("once")
            await asyncio.gather(*publishers._pending_publishes)
        assert await next_invalidation(subscription) == AuthInvalidation("once")
        if batch:
            assert await next_invalidation(subscription) == AuthInvalidation("second")
        assert await subscription.receive(timeout=0) is None
    finally:
        await subscription.close()


@pytest.mark.asyncio
async def test_subscriber_preserves_counter_and_floor_replacements_then_evicts_only_local_caches(server, raw):
    cache = fake_redis_cache(server, namespace="")
    cache.init_async_client().connection_pool.max_connections = 1
    clock = [1000.0]
    extra = InMemoryCache(clock=lambda: clock[0])
    second_extra = InMemoryCache(clock=lambda: clock[0])
    auth = UserApiKeyCache()
    counter = "spend:key:key-id"
    floor = f"spend_db_floor:{counter}"
    auth.in_memory_cache.set_cache(counter, "local auth state")
    auth.in_memory_cache.set_cache("project_id:one", "cached object")
    extra.set_cache("project_id:one", 99.0)
    await raw.set("project_id:one", b"shared state")
    subscriber = AuthCacheInvalidationSubscriber(cache, auth, (extra, second_extra))
    subscriber.start()
    channel = ":litellm_proxy.auth_cache_invalidation"
    try:
        for _ in range(100):
            if await raw.pubsub_numsub(channel) == [(channel.encode(), 1)]:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("Auth subscriber did not subscribe")
        await raw.publish(channel, b"malformed")
        await raw.publish(channel, json.dumps({"cache_key": counter, "new_value": 0.0, "ttl": 60}))
        await raw.publish(channel, json.dumps({"cache_key": floor, "new_value": 12.5, "ttl": 5}))
        await wait_until(lambda: second_extra.get_cache(floor) == 12.5)
        assert extra.get_cache(counter) == second_extra.get_cache(counter) == 0.0
        assert extra.get_cache(floor) == second_extra.get_cache(floor) == 12.5
        assert auth.in_memory_cache.get_cache(counter) == "local auth state"
        clock[0] = 1005.1
        assert extra.get_cache(floor) is None and second_extra.get_cache(floor) is None
        assert extra.get_cache(counter) == second_extra.get_cache(counter) == 0.0
        clock[0] = 1060.1
        assert extra.get_cache(counter) is None and second_extra.get_cache(counter) is None
        await raw.publish(channel, json.dumps({"cache_key": "project_id:one"}))
        await wait_until(lambda: auth.in_memory_cache.get_cache("project_id:one") is None)
        assert extra.get_cache("project_id:one") is None
        assert await raw.get("project_id:one") == b"shared state"
    finally:
        await subscriber.stop()
    assert await cache.init_async_client().ping()
