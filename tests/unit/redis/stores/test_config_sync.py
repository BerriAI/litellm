import asyncio
import json
from unittest.mock import patch

import pytest
from redis.exceptions import ConnectionError

from litellm.proxy.common_utils.config_sync_pubsub import ConfigSyncSubscriber, publish_config_change
from litellm.redis.stores.config_sync import ConfigChange, ReceivedConfigChange, TransportConfigSyncStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


async def next_change(subscription):
    for _ in range(100):
        message = await subscription.receive(timeout=0)
        if message is not None:
            return message
        await asyncio.sleep(0)
    pytest.fail("No config change received")


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", [None, "", "fleet"])
async def test_old_and_new_config_publishers_share_exact_channel_bytes(server, raw, namespace):
    cache = fake_redis_cache(server, namespace=namespace)
    store = TransportConfigSyncStore(cache.transport(), empty_namespace=namespace == "")
    channel = "litellm_proxy.config_change" if namespace is None else f"{namespace}:litellm_proxy.config_change"
    old = raw.pubsub()
    await old.subscribe(channel)
    new = await store.subscribe()
    assert not isinstance(new, RedisFailure)
    try:
        old_payload = json.dumps({"object_type": "models-é", "model_ids": ["id-雪", "id-1"]})
        await raw.publish(channel, old_payload)
        assert await next_change(new) == ReceivedConfigChange(ConfigChange("models-é", ("id-雪", "id-1")))
        await old.get_message(ignore_subscribe_messages=True, timeout=0)
        await old.get_message(ignore_subscribe_messages=True, timeout=0)
        await publish_config_change(cache, "models-é", ("id-雪", "id-1"))
        delivered = await old.get_message(ignore_subscribe_messages=True, timeout=0)
        assert delivered["channel"] == channel.encode()
        assert delivered["data"] == old_payload.encode()
        assert await next_change(new) == ReceivedConfigChange(ConfigChange("models-é", ("id-雪", "id-1")))
        assert await raw.pttl(channel) == -2
    finally:
        await new.close()
        await old.aclose()


@pytest.mark.asyncio
async def test_bad_payload_is_consumed_and_does_not_hide_the_next_change(server, raw):
    store = TransportConfigSyncStore(fake_redis_cache(server).transport())
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    try:
        await raw.publish("litellm_proxy.config_change", b"\xffinvalid")
        malformed = await next_change(subscription)
        assert isinstance(malformed, ReceivedConfigChange)
        assert isinstance(malformed.change, RedisFailure) and malformed.change.kind == "decode"
        await raw.publish("litellm_proxy.config_change", '{"object_type": "settings"}')
        assert await next_change(subscription) == ReceivedConfigChange(ConfigChange("settings"))
    finally:
        await subscription.close()


@pytest.mark.asyncio
async def test_pubsub_stays_unguarded_and_outages_are_failure_values(server, raw):
    cache = fake_redis_cache(server)
    store = TransportConfigSyncStore(cache.transport())
    server.connected = False
    for _ in range(5):
        cache._circuit_breaker.record_failure(is_timeout=True)
    failure = await store.subscribe()
    assert isinstance(failure, RedisFailure) and failure.kind == "transport"
    await publish_config_change(cache, "models")
    server.connected = True
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    try:
        await publish_config_change(cache, "models", ("after-outage",))
        assert await next_change(subscription) == ReceivedConfigChange(ConfigChange("models", ("after-outage",)))
        server.connected = False
        failure = await subscription.receive(timeout=0)
        assert isinstance(failure, RedisFailure) and failure.kind == "transport"
    finally:
        server.connected = True
        await subscription.close()


@pytest.mark.asyncio
async def test_ambiguous_config_publish_is_not_replayed(server, raw):
    cache = fake_redis_cache(server)
    store = TransportConfigSyncStore(cache.transport())
    subscription = await store.subscribe()
    assert not isinstance(subscription, RedisFailure)
    client = cache.init_async_client()
    publish = client.publish

    async def applied_without_reply(channel, message):
        await publish(channel, message)
        raise ConnectionError("reply lost")

    try:
        with patch.object(client, "publish", side_effect=applied_without_reply):
            await publish_config_change(cache, "models", ("once",))
        assert await next_change(subscription) == ReceivedConfigChange(ConfigChange("models", ("once",)))
        assert await subscription.receive(timeout=0) is None
    finally:
        await subscription.close()


@pytest.mark.asyncio
async def test_subscriber_applies_old_messages_and_closes_on_stop(server, raw):
    cache = fake_redis_cache(server, namespace="")
    cache.init_async_client().connection_pool.max_connections = 1
    applied = []
    resynced = asyncio.Event()

    async def apply(model_ids):
        applied.append(model_ids)

    async def resync():
        resynced.set()

    subscriber = ConfigSyncSubscriber(cache, (resync,), apply_model_ids=apply, debounce_seconds=0, jitter_max_seconds=0)
    subscriber.start()
    try:
        for _ in range(100):
            if await raw.pubsub_numsub(":litellm_proxy.config_change") == [(b":litellm_proxy.config_change", 1)]:
                break
            await asyncio.sleep(0)
        else:
            pytest.fail("Config subscriber did not subscribe")
        await raw.publish(":litellm_proxy.config_change", json.dumps({"object_type": "models", "model_ids": ["one"]}))
        await asyncio.wait_for(resynced.wait(), timeout=2)
        assert resynced.is_set()
        assert applied == [("one",)]
    finally:
        await subscriber.stop()
    assert await cache.init_async_client().ping()
