from typing import Final

import fakeredis
import pytest

from litellm.redis.stores.config_sync import ConfigChange, TransportConfigSyncStore, decode_config_change
from litellm.redis.transport.contract import PubSubMessage, RedisFailure, RedisSubscription
from tests._support.fake_redis import fake_redis_cache

OLD_POD_BYTES: Final = b'{"object_type": "litellm_proxymodeltable", "model_ids": ["m-1", "m-2"]}'


async def _received(subscription: RedisSubscription) -> list[PubSubMessage]:
    replies: Final = [await subscription.get_message(timeout=0.05) for _ in range(3)]
    return [reply for reply in replies if isinstance(reply, PubSubMessage)]


@pytest.mark.asyncio
async def test_new_and_old_pods_exchange_the_same_bytes_on_the_namespaced_channel(server: fakeredis.FakeServer) -> None:
    store: Final = TransportConfigSyncStore(fake_redis_cache(server, namespace="prod-eu").transport())
    old_pod: Final = fake_redis_cache(server).transport()
    old_pod_subscription: Final = await old_pod.subscribe("prod-eu:litellm_proxy.config_change")
    subscription: Final = await store.subscribe()
    assert not isinstance(old_pod_subscription, RedisFailure)
    assert not isinstance(subscription, RedisFailure)

    assert await store.publish_change(ConfigChange("litellm_proxymodeltable", ("m-1", "m-2"))) == 2
    assert await old_pod.publish("prod-eu:litellm_proxy.config_change", OLD_POD_BYTES) == 2

    expected: Final = [PubSubMessage(channel=b"prod-eu:litellm_proxy.config_change", data=OLD_POD_BYTES)] * 2
    assert await _received(old_pod_subscription) == expected
    assert await _received(subscription) == expected
    assert decode_config_change(OLD_POD_BYTES) == ConfigChange("litellm_proxymodeltable", ("m-1", "m-2"))
    assert await subscription.close() is None
    assert await old_pod_subscription.close() is None


@pytest.mark.parametrize("payload", [b"change", b'{"object_type": "t", "model_ids": "m-1"}', b"{}", b"\xff"])
def test_unusable_payloads_decode_to_no_change(payload: bytes) -> None:
    assert decode_config_change(payload) is None


@pytest.mark.asyncio
async def test_outage_is_a_transport_failure_for_publish_and_subscribe(server: fakeredis.FakeServer) -> None:
    store: Final = TransportConfigSyncStore(fake_redis_cache(server).transport())
    server.connected = False

    published: Final = await store.publish_change(ConfigChange("litellm_proxymodeltable"))
    subscribed: Final = await store.subscribe()

    assert isinstance(published, RedisFailure) and published.kind == "transport"
    assert isinstance(subscribed, RedisFailure) and subscribed.kind == "transport"
