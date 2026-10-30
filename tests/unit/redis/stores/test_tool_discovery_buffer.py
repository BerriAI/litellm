from typing import Final

import fakeredis
import pytest

from litellm.constants import REDIS_TOOL_DISCOVERY_BUFFER_KEY
from litellm.redis.stores.tool_discovery_buffer import TransportToolDiscoveryBufferStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

KEY: Final = f"ns:{REDIS_TOOL_DISCOVERY_BUFFER_KEY}"


def _store(server: fakeredis.FakeServer) -> TransportToolDiscoveryBufferStore:
    return TransportToolDiscoveryBufferStore(fake_redis_cache(server, namespace="ns").transport())


@pytest.mark.asyncio
async def test_each_push_is_one_json_list_entry_and_a_drain_returns_every_batch(server: fakeredis.FakeServer) -> None:
    store: Final = _store(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)

    assert await store.push(({"tool_name": "a", "call_count": 2},)) == 1
    assert await store.push(({"tool_name": "b", "origin": None, "call_count": 1},)) == 2

    assert await raw.lrange(KEY, 0, -1) == [
        b'[{"tool_name": "a", "call_count": 2}]',
        b'[{"tool_name": "b", "origin": null, "call_count": 1}]',
    ]
    assert await store.drain(count=10) == (
        ({"tool_name": "a", "call_count": 2},),
        ({"tool_name": "b", "origin": None, "call_count": 1},),
    )
    assert await store.drain(count=10) == ()


@pytest.mark.asyncio
async def test_bad_entries_rejections_and_outages_are_failures(server: fakeredis.FakeServer) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    store: Final = _store(server)
    await raw.rpush(KEY, b'{"not": "a list"}')
    malformed: Final = await store.drain(count=10)
    await raw.set(KEY, "not a list")
    rejected: Final = await store.push(({"tool_name": "a", "call_count": 1},))
    server.connected = False
    outage: Final = await store.drain(count=10)

    assert isinstance(malformed, RedisFailure) and malformed.kind == "decode"
    assert isinstance(rejected, RedisFailure) and rejected.kind == "server"
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
