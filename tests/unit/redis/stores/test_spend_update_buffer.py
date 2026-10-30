from typing import Final

import fakeredis
import pytest

from litellm.constants import REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY, REDIS_UPDATE_BUFFER_KEY
from litellm.redis.stores.spend_update_buffer import TransportSpendUpdateBufferStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


def _store(server: fakeredis.FakeServer) -> TransportSpendUpdateBufferStore:
    return TransportSpendUpdateBufferStore(fake_redis_cache(server, namespace="ns").transport())


@pytest.mark.asyncio
async def test_batches_land_on_the_legacy_lists_as_safe_dumps_json(server: fakeredis.FakeServer) -> None:
    store: Final = _store(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)

    assert await store.push("daily_tag", {"k": {"spend": 1.5}}) == 1
    assert await store.push_all((("spend", {"key_list_transactions": {"a": 0.5}}), ("daily_tag", {"k": {}}))) == (1, 2)

    assert await raw.lrange(f"ns:{REDIS_UPDATE_BUFFER_KEY}", 0, -1) == [b'{"key_list_transactions": {"a": 0.5}}']
    assert await raw.lrange(f"ns:{REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY}", 0, -1) == [
        b'{"k": {"spend": 1.5}}',
        b'{"k": {}}',
    ]
    assert await store.pop("daily_tag", count=1) == ('{"k": {"spend": 1.5}}',)
    assert await store.pop_all(("spend", "daily_tag", "window"), count=10) == (
        ('{"key_list_transactions": {"a": 0.5}}',),
        ('{"k": {}}',),
        None,
    )
    assert await raw.exists(f"ns:{REDIS_UPDATE_BUFFER_KEY}", f"ns:{REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY}") == 0


@pytest.mark.asyncio
async def test_a_rejected_push_fails_the_batch_but_the_other_lists_keep_their_push(
    server: fakeredis.FakeServer,
) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(f"ns:{REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY}", "not a list")
    store: Final = _store(server)

    pushed: Final = await store.push_all((("spend", {"a": 1}), ("daily_tag", {"b": 2})))

    assert isinstance(pushed, RedisFailure) and pushed.kind == "server"
    assert await raw.lrange(f"ns:{REDIS_UPDATE_BUFFER_KEY}", 0, -1) == [b'{"a": 1}']


@pytest.mark.asyncio
async def test_undecodable_entries_and_outages_are_failures(server: fakeredis.FakeServer) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.rpush(f"ns:{REDIS_UPDATE_BUFFER_KEY}", b"\xff\xfe")
    store: Final = _store(server)

    undecodable: Final = await store.pop("spend", count=10)
    server.connected = False
    outage: Final = await store.pop_all(("spend",), count=10)

    assert isinstance(undecodable, RedisFailure) and undecodable.kind == "decode"
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
