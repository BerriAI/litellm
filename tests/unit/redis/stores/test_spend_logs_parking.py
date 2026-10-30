from datetime import datetime, timezone
from typing import Final

import fakeredis
import pytest

from litellm.constants import REDIS_SPEND_LOGS_BUFFER_KEY
from litellm.redis.stores.spend_logs_parking import TransportSpendLogsParkingStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache

KEY: Final = f"ns:{REDIS_SPEND_LOGS_BUFFER_KEY}"


def _store(server: fakeredis.FakeServer) -> TransportSpendLogsParkingStore:
    return TransportSpendLogsParkingStore(fake_redis_cache(server, namespace="ns").transport())


@pytest.mark.asyncio
async def test_rows_are_parked_as_json_with_iso_datetimes_and_capped_to_the_newest(
    server: fakeredis.FakeServer,
) -> None:
    store: Final = _store(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    started: Final = datetime(2026, 9, 19, 20, 0, 5, tzinfo=timezone.utc)

    assert await store.park(({"request_id": "old"}, {"request_id": "mid", "startTime": started}), max_rows=2) == 2
    assert await store.park(({"request_id": "new"},), max_rows=2) == 3

    assert await raw.lrange(KEY, 0, -1) == [
        b'{"request_id": "mid", "startTime": "2026-09-19T20:00:05+00:00"}',
        b'{"request_id": "new"}',
    ]
    assert await store.take(1) == ('{"request_id": "mid", "startTime": "2026-09-19T20:00:05+00:00"}',)
    assert await store.take(10) == ('{"request_id": "new"}',)
    assert await store.take(10) is None


@pytest.mark.asyncio
async def test_rejections_undecodable_rows_and_outages_are_failures(server: fakeredis.FakeServer) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    store: Final = _store(server)
    await raw.set(KEY, "not a list")
    rejected: Final = await store.park(({"request_id": "a"},), max_rows=10)
    await raw.delete(KEY)
    await raw.rpush(KEY, b"\xff")
    undecodable: Final = await store.take(10)
    server.connected = False
    outage: Final = await store.take(10)

    assert isinstance(rejected, RedisFailure) and rejected.kind == "server"
    assert isinstance(undecodable, RedisFailure) and undecodable.kind == "decode"
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
