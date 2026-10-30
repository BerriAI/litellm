from datetime import timedelta
from typing import Final

import fakeredis
import pytest

from litellm.redis.stores.legacy_request_limits import (
    InvalidSnapshot,
    MinuteLimit,
    TransportLegacyRequestLimitsStore,
    UsageSnapshot,
)
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest.mark.asyncio
async def test_legacy_dictionary_bytes_round_trip_both_directions(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="limits")
    store: Final = TransportLegacyRequestLimitsStore(cache.transport())
    value: Final[UsageSnapshot] = {"current_requests": 1, "current_tpm": 2.5, "current_rpm": 3, "雪": 4}
    key: Final = "clé::model::2026-01-31-23-59::request_count"
    limit: Final = MinuteLimit("clé", "2026-01-31-23-59", "model")
    await cache.async_set_cache(key, value, ttl=60)
    old_bytes: Final = await raw.get(f"limits:{key}")
    assert old_bytes == b'{"current_requests": 1, "current_tpm": 2.5, "current_rpm": 3, "\\u96ea": 4}'
    assert await store.read(limit) == value
    assert await store.write(limit, value, ttl=timedelta(seconds=60)) is True
    assert await raw.get(f"limits:{key}") == old_bytes
    assert await cache.async_get_cache(key) == value
    assert 0 < await raw.pttl(f"limits:{key}") <= 60000
    assert await raw.get(f"limits:limits:{key}") is None


@pytest.mark.asyncio
async def test_batch_keeps_duplicate_keys_and_last_writer_wins(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="limits")
    store: Final = TransportLegacyRequestLimitsStore(cache.transport())
    limit: Final = MinuteLimit("same-id", "2026-01-31-23-59")
    first: Final[UsageSnapshot] = {"current_requests": 1, "current_tpm": 0, "current_rpm": 1}
    last: Final[UsageSnapshot] = {"current_requests": 2, "current_tpm": 3, "current_rpm": 4}
    assert await store.write_many(((limit, first), (limit, last)), ttl=timedelta(seconds=60)) == (True, True)
    assert await store.read_many((limit, limit, MinuteLimit("missing", limit.minute))) == (last, last, None)
    assert await cache.async_get_cache("same-id::2026-01-31-23-59::request_count") == last
    assert 0 < await raw.pttl("limits:same-id::2026-01-31-23-59::request_count") <= 60000


@pytest.mark.asyncio
async def test_python_literal_compatibility_and_decode_failures(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    store: Final = TransportLegacyRequestLimitsStore(fake_redis_cache(server, namespace="limits").transport())
    limit: Final = MinuteLimit("key", "minute")
    await raw.set("limits:key::minute::request_count", "{'current_requests': 2, 'current_tpm': 0, 'current_rpm': 3}")
    assert await store.read(limit) == {"current_requests": 2, "current_tpm": 0, "current_rpm": 3}
    await raw.set("limits:key::minute::request_count", "broken{")
    malformed: Final = await store.read(limit)
    assert isinstance(malformed, RedisFailure) and malformed.kind == "decode"
    await raw.set("limits:key::minute::request_count", "[]")
    assert await store.read(limit) == InvalidSnapshot(empty=True)


@pytest.mark.asyncio
async def test_outage_is_not_a_missing_minute(server: fakeredis.FakeServer) -> None:
    store: Final = TransportLegacyRequestLimitsStore(fake_redis_cache(server, namespace="limits").transport())
    server.connected = False
    result: Final = await store.read(MinuteLimit("key", "minute"))
    assert isinstance(result, RedisFailure) and result.kind == "transport"


@pytest.mark.asyncio
async def test_malformed_batch_preserves_old_undecoded_result_for_every_present_key(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="limits")
    store: Final = TransportLegacyRequestLimitsStore(cache.transport())
    valid: Final = b'{"current_requests": 2, "current_tpm": 0, "current_rpm": 3}'
    await raw.set("limits:key::minute::request_count", valid)
    await raw.set("limits:bad::minute::request_count", b"broken{")
    assert await cache.async_batch_get_cache(
        ["key::minute::request_count", "bad::minute::request_count", "missing::minute::request_count"]
    ) == {
        "key::minute::request_count": valid,
        "bad::minute::request_count": b"broken{",
        "missing::minute::request_count": None,
    }
    values: Final = await store.read_many(
        tuple(MinuteLimit(identity, "minute") for identity in ("key", "bad", "missing"))
    )
    assert isinstance(values, tuple)
    assert isinstance(values[0], InvalidSnapshot) and not values[0].empty
    assert isinstance(values[1], InvalidSnapshot) and values[1].failure is not None
    assert values[1].failure.kind == "decode"
    assert values[2] is None
