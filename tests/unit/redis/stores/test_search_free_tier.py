from collections.abc import Iterator
from datetime import timedelta
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.redis.stores.search_free_tier import TransportSearchFreeTierStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import (
    fake_redis_cache,
    keys_matching,
    legacy_default_ttl,
    legacy_increment,
    set_reply_callback,
)

_KEY: Final = "ns:litellm:search_free_tier:vertex_ai:shared:2026-10"
_LOGICAL_KEY: Final = "litellm:search_free_tier:vertex_ai:shared:2026-10"


def _not_a_number(_: object) -> bytes:
    return b"not-a-number"


@pytest.fixture(autouse=True)
def frozen_clock() -> Iterator[None]:
    with patch("time.time", return_value=1792497600.0):
        yield


@pytest.mark.asyncio
async def test_old_and_new_writers_share_numeric_bytes_and_do_not_refresh_expiry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    assert await legacy_increment(cache, _LOGICAL_KEY, 2.5, ttl=600) == 2.5
    assert await raw.get(_KEY) == b"2.5"
    assert await raw.pttl(_KEY) == 600_000
    assert await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=900)) == 5.5
    assert await raw.get(_KEY) == b"5.5"
    assert await raw.pttl(_KEY) == 600_000
    assert await legacy_increment(cache, _LOGICAL_KEY, 1.5, ttl=1200) == 7.0
    assert await raw.get(_KEY) == b"7"
    assert await raw.pttl(_KEY) == 600_000
    assert await keys_matching(raw, "*") == (_KEY.encode(),)


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [None, timedelta(seconds=60)])
async def test_new_writer_is_readable_by_legacy_increment(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, ttl: timedelta | None
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    assert await store.allocate_units("vertex_ai", "shared", "2026-10", 4, ttl=ttl) == 4.0
    assert await raw.get(_KEY) == b"4"
    assert await raw.pttl(_KEY) == (-1 if ttl is None else 60_000)
    assert await legacy_increment(cache, _LOGICAL_KEY, 2.5, ttl=None) == 6.5
    assert await raw.get(_KEY) == b"6.5"
    assert await raw.pttl(_KEY) == (legacy_default_ttl(cache) * 1000 if ttl is None else 60_000)


@pytest.mark.asyncio
async def test_provider_group_and_period_have_independent_allowance_counters(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    store: Final = TransportSearchFreeTierStore(fake_redis_cache(server, namespace="ns").transport())
    identities: Final = (
        ("vertex_ai", "shared", "2026-10"),
        ("gemini", "shared", "2026-10"),
        ("vertex_ai", "other", "2026-10"),
        ("vertex_ai", "shared", "2026-11"),
        ("vertex_ai", "shared", "2026-10-20"),
        ("vertex_ai", "shared 雪", "2026-10"),
    )
    for provider, group, period in identities:
        assert await store.allocate_units(provider, group, period, 2, ttl=None) == 2.0
    assert sorted(await keys_matching(raw, "*")) == sorted(
        f"ns:litellm:search_free_tier:{provider}:{group}:{period}".encode() for provider, group, period in identities
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [-1, 0, 300])
async def test_unexpiring_legacy_counter_gets_seconds_based_expiry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, seconds: int
) -> None:
    store: Final = TransportSearchFreeTierStore(fake_redis_cache(server, namespace="ns").transport())
    legacy: Final = fake_redis_cache(server, namespace="legacy")
    legacy_key: Final = f"legacy:{_LOGICAL_KEY}"
    await raw.set(_KEY, b"5")
    await raw.set(legacy_key, b"5")
    assert await legacy_increment(legacy, _LOGICAL_KEY, 2, ttl=seconds) == 7.0
    assert await store.allocate_units("vertex_ai", "shared", "2026-10", 2, ttl=timedelta(seconds=seconds)) == 7.0
    assert await raw.get(_KEY) == await raw.get(legacy_key)
    assert await raw.pttl(_KEY) == await raw.pttl(legacy_key)


@pytest.mark.asyncio
async def test_expiry_remains_a_separate_write_after_increment(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    async def ttl_reply(value: int) -> int:
        assert value == -1
        assert await raw.get(_KEY) == b"3"
        assert await raw.pttl(_KEY) == -1
        await raw.pexpire(_KEY, 900_000)
        return value

    set_reply_callback(cache, "TTL", ttl_reply)
    assert await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60)) == 3.0
    assert await raw.pttl(_KEY) == 60_000


@pytest.mark.asyncio
async def test_counter_disappearing_before_expire_keeps_successful_allocation_result(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    async def ttl_reply(value: int) -> int:
        await raw.delete(_KEY)
        return value

    set_reply_callback(cache, "TTL", ttl_reply)
    assert await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60)) == 3.0
    assert await raw.get(_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("command,remaining_ttl", [("INCRBYFLOAT", -1), ("TTL", -1), ("EXPIRE", 60_000)])
async def test_malformed_replies_are_failures_after_the_counter_was_consumed(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, command: str, remaining_ttl: int
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())
    set_reply_callback(cache, command, _not_a_number)

    outcome: Final = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60))
    assert isinstance(outcome, RedisFailure) and outcome.kind == "decode"
    assert await raw.get(_KEY) == b"3"
    assert await raw.pttl(_KEY) == remaining_ttl


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["INCRBYFLOAT", "TTL", "EXPIRE"])
async def test_lost_reply_reports_unknown_applied_write_without_retry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, command: str
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    def lose_reply(_: object) -> None:
        raise RedisConnectionError("allocation reply lost")

    set_reply_callback(cache, command, lose_reply)
    outcome: Final = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60))
    assert isinstance(outcome, RedisFailure) and outcome.kind == "transport"
    assert await raw.get(_KEY) == b"3"
    assert await raw.pttl(_KEY) == (60_000 if command == "EXPIRE" else -1)


@pytest.mark.asyncio
async def test_corrupt_counter_is_not_reset_or_expired(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    store: Final = TransportSearchFreeTierStore(fake_redis_cache(server, namespace="ns").transport())
    await raw.set(_KEY, b"corrupt")
    outcome: Final = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60))
    assert isinstance(outcome, RedisFailure) and outcome.kind == "server"
    assert await raw.get(_KEY) == b"corrupt"
    assert await raw.pttl(_KEY) == -1


@pytest.mark.asyncio
async def test_outage_opens_existing_cache_breaker_and_rejects_writes_after_reconnection(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    store: Final = TransportSearchFreeTierStore(fake_redis_cache(server, namespace="ns").transport())
    server.connected = False
    for _ in range(10):
        outcome = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=None)
        assert isinstance(outcome, RedisFailure) and outcome.kind in ("transport", "breaker_open")
    server.connected = True
    blocked: Final = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=None)
    assert isinstance(blocked, RedisFailure) and blocked.kind == "breaker_open"
    assert await raw.get(_KEY) is None


@pytest.mark.asyncio
async def test_expiry_server_failure_reports_consumed_allocation_without_retry(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    store: Final = TransportSearchFreeTierStore(cache.transport())

    def denied(_: object) -> None:
        raise ResponseError("NOPERM expiry denied")

    set_reply_callback(cache, "EXPIRE", denied)
    outcome: Final = await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60))
    assert isinstance(outcome, RedisFailure) and outcome.kind == "server"
    assert await raw.get(_KEY) == b"3"


@pytest.mark.asyncio
async def test_allocation_flushes_pending_post_call_work_on_the_same_transport(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="ns")
    transport: Final = cache.transport()
    store: Final = TransportSearchFreeTierStore(transport)
    with request_redis_batch_scope():
        batch: Final = transport.post_call_batch()
        assert batch is not None
        pending: Final = batch.set("other-keyspace", b"queued", ttl=None)
        assert await store.allocate_units("vertex_ai", "shared", "2026-10", 3, ttl=timedelta(seconds=60)) == 3.0
        assert pending.done and await pending is True
    assert await raw.get("ns:other-keyspace") == b"queued"
    assert await raw.get(_KEY) == b"3"
    assert await raw.pttl(_KEY) == 60_000
