from datetime import timedelta
from typing import Final, Protocol, cast
from unittest.mock import patch

import fakeredis
import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.redis.stores.scheduler_queue import TransportSchedulerQueueStore
from litellm.redis.transport.contract import Arg, RedisFailure
from litellm.scheduler import FlowItem, Scheduler
from tests._support.fake_redis import fake_redis_cache


class LegacyQueueCache(Protocol):
    async def async_set_cache(self, key: str, value: object, *, ttl: int) -> bool | None: ...

    async def async_get_cache(self, key: str) -> object: ...


class RedisQueueWriter(Protocol):
    async def set(
        self, name: str, value: Arg, *, ex: timedelta | None = None, px: int | None = None, nx: bool = False
    ) -> object: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [1.9, 0.9, 0, -0.9, -1.9])
@patch("time.time", return_value=1800000000.0)
async def test_queue_fractional_seconds_match_legacy_set_expiry(
    _clock: object, server: fakeredis.FakeServer, seconds: float
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    key: Final = "tenant:scheduler:queue:model"
    await raw.set(key, b'[[2, "older"]]')
    store: Final = TransportSchedulerQueueStore(cache.transport())
    result: Final = await store.replace_queue("model", ((1, "new"),), ttl=timedelta(seconds=seconds))
    if seconds >= 1:
        assert result is None
        assert await raw.get(key) == b'[[1, "new"]]'
        assert await raw.pttl(key) == 1000
    else:
        assert isinstance(result, RedisFailure) and result.kind == "server"
        assert await raw.get(key) == b'[[2, "older"]]'
        assert await raw.pttl(key) == -1


@pytest.mark.asyncio
@patch("time.time", return_value=1800000000.0)
async def test_queue_rolling_deploy_preserves_bytes_ttl_and_local_snapshot(
    _clock: object, server: fakeredis.FakeServer
) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    scheduler: Final = Scheduler(redis_cache=cache)
    legacy: Final = cast(
        LegacyQueueCache, cache
    )  # cast-ok: narrow the untyped legacy API for wire-compatibility probes
    key: Final = "scheduler:queue:mødel"
    with patch("time.time", return_value=1800000000.0):
        await legacy.async_set_cache(key, [[2, "older-雪"]], ttl=5)
        assert await scheduler.get_queue("mødel") == [(2, "older-雪")]
        await scheduler.add_request(FlowItem(priority=1, request_id="new-雪", model_name="mødel"))
        assert await raw.get("tenant:" + key) == b'[[1, "new-\\u96ea"], [2, "older-\\u96ea"]]'
        assert await legacy.async_get_cache(key) == [[1, "new-雪"], [2, "older-雪"]]
        assert await raw.pttl("tenant:" + key) == 5000
        await raw.delete("tenant:" + key)
        assert await scheduler.get_queue("mødel") == [(1, "new-雪"), (2, "older-雪")]
        scheduler.cache.in_memory_cache.flush_cache()
        assert await scheduler.get_queue("mødel") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_value", [b"garbage", b"null", b"{}", b'"text"', b"\xff"])
async def test_queue_unreadable_or_non_list_snapshot_is_empty(server: fakeredis.FakeServer, raw_value: bytes) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set("scheduler:queue:model", raw_value)
    assert await Scheduler(redis_cache=cache).get_queue("model") == []


@pytest.mark.asyncio
async def test_queue_invalid_entries_still_raise_and_python_literals_still_read(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set("scheduler:queue:valid", b"[(2, 'old')]")
    await raw.set("scheduler:queue:invalid", b'[["bad", "id"]]')
    scheduler: Final = Scheduler(redis_cache=cache)
    assert await scheduler.get_queue("valid") == [(2, "old")]
    with pytest.raises(ValidationError, match="int_parsing"):
        await scheduler.get_queue("invalid")
    await raw.set("scheduler:queue:invalid", b'[[1, "fixed"]]')
    with pytest.raises(ValidationError, match="int_parsing"):
        await scheduler.get_queue("invalid")
    scheduler.cache.in_memory_cache.flush_cache()
    assert await scheduler.get_queue("invalid") == [(1, "fixed")]


@pytest.mark.asyncio
async def test_queue_outage_keeps_local_writes_and_uses_shared_breaker() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server)
    scheduler: Final = Scheduler(redis_cache=cache)
    server.connected = False
    await scheduler.save_queue([(1, "local")], "model")
    assert await scheduler.get_queue("model") == [(1, "local")]
    scheduler.cache.in_memory_cache.flush_cache()
    assert await scheduler.get_queue("model") == []
    for _ in range(5):
        await scheduler.get_queue("model")
    result: Final = await TransportSchedulerQueueStore(cache.transport()).read_queue("model")
    assert isinstance(result, RedisFailure) and result.kind == "breaker_open"


@pytest.mark.asyncio
@patch("time.time", return_value=1800000000.0)
async def test_queue_late_redis_attachment_uses_backend_default_ttl(
    _clock: object, server: fakeredis.FakeServer
) -> None:
    scheduler: Final = Scheduler()
    await scheduler.save_queue([(1, "local")], "model")
    cache: Final = fake_redis_cache(server, namespace="tenant")
    scheduler.cache.attach_redis_cache(cache)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get("tenant:scheduler:queue:model") is None
    with patch("time.time", return_value=1800000000.0):
        await scheduler.save_queue([(2, "shared")], "model")
        assert await raw.pttl("tenant:scheduler:queue:model") == cache.default_ttl * 1000


@pytest.mark.asyncio
async def test_queue_unknown_applied_write_is_not_retried_or_rolled_back(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    original_set: Final = cast(type[RedisQueueWriter], fakeredis.FakeAsyncRedis).set  # cast-ok: typed redis-py boundary
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    scheduler: Final = Scheduler(redis_cache=cache)

    async def applied_then_disconnected(
        name: str, value: Arg, *, ex: timedelta | None = None, px: int | None = None, nx: bool = False
    ) -> None:
        await original_set(raw, name, value, ex=ex, px=px, nx=nx)
        raise RedisConnectionError("response lost after write")

    with patch.object(fakeredis.FakeAsyncRedis, "set", side_effect=applied_then_disconnected) as write:
        await scheduler.save_queue([(1, "written")], "model")
        assert write.call_count == 1
    assert await raw.get("tenant:scheduler:queue:model") == b'[[1, "written"]]'
    assert await scheduler.get_queue("model") == [(1, "written")]
