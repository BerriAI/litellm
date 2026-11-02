from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm._internal_context import current_service_target
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore

if TYPE_CHECKING:
    from tests.unit.conftest import RouterRedis


@dataclass
class Clock:
    now: float = 1000.0
    failure: RuntimeError | None = None

    def __call__(self) -> float:
        if self.failure is not None:
            raise self.failure
        return self.now


def _entry(timestamp: float = 1000.0, ttl: float = 120.0) -> CooldownCacheValue:
    return {"exception_received": "limited", "status_code": "429", "timestamp": timestamp, "cooldown_time": ttl}


@pytest.mark.parametrize("suffix", (None, "", "timeout"))
def test_allowed_fails_preserves_local_count_during_redis_failure(suffix: str | None) -> None:
    redis: Final = MagicMock(spec=RedisCache)
    redis.increment_cache.side_effect = ConnectionError("offline")
    clock: Final = Clock()
    store: Final = RouterCooldownStore(redis, clock=clock)
    assert store.increment_allowed_fails("dep", suffix, 30.0) == 1
    assert store.increment_allowed_fails("dep", suffix, 30.0) == 2
    clock.now += 31.0
    assert store.increment_allowed_fails("dep", suffix, 30.0) == 1
    expected: Final = "deployment:dep:allowed_fails" + (f":{suffix}" if suffix else "")
    redis.increment_cache.assert_called_with(expected, 1, ttl=30.0)


def test_sync_wire_write_and_reads_keep_service_target_and_float_ttl() -> None:
    redis: Final = MagicMock(spec=RedisCache)
    seen: Final[list[str | None]] = []
    redis.set_cache.side_effect = lambda *args, **kw: seen.append(current_service_target())
    redis.batch_get_cache.side_effect = lambda **kw: {key: _entry() for key in kw["key_list"]}
    store: Final = RouterCooldownStore(redis, clock=Clock())
    store.put("dep", _entry(ttl=0.5), 0.5)
    redis.set_cache.assert_called_once_with("deployment:dep:cooldown", _entry(ttl=0.5), ttl=0.5)
    reader: Final = RouterCooldownStore(redis, clock=Clock())
    assert reader.active_sync(["dep"], None) == (("dep", _entry()),)
    assert reader.minimum(["dep"], 60.0, None) == 120.0
    assert seen == ["router_cooldowns"]
    assert current_service_target() is None


@pytest.mark.asyncio
async def test_remote_backfill_expires_at_corrected_sixty_second_cap() -> None:
    clock: Final = Clock()
    redis: Final = MagicMock(spec=RedisCache)
    redis.async_batch_get_cache = AsyncMock(return_value={"deployment:dep:cooldown": _entry()})
    store: Final = RouterCooldownStore(redis, clock=clock)
    assert await store.active(["dep"], None) == (("dep", _entry()),)
    clock.now += 61.0
    assert await store.active(["dep"], None) == (("dep", _entry()),)
    assert redis.async_batch_get_cache.await_count == 2
    clock.now += 60.0
    assert await store.active(["dep"], None) == ()


@pytest.mark.asyncio
async def test_lazy_attach_and_failure_retry_do_not_throttle_next_read() -> None:
    store: Final = RouterCooldownStore(None, clock=Clock())
    assert await store.active(["dep"], None) == ()
    redis: Final = MagicMock(spec=RedisCache)
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("offline"))
    store.attach_redis(redis)
    assert await store.active(["dep"], None) == ()
    redis.async_batch_get_cache.side_effect = None
    redis.async_batch_get_cache.return_value = {"deployment:dep:cooldown": _entry()}
    assert await store.active(["dep"], None) == (("dep", _entry()),)
    assert redis.async_batch_get_cache.await_count == 2


def test_minimum_preserves_zero_default_and_does_not_correct_expired_entries() -> None:
    store: Final = RouterCooldownStore(None, clock=Clock())
    store.put("zero", _entry(ttl=0), 600)
    store.put("expired", _entry(timestamp=0, ttl=20), 600)
    assert store.minimum(["zero"], 60, None) == 60
    assert store.minimum(["expired"], 60, None) == 20
    assert store.active_sync(["expired"], None) == ()


@pytest.mark.asyncio
async def test_prefetch_backfill_failure_allows_an_immediate_read_retry(router_redis: "RouterRedis") -> None:
    redis, _ = router_redis
    clock: Final = Clock()
    value: Final = _entry()
    writer: Final = RouterCooldownStore(redis, clock=clock)
    reader: Final = RouterCooldownStore(redis, clock=clock)
    writer.put("dep", value, 120.0)

    with request_redis_batch_scope() as request:
        reader.arm_routing_read(("dep",), None)
        clock.failure = RuntimeError("backfill clock unavailable")
        await request.flush_all()

    clock.failure = None
    assert await reader.active(("dep",), None) == (("dep", value),)
