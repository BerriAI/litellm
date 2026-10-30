import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    build_model_max_budget_usage,
    resolve_model_budget,
    team_member_budget_entity_id,
)
from tests._support.fake_redis import fake_redis_cache

NOW: Final = datetime(2026, 1, 31, 12, tzinfo=timezone.utc).timestamp()
BUDGET: Final = {"métric:*": {"budget_limit": 10.0, "time_period": "1mo"}}
PREFIXES: Final = (
    (Litellm_EntityType.KEY, "virtual_key_spend", "virtual_key_budget_start_time"),
    (Litellm_EntityType.USER, "user_model_spend", "user_model_budget_start_time"),
    (Litellm_EntityType.END_USER, "end_user_model_spend", "end_user_budget_start_time"),
    (Litellm_EntityType.TEAM, "team_model_spend", "team_model_budget_start_time"),
    (Litellm_EntityType.TEAM_MEMBER, "team_member_model_spend", "team_member_model_budget_start_time"),
)


@pytest.fixture(autouse=True)
def redis_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fakeredis._basefakesocket.time", SimpleNamespace(time=lambda: NOW))


async def _charge(
    limiter: PROXY_VirtualKeyModelMaxBudgetLimiter,
    entity: Litellm_EntityType = Litellm_EntityType.KEY,
    entity_id: str = "key",
    cost: float = 0.125,
    batch: str | None = None,
) -> None:
    resolved: Final = resolve_model_budget("métric:served", BUDGET)
    assert resolved is not None
    await limiter._charge_entity(entity, entity_id, resolved, cost, batch)


@pytest.mark.asyncio
@pytest.mark.parametrize("entity,spend_prefix,start_prefix", PREFIXES)
async def test_old_and_new_writers_share_bytes_and_calendar_expiry(
    entity: Litellm_EntityType, spend_prefix: str, start_prefix: str
) -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="budgets")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    identity: Final = team_member_budget_entity_id("ü:ser", "team:é")
    assert identity == "5:ü:ser:team:é"
    spend: Final = f"{spend_prefix}:{identity}:métric:*:1mo"
    start: Final = f"{start_prefix}:{identity}:métric:*:1mo"
    await cache.async_set_cache(spend, 1.25, ttl=2419200)
    await cache.async_set_cache(start, NOW, ttl=2419200)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    before: Final = await build_model_max_budget_usage(entity, identity, BUDGET, cache=limiter.dual_cache)
    assert before["métric:*"]["current_spend"] == 1.25
    await _charge(limiter, entity, identity)
    assert await limiter._push_in_memory_increments_to_redis()
    assert await wire.get(f"budgets:{spend}") == b"1.375"
    assert await cache.async_get_cache(spend) == 1.375
    assert await wire.get(f"budgets:{start}") == json.dumps(NOW).encode()
    assert await wire.pttl(f"budgets:{spend}") == 2419200000
    assert await wire.get(f"budgets:budgets:{spend}") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_prefix", ("virtual_key_budget_start_time", "virtual_key_spend"))
async def test_new_start_and_reset_keep_json_scalars_and_separate_writes(
    monkeypatch: pytest.MonkeyPatch, failed_prefix: str
) -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await cache.async_set_cache("virtual_key_budget_start_time:key:métric:*:1mo", NOW - 86400 * 60)
    original: Final = fakeredis.FakeAsyncRedis.set

    async def fail_start(
        self: fakeredis.FakeAsyncRedis, name: str, value: str, *, px: int | None = None, nx: bool = False
    ) -> object:
        if failed_prefix in name:
            raise RedisConnectionError("lost window write")
        return await original(self, name, value, px=px, nx=nx)

    monkeypatch.setattr(fakeredis.FakeAsyncRedis, "set", fail_start)
    await _charge(limiter)
    spend: Final = None if failed_prefix == "virtual_key_spend" else b"0.125"
    start: Final = NOW if failed_prefix == "virtual_key_spend" else NOW - 86400 * 60
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") == spend
    assert await wire.get("n:virtual_key_budget_start_time:key:métric:*:1mo") == json.dumps(start).encode()
    assert await wire.pttl("n:virtual_key_spend:key:métric:*:1mo") == (-2 if spend is None else 2419200000)


@pytest.mark.asyncio
async def test_batch_marker_refreshes_and_only_first_poll_charges() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await _charge(limiter, batch="b:ü")
    assert await limiter._push_in_memory_increments_to_redis()
    marker: Final = "n:virtual_key_spend:key:métric:*:1mo:batch:b:ü"
    await wire.expire(marker, 1)
    await _charge(limiter, batch="b:ü")
    assert await limiter._push_in_memory_increments_to_redis()
    assert await wire.get(marker) == b"2"
    assert await wire.pttl(marker) > 86400000
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") == b"0.125"


@pytest.mark.asyncio
async def test_unknown_applied_increment_is_requeued_by_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await _charge(limiter)
    original: Final = fakeredis.aioredis.FakeConnection.read_response

    async def lose_reply(self: fakeredis.aioredis.FakeConnection, disable_decoding: bool = False) -> object:
        await original(self, disable_decoding=disable_decoding)
        raise RedisConnectionError("write applied but reply lost")

    with monkeypatch.context() as failures:
        failures.setattr(fakeredis.aioredis.FakeConnection, "read_response", lose_reply)
        assert not await limiter._push_in_memory_increments_to_redis()
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") == b"0.125"
    assert await limiter._push_in_memory_increments_to_redis()
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") == b"0.25"


@pytest.mark.asyncio
async def test_cancelled_flush_settles_before_letting_cancellation_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await _charge(limiter)
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    original: Final = fakeredis.aioredis.FakeConnection.read_response

    async def hold_reply(self: fakeredis.aioredis.FakeConnection, disable_decoding: bool = False) -> object:
        result: Final = await original(self, disable_decoding=disable_decoding)
        started.set()
        await release.wait()
        return result

    monkeypatch.setattr(fakeredis.aioredis.FakeConnection, "read_response", hold_reply)
    flush: Final = asyncio.create_task(limiter._push_in_memory_increments_to_redis())
    await started.wait()
    flush.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert await limiter._push_in_memory_increments_to_redis()
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") == b"0.125"


@pytest.mark.asyncio
async def test_outage_requeues_and_breaker_remains_shared_with_cache() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await _charge(limiter)
    server.connected = False
    assert not await limiter._push_in_memory_increments_to_redis()
    server.connected = True
    assert await limiter._push_in_memory_increments_to_redis()
    key: Final = UserAPIKeyAuth(token="key", model_max_budget=BUDGET)
    await cache.async_set_cache("virtual_key_spend:key:métric:*:1mo", 12.0)
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_key_within_model_budget(key, "métric:served")
    for _ in range(cache._circuit_breaker.failure_threshold):
        cache._circuit_breaker.record_failure()
    with pytest.raises(RedisCircuitBreakerOpenError):
        await limiter.is_key_within_model_budget(key, "métric:served")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity,prefix",
    ((Litellm_EntityType.KEY, "virtual_key_spend"), (Litellm_EntityType.END_USER, "end_user_model_spend")),
)
async def test_startup_reads_add_old_requested_model_and_new_configured_spend(
    entity: Litellm_EntityType, prefix: str
) -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await cache.async_set_cache(f"{prefix}:key:métric:served:1mo", 6.0, ttl=86400)
    await _charge(limiter, entity, cost=5.0)
    assert await limiter._push_in_memory_increments_to_redis()
    resolved: Final = resolve_model_budget("métric:served", BUDGET)
    assert resolved is not None
    assert await limiter._get_spend_for_model_budget(entity, "key", "métric:served", resolved) == 11.0
    assert await cache.async_get_cache(f"{prefix}:key:métric:served:1mo") == 6.0


@pytest.mark.asyncio
async def test_malformed_start_does_not_reset_or_charge_the_window() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server, namespace="n")
    wire: Final = fakeredis.FakeAsyncRedis(server=server)
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(DualCache(redis_cache=cache), clock=lambda: NOW)
    await cache.async_set_cache("virtual_key_budget_start_time:key:métric:*:1mo", "bad start")
    with pytest.raises(ValueError, match="not a number"):
        await _charge(limiter)
    assert await limiter._push_in_memory_increments_to_redis()
    assert await wire.get("n:virtual_key_spend:key:métric:*:1mo") is None
    assert await wire.get("n:virtual_key_budget_start_time:key:métric:*:1mo") == b'"bad start"'
