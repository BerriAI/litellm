"""Committed spend twins hold only charged cost, so the remaining budget gauges never count a budget hold."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis, FakeServer

from litellm._service_logger import ServiceLogging
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.proxy.spend_tracking.committed_spend import (
    EntitySpendCounterKeys,
    committed_spend_key,
    read_committed_budget_spend,
    seed_committed_spend,
)
from litellm.types.proxy.carried_budget_state import CommittedBudgetSpend


class _FakeRedisSettingMax(FakeAsyncRedis):
    """fakeredis runs no Lua without lupa, so this stands in for the GET/compare/SET script of ``async_set_max``."""

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> bytes:
        key, value, _ttl = keys_and_args
        current: Final = await self.get(key)
        if current is not None and float(current) >= float(value):
            return current
        await self.set(key, value)
        return value.encode()


class _RedisCacheOnClient(RedisCache):
    def __init__(self, client: FakeAsyncRedis) -> None:
        self._injected_client: Final = client
        self.namespace = None
        self._circuit_breaker = RedisCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        self.service_logger_obj = ServiceLogging()
        self.default_ttl = None

    def init_async_client(self) -> FakeAsyncRedis:
        return self._injected_client


class _FindUniqueTable:
    def __init__(self, row: SimpleNamespace | None) -> None:
        self._row: Final = row

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace | None:
        return self._row


def _prisma_with_spend(spend: float | None) -> SimpleNamespace:
    row: Final = None if spend is None else SimpleNamespace(spend=spend)
    return SimpleNamespace(
        db=SimpleNamespace(
            litellm_verificationtoken=_FindUniqueTable(row),
            litellm_teamtable=_FindUniqueTable(row),
            litellm_endusertable=_FindUniqueTable(row),
        )
    )


@pytest.fixture(params=["in_memory", "redis"])
def spend_counter_cache(request: pytest.FixtureRequest) -> DualCache:
    if request.param == "in_memory":
        return DualCache(in_memory_cache=InMemoryCache())
    return DualCache(
        in_memory_cache=InMemoryCache(), redis_cache=_RedisCacheOnClient(_FakeRedisSettingMax(server=FakeServer()))
    )


async def _store(spend_counter_cache: DualCache, key: str, value: float) -> None:
    if spend_counter_cache.redis_cache is None:
        spend_counter_cache.in_memory_cache.set_cache(key=key, value=value)
        return
    await spend_counter_cache.redis_cache.async_set_cache(key=key, value=value)


@pytest.mark.asyncio
async def test_read_returns_each_committed_twin_and_none_for_one_never_charged(spend_counter_cache: DualCache):
    await _store(spend_counter_cache, committed_spend_key("spend:key:hashed"), 1.5)
    await _store(spend_counter_cache, committed_spend_key("spend:end_user:cust-1"), 0.25)
    await _store(spend_counter_cache, "spend:team:t1", 99.0)

    committed: Final = await read_committed_budget_spend(
        spend_counter_cache,
        EntitySpendCounterKeys.of(token="hashed", team_id="t1", user_id=None, org_id=None, end_user_id="cust-1"),
    )

    assert committed == CommittedBudgetSpend(key=1.5, end_user=0.25)


@pytest.mark.asyncio
async def test_read_of_a_twin_holding_garbage_reports_no_committed_spend():
    spend_counter_cache: Final = DualCache(in_memory_cache=InMemoryCache())
    spend_counter_cache.in_memory_cache.set_cache(key=committed_spend_key("spend:key:hashed"), value="not-a-number")
    spend_counter_cache.in_memory_cache.set_cache(key=committed_spend_key("spend:team:t1"), value=4.0)

    committed: Final = await read_committed_budget_spend(
        spend_counter_cache,
        EntitySpendCounterKeys.of(token="hashed", team_id="t1", user_id=None, org_id=None, end_user_id=None),
    )

    assert committed == CommittedBudgetSpend()


@pytest.mark.parametrize(
    ("counter_keys", "expected"),
    [
        (EntitySpendCounterKeys(key="spend:key:hashed"), CommittedBudgetSpend(key=12.0)),
        (EntitySpendCounterKeys(end_user="spend:end_user:cust-1"), CommittedBudgetSpend(end_user=12.0)),
    ],
)
@pytest.mark.asyncio
async def test_seed_writes_the_db_spend_when_the_twin_is_absent(
    spend_counter_cache: DualCache, counter_keys: EntitySpendCounterKeys, expected: CommittedBudgetSpend
):
    (counter_key,) = counter_keys.tracked()

    seeded: Final = await seed_committed_spend(spend_counter_cache, _prisma_with_spend(12.0), counter_key)

    assert seeded is True
    assert await read_committed_budget_spend(spend_counter_cache, counter_keys) == expected


@pytest.mark.asyncio
async def test_seed_never_lowers_a_twin_another_replica_already_advanced(spend_counter_cache: DualCache):
    await _store(spend_counter_cache, committed_spend_key("spend:team:t1"), 20.0)

    seeded: Final = await seed_committed_spend(spend_counter_cache, _prisma_with_spend(12.0), "spend:team:t1")

    committed: Final = await read_committed_budget_spend(
        spend_counter_cache, EntitySpendCounterKeys(team="spend:team:t1")
    )
    assert seeded is True
    assert committed.team == 20.0


@pytest.mark.asyncio
async def test_read_without_any_counter_key_sends_redis_nothing():
    redis_cache: Final = AsyncMock(spec=RedisCache)
    spend_counter_cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)

    committed: Final = await read_committed_budget_spend(spend_counter_cache, EntitySpendCounterKeys())

    assert committed == CommittedBudgetSpend()
    redis_cache.async_batch_get_cache.assert_not_awaited()


class _UnreachableRedis(FakeAsyncRedis):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> bytes:
        raise ConnectionError("redis unreachable")


@pytest.mark.asyncio
async def test_seed_reports_no_twin_when_redis_rejects_the_write():
    spend_counter_cache: Final = DualCache(
        in_memory_cache=InMemoryCache(), redis_cache=_RedisCacheOnClient(_UnreachableRedis(server=FakeServer()))
    )

    seeded: Final = await seed_committed_spend(spend_counter_cache, _prisma_with_spend(12.0), "spend:team:t1")

    committed: Final = await read_committed_budget_spend(
        spend_counter_cache, EntitySpendCounterKeys(team="spend:team:t1")
    )
    assert seeded is False
    assert committed == CommittedBudgetSpend()


@pytest.mark.asyncio
async def test_seed_creates_no_twin_when_the_db_has_no_spend_for_the_entity(spend_counter_cache: DualCache):
    seeded: Final = await seed_committed_spend(spend_counter_cache, _prisma_with_spend(None), "spend:key:hashed")

    committed: Final = await read_committed_budget_spend(
        spend_counter_cache, EntitySpendCounterKeys(key="spend:key:hashed")
    )
    assert seeded is False
    assert committed == CommittedBudgetSpend()
