"""Committed spend twins hold only charged cost, so the remaining budget gauges never count a budget hold."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy.spend_tracking.committed_spend import (
    bind_committed_admission_counters,
    drop_committed_spend,
    read_admission_committed_spend,
    read_committed_budget_spend,
    seed_committed_spend,
)
from litellm.proxy.spend_tracking.spend_counter_batch import (
    active_spend_counter_batch,
    read_batched_spend_counter,
    spend_counter_batch_scope,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.carried_budget_state import CommittedBudgetSpend, EntitySpendCounterKeys
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


def _no_side_effect() -> None:
    return None


class _FindUniqueTable:
    def __init__(self, row: SimpleNamespace | None, on_read: Callable[[], None]) -> None:
        self._row: Final = row
        self._on_read: Final = on_read
        self.reads = 0

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace | None:
        self.reads += 1
        await asyncio.sleep(0)
        self._on_read()
        return self._row


def _prisma_with_spend(spend: float | None, on_read: Callable[[], None] = _no_side_effect) -> SimpleNamespace:
    row: Final = None if spend is None else SimpleNamespace(spend=spend)
    return SimpleNamespace(
        db=SimpleNamespace(
            litellm_verificationtoken=_FindUniqueTable(row, on_read),
            litellm_teamtable=_FindUniqueTable(row, on_read),
            litellm_endusertable=_FindUniqueTable(row, on_read),
        )
    )


@pytest.fixture(params=[False, True], ids=["in_memory", "redis"])
def has_redis(request: pytest.FixtureRequest) -> bool:
    return request.param


def _spend_counters(has_redis: bool, counters: Mapping[str, float] = MappingProxyType({})) -> FakeSpendCounters:
    if has_redis:
        return FakeSpendCounters(redis=dict(counters))
    return FakeSpendCounters(redis=None, memory=dict(counters))


@pytest.mark.asyncio
async def test_read_returns_each_committed_twin_and_none_for_one_never_charged(has_redis: bool):
    spend: Final = _spend_counters(
        has_redis,
        {"spend:committed:key:hashed": 1.5, "spend:committed:end_user:cust-1": 0.25, "spend:team:t1": 99.0},
    )
    counter_keys: Final = EntitySpendCounterKeys.of(
        token="hashed", team_id="t1", user_id=None, org_id=None, end_user_id="cust-1"
    )

    committed: Final = await read_committed_budget_spend(spend.store(), counter_keys)

    assert committed == CommittedBudgetSpend(counter_keys=counter_keys, key=1.5, end_user=0.25)


@pytest.mark.asyncio
async def test_a_twin_holding_garbage_reads_as_unknown_without_hiding_the_other_twins():
    spend: Final = FakeSpendCounters(
        redis={"spend:committed:key:hashed": "not-a-number", "spend:committed:team:t1": 4.0}
    )
    counter_keys: Final = EntitySpendCounterKeys.of(
        token="hashed", team_id="t1", user_id=None, org_id=None, end_user_id=None
    )

    committed: Final = await read_committed_budget_spend(spend.store(), counter_keys)

    assert committed == CommittedBudgetSpend(counter_keys=counter_keys, team=4.0)


@pytest.mark.parametrize(
    ("counter_keys", "expected_spend"),
    [
        (EntitySpendCounterKeys(key="spend:key:hashed"), {"key": 12.0}),
        (EntitySpendCounterKeys(end_user="spend:end_user:cust-1"), {"end_user": 12.0}),
    ],
)
@pytest.mark.asyncio
async def test_seed_writes_the_db_spend_when_the_twin_is_absent(
    has_redis: bool, counter_keys: EntitySpendCounterKeys, expected_spend: dict[str, float]
):
    store: Final = _spend_counters(has_redis).store()
    (counter_key,) = counter_keys.tracked()

    seeded: Final = await seed_committed_spend(store, _prisma_with_spend(12.0), counter_key)

    assert seeded is True
    assert await read_committed_budget_spend(store, counter_keys) == CommittedBudgetSpend(
        counter_keys=counter_keys, **expected_spend
    )


@pytest.mark.asyncio
async def test_concurrent_charges_on_a_cold_twin_read_the_db_once(has_redis: bool):
    store: Final = _spend_counters(has_redis).store()
    prisma_client: Final = _prisma_with_spend(12.0)
    counter_key: Final = f"spend:key:{uuid4()}"

    seeded: Final = await asyncio.gather(*(seed_committed_spend(store, prisma_client, counter_key) for _ in range(5)))

    assert seeded == [True] * 5
    assert prisma_client.db.litellm_verificationtoken.reads == 1


@pytest.mark.asyncio
async def test_seed_reports_no_twin_when_the_twin_cannot_be_read():
    async def unreachable(counter: str) -> None:
        raise ConnectionError("redis unreachable")

    spend: Final = FakeSpendCounters(read_hook=unreachable)

    assert await seed_committed_spend(spend.store(), _prisma_with_spend(12.0), "spend:team:t1") is False


@pytest.mark.asyncio
async def test_seed_never_lowers_a_twin_another_replica_already_advanced(has_redis: bool):
    store: Final = _spend_counters(has_redis, {"spend:committed:team:t1": 20.0}).store()

    seeded: Final = await seed_committed_spend(store, _prisma_with_spend(12.0), "spend:team:t1")

    committed: Final = await read_committed_budget_spend(store, EntitySpendCounterKeys(team="spend:team:t1"))
    assert seeded is True
    assert committed.team == 20.0


@pytest.mark.asyncio
async def test_seed_keeps_the_twin_another_worker_wrote_while_this_one_read_the_db():
    spend: Final = FakeSpendCounters(redis={})

    def another_worker_seeds_and_charges() -> None:
        assert spend.redis is not None
        spend.redis["spend:committed:team:t1"] = 20.0

    seeded: Final = await seed_committed_spend(
        spend.store(), _prisma_with_spend(12.0, on_read=another_worker_seeds_and_charges), "spend:team:t1"
    )

    committed: Final = await read_committed_budget_spend(spend.store(), EntitySpendCounterKeys(team="spend:team:t1"))
    assert seeded is True
    assert committed.team == 20.0


@pytest.mark.asyncio
async def test_read_without_any_counter_key_sends_redis_nothing():
    spend: Final = FakeSpendCounters(redis={})

    committed: Final = await read_committed_budget_spend(spend.store(), EntitySpendCounterKeys())

    assert committed == CommittedBudgetSpend()
    assert spend.commands == []


@pytest.mark.asyncio
async def test_seed_reports_no_twin_when_redis_rejects_the_write():
    spend: Final = FakeSpendCounters(redis={}, fail=True)

    seeded: Final = await seed_committed_spend(spend.store(), _prisma_with_spend(12.0), "spend:team:t1")

    committed: Final = await read_committed_budget_spend(spend.store(), EntitySpendCounterKeys(team="spend:team:t1"))
    assert seeded is False
    assert committed.team is None


@pytest.mark.asyncio
async def test_seed_creates_no_twin_when_the_db_has_no_spend_for_the_entity(has_redis: bool):
    store: Final = _spend_counters(has_redis).store()

    seeded: Final = await seed_committed_spend(store, _prisma_with_spend(None), "spend:key:hashed")

    committed: Final = await read_committed_budget_spend(store, EntitySpendCounterKeys(key="spend:key:hashed"))
    assert seeded is False
    assert committed.key is None


@pytest.mark.asyncio
async def test_admission_read_rides_the_admission_mget_with_the_enforcement_counters(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "callbacks", [MagicMock(spec=PrometheusLogger)])
    spend: Final = FakeSpendCounters(redis={"spend:committed:key:hashed": 1.5})
    store: Final = spend.store()
    token: Final = UserAPIKeyAuth(token="hashed", team_id="t1")
    counter_keys: Final = EntitySpendCounterKeys(key="spend:key:hashed", team="spend:team:t1")

    with spend_counter_batch_scope(store, counter_keys=frozenset(("spend:key:hashed",))):
        bind_committed_admission_counters(token, end_user_id=None)
        await read_batched_spend_counter("spend:key:hashed")
        committed: Final = await read_admission_committed_spend(store, counter_keys)

    assert committed == CommittedBudgetSpend(counter_keys=counter_keys, key=1.5)
    assert len(spend.commands) == 1
    assert {"spend:committed:key:hashed", "spend:committed:team:t1"} <= set(spend.commands[0].split()[1:])


@pytest.mark.asyncio
async def test_admission_read_outside_a_batch_leaves_redis_alone_and_the_spend_unknown():
    spend: Final = FakeSpendCounters(redis={"spend:committed:key:hashed": 1.5})
    counter_keys: Final = EntitySpendCounterKeys(key="spend:key:hashed")

    committed: Final = await read_admission_committed_spend(spend.store(), counter_keys)

    assert committed == CommittedBudgetSpend(counter_keys=counter_keys)
    assert spend.commands == []


@pytest.mark.asyncio
async def test_admission_read_without_redis_reads_the_in_memory_twin():
    spend: Final = FakeSpendCounters(redis=None, memory={"spend:committed:key:hashed": 1.5})

    committed: Final = await read_admission_committed_spend(
        spend.store(), EntitySpendCounterKeys(key="spend:key:hashed")
    )

    assert committed.key == 1.5


def test_no_committed_counter_is_bound_without_a_prometheus_logger():
    with spend_counter_batch_scope(FakeSpendCounters(redis={}).store()):
        bind_committed_admission_counters(UserAPIKeyAuth(token="hashed"), end_user_id=None)
        batch: Final = active_spend_counter_batch()
        assert batch is not None
        bound: Final = batch.counter_keys

    assert bound == frozenset()


@pytest.mark.asyncio
async def test_drop_removes_the_twin_and_leaves_the_enforcement_counter(has_redis: bool):
    store: Final = _spend_counters(has_redis, {"spend:committed:org:o1": 5.0, "spend:org:o1": 5.0}).store()
    counter_keys: Final = EntitySpendCounterKeys(org="spend:org:o1")

    await drop_committed_spend(store, ("spend:org:o1",))

    assert (await read_committed_budget_spend(store, counter_keys)).org is None
    assert (await store.read("spend:org:o1"))[0] == 5.0


@pytest.mark.asyncio
async def test_a_failed_drop_never_fails_the_write_that_already_committed():
    spend: Final = FakeSpendCounters(redis={"spend:committed:org:o1": 5.0}, failing=frozenset({"delete"}))

    await drop_committed_spend(spend.store(), ("spend:org:o1",))

    assert spend.deleted() == frozenset({"spend:committed:org:o1"})
