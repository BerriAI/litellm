from collections import Counter
from collections.abc import Sequence
from typing import Final
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.committed_spend import committed_spend_key
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.spend_counters import user_window_counter
from litellm.rust_bridge._native import NativeRedisStoreError
from litellm.types.proxy.carried_budget_state import EntitySpendCounterKeys
from tests.unit.proxy.proxy_server import test_spend_counters as scenarios
from tests.unit.proxy.spend_counters_ledger_twin import (
    RedisCalls,
    SlowIncrementsRedis,
    force_ledger_windows,
    ledger_scenarios,
    rust_memory_store,
    rust_redis_store,
)
from tests.unit.proxy.spend_tracking.native_budgets_replica import RespRedis, native_reservation

# The Rust step warms every counter with one MGET and never reads a counter on its own, so the per-counter read hook
# this scenario pins concurrency with has nothing to observe. tests/charge.rs pins the concurrent scopes in Rust
_FAKE_ONLY_SCENARIOS: Final = frozenset({"test_increment_spend_counters_runs_scopes_concurrently"})

globals().update(ledger_scenarios(scenarios, lambda name: name not in _FAKE_ONLY_SCENARIOS))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)


# The scenarios below are the original ones whose store was FakeSpendCounters, redefined under the same name on the
# Rust store the ledger's counter step runs on. Each keeps the original's assertions that the Rust store can observe


class _EmptyUserCache:
    """An auth cache that holds nothing and records the keys read from it."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    async def async_get_cache(self, key: str, **_: object) -> None:
        self.reads.append(key)

    async def async_batch_get_cache(self, keys: list[str], **_: object) -> list[None]:
        self.reads.extend(keys)
        return [None for _ in keys]

    async def get_team_fields(self, key: AuthObject, fields: Sequence[str] | None = None, **_: object) -> None:
        self.reads.append(key.key)

    async def async_set_cache_pipeline(self, *_: object, **__: object) -> None:
        return None


def _reads_on_the_wire(redis: RespRedis) -> list[int]:
    commands: Final = [command[0].upper() for command in redis.commands if command[0].upper() != "CLIENT"]
    return [position for position, name in enumerate(commands) if name == "MGET"]


def _increments(redis: RespRedis) -> set[str]:
    return {command[1] for command in redis.commands if command[0].upper() == "INCRBYFLOAT"}


@pytest.mark.asyncio
async def test_increment_spend_counters_increments_all_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    redis: Final = RespRedis(values={})
    calls: Final = RedisCalls()
    user_cache: Final = _EmptyUserCache()
    monkeypatch.setattr(ps, "user_api_key_cache", user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis, calls):
        await ps.increment_spend_counters(token="hashed-tok", team_id="t1", user_id="u1", response_cost=5.0)
        reported: Final = await calls.reported(2)

    keys: Final = {"spend:key:hashed-tok", "spend:team:t1", "spend:team_member:u1:t1", "spend:user:u1"}
    assert Counter[str](reported) == Counter[str]({"MGET": 1, "pipeline": 1})
    assert _reads_on_the_wire(redis) == [0]
    assert _increments(redis) == keys
    assert redis.values == dict.fromkeys(keys, 5.0)
    assert user_cache.reads != [], "the step reads the budget windows from the auth cache"


@pytest.mark.asyncio
async def test_user_spend_increments_both_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    redis: Final = RespRedis(
        values={"spend:user:u": 0.0, user_window_counter("u", "1h"): 0.0, user_window_counter("u", "30d"): 0.0}
    )
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=AuthObject.user("u"),
        value=LiteLLM_UserTable.model_validate(
            {
                "user_id": "u",
                "budget_limits": [
                    {"budget_duration": "1h", "max_budget": 5.0, "reset_at": "2026-08-01T01:00:00Z"},
                    {"budget_duration": "30d", "max_budget": 100.0, "reset_at": "2026-08-31T00:00:00Z"},
                ],
            }
        ),
    )
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis):
        await ps.increment_spend_counters(token=None, team_id=None, user_id="u", response_cost=2.0)

    assert redis.values == {
        "spend:user:u": 2.0,
        user_window_counter("u", "1h"): 2.0,
        user_window_counter("u", "30d"): 2.0,
    }
    windows: Final = await ps.accounting_window_keys(None, None, user_id="u")
    assert {window["window_duration"] for window in windows} == {"1h", "30d"}


@pytest.mark.asyncio
async def test_increment_spend_counters_skips_reserved_counter_keys(
    monkeypatch: pytest.MonkeyPatch, native_budgets_on: None
) -> None:
    """The reserved key and org counters hold the reservation, which the charge reconciles instead of adding to."""
    store: Final = rust_memory_store(monkeypatch)
    monkeypatch.setattr(ps, "user_api_key_cache", _EmptyUserCache())
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation: Final = await native_reservation(
        store,
        UserAPIKeyAuth(token="hashed-tok"),
        [
            {"counter_key": "spend:key:hashed-tok", "entity_type": "Key", "entity_id": "hashed-tok"},
            {"counter_key": "spend:org:org1", "entity_type": "Organization", "entity_id": "org1"},
        ],
        5.0,
    )

    await ps.increment_spend_counters(
        token="hashed-tok",
        team_id="t1",
        user_id="u1",
        org_id="org1",
        end_user_id="eu1",
        tags=["a"],
        response_cost=5.0,
        budget_reservation=reservation,
    )

    assert reservation["finalized"] is True
    assert store.snapshot() == {
        "spend:key:hashed-tok": 5.0,
        "spend:org:org1": 5.0,
        "spend:team:t1": 5.0,
        "spend:team_member:u1:t1": 5.0,
        "spend:user:u1": 5.0,
        "spend:end_user:eu1": 5.0,
        "spend:tag:a": 5.0,
    }


@pytest.mark.asyncio
async def test_increment_spend_counters_failing_scope_propagates_after_siblings_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The team counter's stored spend read fails. The error reaches the caller, the reservation stays open, and every
    other counter is still charged in the one pipeline."""

    async def from_db(prisma_client: object, counter_key: str) -> float | None:
        if counter_key == "spend:team:t1":
            raise RuntimeError("reseed failed")
        return None

    keys: Final = (
        "spend:key:hashed-tok",
        "spend:team_member:u1:t1",
        "spend:user:u1",
        "spend:end_user:eu1",
        "spend:tag:a",
        "spend:org:org1",
    )
    redis: Final = RespRedis(values=dict.fromkeys(keys, 1.0))
    calls: Final = RedisCalls()
    monkeypatch.setattr(ps, "user_api_key_cache", _EmptyUserCache())
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(side_effect=from_db))
    reservation: Final[dict[str, object]] = {"finalized": False}

    with rust_redis_store(monkeypatch, redis, calls):
        with pytest.raises(RuntimeError, match="reseed failed"):
            await ps.increment_spend_counters(
                token="hashed-tok",
                team_id="t1",
                user_id="u1",
                org_id="org1",
                end_user_id="eu1",
                tags=["a"],
                response_cost=5.0,
                budget_reservation=reservation,
            )
        reported: Final = await calls.reported(2)

    assert reservation["finalized"] is False
    assert Counter[str](reported) == Counter[str]({"MGET": 1, "pipeline": 1})
    assert _reads_on_the_wire(redis) == [0]
    assert redis.values == dict.fromkeys(keys, 6.0)


@pytest.mark.asyncio
async def test_increment_spend_counters_zero_cost_is_noop_finalizes_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = RespRedis(values={})
    monkeypatch.setattr(ps, "user_api_key_cache", _EmptyUserCache())
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation: Final[dict[str, object]] = {"finalized": False}

    with rust_redis_store(monkeypatch, redis):
        await ps.increment_spend_counters(
            token="hashed-tok", team_id="t1", user_id="u1", response_cost=0, budget_reservation=reservation
        )

    assert reservation == {"finalized": True}
    assert redis.commands == []


_PENDING_SPEND_COUNTER_KEYS: Final = (
    "spend:key:hashed",
    "spend:team:team-1",
    "spend:team_member:user-1:team-1",
    "spend:user:user-1",
    "spend:end_user:eu-1",
    "spend:tag:tag-a",
    "spend:tag:tag-b",
    "spend:org:org-1",
)


async def _increment_every_spend_counter() -> None:
    await ps.increment_spend_counters(
        token="hashed",
        team_id="team-1",
        user_id="user-1",
        response_cost=0.5,
        org_id="org-1",
        end_user_id="eu-1",
        tags=["tag-a", "tag-b"],
    )


@pytest.mark.asyncio
async def test_increment_spend_counters_pipelines_all_scopes_in_one_redis_call(monkeypatch: pytest.MonkeyPatch) -> None:
    redis: Final = RespRedis(values=dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.0))
    calls: Final = RedisCalls()
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis, calls) as store:
        await _increment_every_spend_counter()
        reported: Final = await calls.reported(2)
        memory: Final = store.snapshot()

    assert Counter[str](reported) == Counter[str]({"MGET": 1, "pipeline": 1})
    assert _reads_on_the_wire(redis) == [0]
    assert _increments(redis) == set(_PENDING_SPEND_COUNTER_KEYS)
    assert memory == dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.5)


@pytest.mark.asyncio
async def test_increment_spend_counters_pipeline_failure_invalidates_all_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing pipeline invalidates every pending counter, in Redis and in memory, and the error propagates."""
    redis: Final = RespRedis(values=dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.0), failing=frozenset({"INCRBYFLOAT"}))
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis) as store:
        for counter in _PENDING_SPEND_COUNTER_KEYS:
            store.set_local(counter, 1.0)
        with pytest.raises(NativeRedisStoreError):
            await _increment_every_spend_counter()
        memory: Final = store.snapshot()

    assert [
        command[0] for command in redis.commands if command[0] not in ("CLIENT", "INCRBYFLOAT", "EXPIRE", "DEL")
    ] == ["MGET"], "only the warm-check read runs besides the failed pipeline and the invalidation"
    assert redis.deleted() == set(_PENDING_SPEND_COUNTER_KEYS)
    assert redis.values == {}
    assert memory == {}


@pytest.mark.asyncio
async def test_increment_spend_counters_timeout_invalidates_all_counters_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline that timed out invalidates the counters in memory and the request still succeeds."""
    redis: Final = SlowIncrementsRedis(values=dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.0))
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis, socket_timeout=0.2) as store:
        for counter in _PENDING_SPEND_COUNTER_KEYS:
            store.set_local(counter, 1.0)
        await _increment_every_spend_counter()
        memory: Final = store.snapshot()

    assert memory == {}


class _VerificationTokenTable:
    def __init__(self, spend: float) -> None:
        self._spend: Final = spend
        self.lookups: list[dict[str, str]] = []

    async def find_unique(self, where: dict[str, str]) -> MagicMock:
        self.lookups.append(where)
        return MagicMock(token=where["token"], spend=self._spend)


def _prisma_with_tokens(tokens: _VerificationTokenTable) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(litellm_verificationtoken=tokens))


@pytest.mark.asyncio
async def test_a_charge_outside_the_committed_scope_keeps_no_twin_and_reads_no_stored_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store: Final = rust_memory_store(monkeypatch, {"spend:key:hashed": 10.0})
    tokens: Final = _VerificationTokenTable(10.0)
    key_cache: Final = UserApiKeyCache()
    key_cache.set_cache(key=AuthObject.virtual_key("hashed"), value=UserAPIKeyAuth(token="hashed"))
    monkeypatch.setattr(ps, "user_api_key_cache", key_cache)
    monkeypatch.setattr(ps, "prisma_client", _prisma_with_tokens(tokens))

    await ps.increment_spend_counters(token="hashed", team_id=None, user_id=None, response_cost=0.5)

    assert store.snapshot() == {"spend:key:hashed": 10.5}
    assert tokens.lookups == []


@pytest.mark.asyncio
async def test_only_the_entities_in_the_committed_scope_take_the_charge_on_their_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store: Final = rust_memory_store(monkeypatch, {"spend:key:hashed": 10.0, "spend:team:t1": 10.0})
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(ps, "prisma_client", _prisma_with_tokens(_VerificationTokenTable(10.0)))

    await ps.increment_spend_counters(
        token="hashed",
        team_id="t1",
        user_id=None,
        response_cost=0.5,
        committed_spend_scope=EntitySpendCounterKeys(key="spend:key:hashed"),
    )

    assert store.snapshot().get(committed_spend_key("spend:key:hashed")) == 10.5
    assert committed_spend_key("spend:team:t1") not in store.snapshot()


@pytest.mark.asyncio
async def test_a_budgeted_team_whose_id_names_a_window_keeps_its_charge_and_gets_no_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_counter_key: Final = "spend:team:t1:window:1d"
    store: Final = rust_memory_store(monkeypatch, {team_counter_key: 1.0})
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(ps, "prisma_client", MagicMock())

    await ps.increment_spend_counters(
        token=None,
        team_id="t1:window:1d",
        user_id=None,
        response_cost=0.5,
        committed_spend_scope=EntitySpendCounterKeys(team=team_counter_key),
    )

    assert store.snapshot() == {team_counter_key: 1.5}
