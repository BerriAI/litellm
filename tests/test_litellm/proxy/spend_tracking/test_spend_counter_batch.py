"""Exact Redis round-trip counts for the spend counters admission reads within one auth scope."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import (
    SpendCounterBatch,
    active_spend_counter_batch,
    admission_counter_keys,
    bind_admission_counter_keys,
    release_spend_counter_batch,
    spend_counter_batch_scope,
)

TOKEN = UserAPIKeyAuth(token="hashed", team_id="team", user_id="user", org_id="org")
TOKEN_KEYS = frozenset(
    {
        "spend:key:hashed",
        "spend:team:team",
        "spend:team_member:user:team",
        "spend:user:user",
        "spend:end_user:eu",
        "spend:org:org",
    }
)


class CountingRedis(RedisCache):
    def __init__(self, store: dict[str, object] | None = None, fail: bool = False) -> None:
        self.store: dict[str, object] = dict(store or {})
        self.fail = fail
        self.commands: list[str] = []

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        if self.fail:
            raise ConnectionError("redis down")
        self.commands.append(f"GET {key}")
        return self.store.get(key)

    async def async_batch_get_cache(self, key_list: Sequence[str], **kwargs: object) -> dict[str, object]:
        if self.fail:
            raise ConnectionError("redis down")
        self.commands.append(f"MGET {' '.join(key_list)}")
        return {key: self.store.get(key) for key in key_list}

    def get_ttl(self, **kwargs: object) -> int | None:
        return None

    async def async_increment(self, key: str, value: float, **kwargs: object) -> float:
        self.commands.append(f"INCRBYFLOAT {key} {value}")
        return self._incr(key, value)

    async def async_increment_pipeline(
        self, increment_list: Sequence[Mapping[str, object]], **kwargs: object
    ) -> list[float]:
        self.commands.append(f"PIPELINE {' '.join(str(op['key']) for op in increment_list)}")
        return [self._incr(str(op["key"]), float(str(op["increment_value"]))) for op in increment_list]

    def _incr(self, key: str, value: float) -> float:
        total = float(str(self.store.get(key, 0.0))) + value
        self.store[key] = total
        return total


def _spend_counter_cache(redis: RedisCache | None, in_memory: dict[str, float] | None = None) -> MagicMock:
    cache = MagicMock()
    cache.redis_cache = redis
    cache.in_memory_cache.get_cache = MagicMock(side_effect=lambda key: (in_memory or {}).get(key))
    return cache


def test_admission_counter_keys_cover_every_entity_the_checks_read():
    assert admission_counter_keys(TOKEN, end_user_id="eu") == TOKEN_KEYS
    assert admission_counter_keys(UserAPIKeyAuth(token="hashed"), end_user_id=None) == {"spend:key:hashed"}
    assert "spend:team_member:user:team" not in admission_counter_keys(
        UserAPIKeyAuth(token="hashed", user_id="user"), end_user_id=None
    )


@pytest.mark.asyncio
async def test_bound_counters_share_one_mget_and_a_clean_miss_is_authoritative():
    redis = CountingRedis({"spend:key:hashed": 1.5, "spend:team:team": 2.5})
    batch = SpendCounterBatch(redis)
    batch.bind(TOKEN_KEYS)

    reads = await asyncio.gather(*(batch.read(key) for key in sorted(TOKEN_KEYS)))

    assert len(redis.commands) == 1
    assert set(redis.commands[0].split()[1:]) == TOKEN_KEYS
    assert dict(zip(sorted(TOKEN_KEYS), reads)) == {
        "spend:end_user:eu": (None, True),
        "spend:key:hashed": (1.5, True),
        "spend:org:org": (None, True),
        "spend:team:team": (2.5, True),
        "spend:team_member:user:team": (None, True),
        "spend:user:user": (None, True),
    }


@pytest.mark.asyncio
async def test_unbound_counter_and_closed_batch_leave_the_read_to_the_caller():
    redis = CountingRedis({"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(redis)
    batch.bind(frozenset({"spend:key:hashed"}))

    assert await batch.read("spend:tag:prod") is None
    assert redis.commands == []
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.close()
    batch.bind(frozenset({"spend:org:org"}))
    assert await batch.read("spend:key:hashed") is None
    assert await batch.read("spend:org:org") is None
    assert len(redis.commands) == 1


@pytest.mark.asyncio
async def test_keys_bound_after_the_first_read_join_one_more_mget_for_only_the_new_keys():
    redis = CountingRedis({"spend:key:hashed": 1.0, "spend:org:org": 9.0})
    batch = SpendCounterBatch(redis)
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.bind(frozenset({"spend:org:org", "spend:key:hashed"}))
    assert await batch.read("spend:org:org") == (9.0, True)
    assert await batch.read("spend:key:hashed") == (1.0, True)

    assert redis.commands == ["MGET spend:key:hashed", "MGET spend:org:org"]


@pytest.mark.asyncio
async def test_a_recorded_write_result_answers_later_reads_without_another_redis_read():
    redis = CountingRedis({"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(redis)
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.record("spend:key:hashed", 3.5)
    batch.record("spend:org:org", 7.0)

    assert await batch.read("spend:key:hashed") == (3.5, True)
    assert await batch.read("spend:org:org") == (7.0, True)
    assert redis.commands == ["MGET spend:key:hashed"]


@pytest.mark.asyncio
async def test_a_forgotten_counter_is_read_fresh_from_redis_when_it_is_bound_again():
    redis = CountingRedis({"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(redis)
    batch.bind(frozenset({"spend:key:hashed"}))
    batch.record("spend:key:hashed", 3.5)

    batch.forget("spend:key:hashed")
    assert await batch.read("spend:key:hashed") is None

    redis.store["spend:key:hashed"] = 9.0
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (9.0, True)
    assert redis.commands == ["MGET spend:key:hashed"]


@pytest.mark.asyncio
async def test_failed_mget_hands_every_counter_back_to_the_caller():
    batch = SpendCounterBatch(CountingRedis(fail=True))
    batch.bind(TOKEN_KEYS)

    assert await batch.read("spend:key:hashed") is None


@pytest.mark.asyncio
async def test_non_numeric_counter_payload_hands_the_batch_back_to_the_caller():
    redis = CountingRedis()
    redis.store["spend:key:hashed"] = "garbage"
    batch = SpendCounterBatch(redis)
    batch.bind(frozenset({"spend:key:hashed"}))

    assert await batch.read("spend:key:hashed") is None


def test_scope_installs_a_batch_only_when_redis_exists_and_release_closes_without_clearing():
    assert active_spend_counter_batch() is None
    with spend_counter_batch_scope(None):
        assert active_spend_counter_batch() is None
        bind_admission_counter_keys(TOKEN, end_user_id=None)

    with spend_counter_batch_scope(CountingRedis()):
        batch = active_spend_counter_batch()
        assert batch is not None
        bind_admission_counter_keys(TOKEN, end_user_id="eu")
        assert batch.counter_keys == TOKEN_KEYS
        release_spend_counter_batch()
        assert active_spend_counter_batch() is batch
        batch.bind(frozenset({"spend:tag:x"}))
        assert batch.counter_keys == TOKEN_KEYS
    assert active_spend_counter_batch() is None


@pytest.mark.asyncio
async def test_get_current_spend_inside_the_scope_costs_one_mget_for_all_admission_counters(monkeypatch):
    redis = CountingRedis({"spend:key:hashed": 3.0, "spend:team:team": 4.0, "spend:org:org": 5.0})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", None)

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id="eu")
        key_spend = await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0)
        team_spend = await ps.get_current_spend(counter_key="spend:team:team", fallback_spend=0.0)
        org_spend = await ps.get_current_spend(counter_key="spend:org:org", fallback_spend=0.0)
        user_spend = await ps.get_current_spend(counter_key="spend:user:user", fallback_spend=7.0)

    assert (key_spend, team_spend, org_spend, user_spend) == (3.0, 4.0, 5.0, 7.0)
    assert [c for c in redis.commands if c.startswith("GET ")] == [], "the cold reseed reuses the MGET miss"
    assert [c for c in redis.commands if c.startswith("MGET ")] == [f"MGET {' '.join(sorted(TOKEN_KEYS))}"]


@pytest.mark.asyncio
async def test_get_current_spend_outside_the_scope_still_reads_redis_per_counter(monkeypatch):
    redis = CountingRedis({"spend:key:hashed": 3.0})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))

    assert await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0) == 3.0
    assert redis.commands == ["GET spend:key:hashed"]


@pytest.mark.asyncio
async def test_after_release_a_read_goes_to_redis_directly_so_read_then_write_sees_fresh_values(monkeypatch):
    redis = CountingRedis({"spend:key:hashed": 3.0})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (3.0, True)
        redis.store["spend:key:hashed"] = 8.0
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (3.0, True)
        release_spend_counter_batch()
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (8.0, True)

    assert [c.split()[0] for c in redis.commands] == ["MGET", "GET"]


@pytest.mark.asyncio
async def test_batched_clean_miss_does_not_fall_back_to_the_per_pod_in_memory_copy(monkeypatch):
    redis = CountingRedis()
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis, in_memory={"spend:key:hashed": 99.0}))

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (None, True)


@pytest.mark.asyncio
async def test_batched_redis_failure_falls_back_to_the_per_pod_in_memory_copy(monkeypatch):
    redis = CountingRedis(fail=True)
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis, in_memory={"spend:key:hashed": 99.0}))

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (99.0, False)


@pytest.mark.asyncio
async def test_scope_is_per_task_so_concurrent_requests_do_not_share_a_batch():
    redis = CountingRedis({"spend:key:a": 1.0, "spend:key:b": 2.0})

    async def request(token: str) -> tuple[float | None, bool] | None:
        with spend_counter_batch_scope(redis):
            bind_admission_counter_keys(UserAPIKeyAuth(token=token), end_user_id=None)
            batch = active_spend_counter_batch()
            assert batch is not None
            await asyncio.sleep(0)
            return await batch.read(f"spend:key:{token}")

    assert await asyncio.gather(request("a"), request("b")) == [(1.0, True), (2.0, True)]
    assert sorted(redis.commands) == ["MGET spend:key:a", "MGET spend:key:b"]


@pytest.mark.asyncio
async def test_batch_reads_never_touch_a_prisma_client_when_redis_answers(monkeypatch):
    redis = CountingRedis({"spend:key:hashed": 3.0})
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock()
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", prisma)

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0) == 3.0

    assert prisma.db.mock_calls == []


def _reseed_prisma(spend: float) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=MagicMock(spend=spend))
    return prisma


@pytest.mark.asyncio
async def test_reseed_reuses_the_admission_mget_instead_of_its_own_get():
    redis = CountingRedis({"spend:key:hashed": 7.5})
    prisma = _reseed_prisma(spend=1.0)
    cache = _spend_counter_cache(redis)

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        value = await SpendCounterReseed.coalesced(prisma, cache, counter_key="spend:key:hashed")

    assert value == 7.5
    assert redis.commands == [
        "MGET spend:key:hashed spend:org:org spend:team:team spend:team_member:user:team spend:user:user"
    ]
    assert prisma.db.mock_calls == []


@pytest.mark.asyncio
async def test_reseed_treats_a_batched_clean_miss_as_authoritative_and_seeds_from_the_db():
    redis = CountingRedis()
    redis.async_set_cache = AsyncMock(return_value=True)
    prisma = _reseed_prisma(spend=2.25)
    cache = _spend_counter_cache(redis, in_memory={"spend:key:hashed": 99.0})

    with spend_counter_batch_scope(redis):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        value = await SpendCounterReseed.coalesced(prisma, cache, counter_key="spend:key:hashed")

    assert value == 2.25
    assert [c for c in redis.commands if c.startswith("GET")] == []
    redis.async_set_cache.assert_awaited_once_with(key="spend:key:hashed", value=2.25, nx=True)


@pytest.mark.asyncio
async def test_reseed_outside_the_scope_still_re_checks_redis_itself():
    redis = CountingRedis({"spend:key:hashed": 4.0})

    value = await SpendCounterReseed.coalesced(
        _reseed_prisma(spend=1.0), _spend_counter_cache(redis), "spend:key:hashed"
    )

    assert value == 4.0
    assert redis.commands == ["GET spend:key:hashed"]


POST_CALL_KEYS = TOKEN_KEYS | {"spend:tag:prod", "spend:model_access_group:premium"}


@pytest.mark.asyncio
async def test_post_call_increment_for_every_entity_costs_one_mget_and_one_pipeline(monkeypatch):
    redis = CountingRedis({key: 1.0 for key in POST_CALL_KEYS})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", None)

    await ps.increment_spend_counters(
        token="hashed",
        team_id="team",
        user_id="user",
        org_id="org",
        end_user_id="eu",
        tags=["prod"],
        model_access_groups=["premium"],
        response_cost=0.5,
    )

    assert [c.split()[0] for c in redis.commands] == ["MGET", "PIPELINE"], redis.commands
    assert set(redis.commands[0].split()[1:]) == POST_CALL_KEYS
    assert set(redis.commands[1].split()[1:]) == POST_CALL_KEYS
    assert {key: redis.store[key] for key in POST_CALL_KEYS} == {key: 1.5 for key in POST_CALL_KEYS}


@pytest.mark.asyncio
async def test_post_call_cold_counters_seed_from_the_mget_miss_without_a_second_read(monkeypatch):
    redis = CountingRedis({"spend:key:hashed": 1.0})
    redis.async_set_cache = AsyncMock(return_value=True)
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=MagicMock(spend=4.0))
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", prisma)

    await ps.increment_spend_counters(token="hashed", team_id="team", user_id=None, response_cost=0.5)

    assert [c.split()[0] for c in redis.commands] == ["MGET", "PIPELINE"], redis.commands
    redis.async_set_cache.assert_awaited_once_with(key="spend:team:team", value=4.0, nx=True)
    assert redis.store["spend:key:hashed"] == 1.5


RESERVED_KEYS = frozenset(
    {"spend:key:hashed", "spend:team:team", "spend:team_member:user:team", "spend:end_user:eu", "spend:org:org"}
)


def _reservation(reserved_cost: float, counter_keys: frozenset[str] = RESERVED_KEYS) -> dict[str, object]:
    return {
        "reserved_cost": reserved_cost,
        "entries": [
            {"counter_key": key, "entity_type": "Key", "entity_id": key, "reserved_cost": reserved_cost}
            for key in sorted(counter_keys)
        ],
    }


@pytest.mark.asyncio
async def test_post_call_with_a_reservation_costs_one_mget_one_reconcile_pipeline_one_increment_pipeline(monkeypatch):
    redis = CountingRedis({key: 1.0 for key in POST_CALL_KEYS})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation = _reservation(reserved_cost=0.4)

    await ps.increment_spend_counters(
        token="hashed",
        team_id="team",
        user_id="user",
        org_id="org",
        end_user_id="eu",
        tags=["prod"],
        model_access_groups=["premium"],
        response_cost=0.5,
        budget_reservation=reservation,
    )

    assert [c.split()[0] for c in redis.commands] == ["MGET", "PIPELINE", "PIPELINE"], redis.commands
    assert set(redis.commands[0].split()[1:]) == POST_CALL_KEYS, "reconcile and warm checks share the MGET"
    assert set(redis.commands[1].split()[1:]) == RESERVED_KEYS
    assert set(redis.commands[2].split()[1:]) == POST_CALL_KEYS - RESERVED_KEYS
    assert {key: round(redis.store[key], 6) for key in POST_CALL_KEYS} == {
        key: (1.1 if key in RESERVED_KEYS else 1.5) for key in POST_CALL_KEYS
    }
    assert [round(entry["applied_adjustment"], 6) for entry in reservation["entries"]] == [0.1] * len(RESERVED_KEYS)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reconcile_settles_a_flushed_counter_on_its_own_after_the_shared_pipeline(monkeypatch):
    from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation

    redis = CountingRedis({key: 1.0 for key in RESERVED_KEYS - {"spend:team:team"}})
    redis.async_set_max = AsyncMock(return_value=4.0)
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=MagicMock(spend=4.0))
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", prisma)
    reservation = _reservation(reserved_cost=0.4)

    await reconcile_budget_reservation(budget_reservation=reservation, actual_cost=0.5)

    assert [c.split()[0] for c in redis.commands] == ["MGET", "PIPELINE", "INCRBYFLOAT"], redis.commands
    assert set(redis.commands[1].split()[1:]) == RESERVED_KEYS - {"spend:team:team"}
    assert redis.commands[2] == "INCRBYFLOAT spend:team:team 0.5"
    redis.async_set_max.assert_awaited_once()
    assert redis.async_set_max.await_args.kwargs["key"] == "spend:team:team"
    assert all(round(entry["applied_adjustment"], 6) == 0.1 for entry in reservation["entries"])


@pytest.mark.asyncio
async def test_pre_call_resize_against_an_inconsistent_counter_writes_nothing_and_denies(monkeypatch):
    from litellm.proxy.spend_tracking.budget_reservation import _resize_applied_reservation

    redis = CountingRedis({"spend:key:hashed": 1.0})
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", None)
    entries = _reservation(reserved_cost=0.4, counter_keys=frozenset({"spend:key:hashed", "spend:team:team"}))[
        "entries"
    ]

    with pytest.raises(RuntimeError, match="spend:team:team"):
        await _resize_applied_reservation(entries=entries, current_reserved_cost=0.4, new_reserved_cost=0.9)

    assert [c.split()[0] for c in redis.commands] == ["MGET"], redis.commands
    assert redis.store["spend:key:hashed"] == 1.0
    assert all("applied_adjustment" not in entry for entry in entries)


@pytest.mark.asyncio
async def test_a_failed_reconcile_pipeline_invalidates_every_reserved_counter_and_falls_back(monkeypatch):
    redis = CountingRedis({key: 1.0 for key in POST_CALL_KEYS})
    redis.async_delete_cache = AsyncMock()
    reconcile_pipeline_failed = False

    async def _pipeline(increment_list: Sequence[Mapping[str, object]], **kwargs: object) -> list[float]:
        nonlocal reconcile_pipeline_failed
        if not reconcile_pipeline_failed:
            reconcile_pipeline_failed = True
            raise ConnectionError("redis down")
        return await CountingRedis.async_increment_pipeline(redis, increment_list, **kwargs)

    redis.async_increment_pipeline = _pipeline  # pyright: ignore[reportAttributeAccessIssue]  # instance override
    monkeypatch.setattr(ps, "spend_counter_cache", _spend_counter_cache(redis))
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation = _reservation(reserved_cost=0.4)

    await ps.increment_spend_counters(
        token="hashed",
        team_id="team",
        user_id="user",
        org_id="org",
        end_user_id="eu",
        response_cost=0.5,
        budget_reservation=reservation,
    )

    assert {call.kwargs["key"] for call in redis.async_delete_cache.await_args_list} == RESERVED_KEYS
    assert all("applied_adjustment" not in entry for entry in reservation["entries"])
    assert redis.commands[-1].split()[0] == "PIPELINE"
    assert set(redis.commands[-1].split()[1:]) == RESERVED_KEYS | {"spend:user:user"}


def test_a_scope_opened_inside_an_open_scope_joins_its_batch_and_a_closed_one_gets_its_own():
    redis = CountingRedis()
    with spend_counter_batch_scope(redis, counter_keys=frozenset({"spend:key:a"})):
        outer = active_spend_counter_batch()
        assert outer is not None
        with spend_counter_batch_scope(redis, counter_keys=frozenset({"spend:key:b"})):
            assert active_spend_counter_batch() is outer
        assert outer.counter_keys == {"spend:key:a", "spend:key:b"}
        release_spend_counter_batch()
        with spend_counter_batch_scope(redis, counter_keys=frozenset({"spend:key:c"})):
            inner = active_spend_counter_batch()
            assert inner is not outer
            assert inner is not None and inner.counter_keys == {"spend:key:c"}
        assert active_spend_counter_batch() is outer
