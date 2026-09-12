"""Exact Redis round-trip counts for the spend counters admission reads within one auth scope."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
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
