"""Exact Redis round-trip counts for the spend counters admission reads within one auth scope."""

from __future__ import annotations

import asyncio
import gc
import warnings
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock
from weakref import ReferenceType, ref

import pytest
from starlette.requests import Request

import litellm
import litellm.proxy.proxy_server as ps
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.spend_counter_batch import (
    SpendCounterBatch,
    active_spend_counter_batch,
    admission_counter_keys,
    bind_admission_counter_keys,
    post_call_counter_keys,
    read_batched_spend_counters,
    release_spend_counter_batch,
    spend_counter_batch_scope,
)
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.caching.redis_batch_double import FakeNativePending, FakeNatives, fake_send
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

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


def _request_key_supplier(batch: SpendCounterBatch) -> ReferenceType[Request]:
    request: Final = Request({"type": "http", "headers": [(b"x-counter-key", b"spend:customer_org:org")]})
    batch.provide_counter_keys(lambda: frozenset((request.headers["x-counter-key"],)))
    return ref(request)


def test_consumed_counter_key_supplier_releases_its_request() -> None:
    batch: Final = SpendCounterBatch(FakeSpendCounters().store())
    request: Final = _request_key_supplier(batch)
    assert request() is not None

    batch.bind(frozenset(("spend:key:key",)))

    assert batch.counter_keys == frozenset(("spend:key:key", "spend:customer_org:org"))
    gc.collect()
    assert request() is None


def test_closed_batch_releases_an_unused_counter_key_supplier_request() -> None:
    batch: Final = SpendCounterBatch(FakeSpendCounters().store())
    request: Final = _request_key_supplier(batch)
    assert request() is not None

    batch.close()

    assert not batch.is_open
    gc.collect()
    assert request() is None


def test_admission_counter_keys_cover_every_entity_the_checks_read():
    assert admission_counter_keys(TOKEN, end_user_id="eu") == TOKEN_KEYS
    assert admission_counter_keys(UserAPIKeyAuth(token="hashed"), end_user_id=None) == {"spend:key:hashed"}
    assert "spend:team_member:user:team" not in admission_counter_keys(
        UserAPIKeyAuth(token="hashed", user_id="user"), end_user_id=None
    )


def test_post_call_counter_keys_skip_ids_that_are_not_strings():
    """A synthetic logging payload (batch cost polling, tests) can carry placeholders where the ids belong; those
    have no counter, and deriving the key set must never raise inside the cost callback."""
    placeholder = object()
    assert post_call_counter_keys(
        token=placeholder,  # pyright: ignore[reportArgumentType]  # synthetic payload placeholder, not an id
        team_id="team",
        user_id=None,
        org_id=placeholder,  # pyright: ignore[reportArgumentType]  # synthetic payload placeholder, not an id
        end_user_id="eu",
        tags=[placeholder, "t1"],
        model_access_groups=None,
    ) == {"spend:team:team", "spend:end_user:eu", "spend:tag:t1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("available", (False, True))
async def test_bulk_read_distinguishes_an_authoritative_miss_from_an_unavailable_batch(available: bool) -> None:
    handle: Final = FakeSpendCounters(redis={"spend:key:hashed": 1.5}, batch_reads=available)
    keys: Final = frozenset(("spend:key:hashed", "spend:team:team"))
    with spend_counter_batch_scope(handle.store()):
        assert await read_batched_spend_counters(frozenset()) == {}
        assert await read_batched_spend_counters(keys) == (
            {"spend:key:hashed": 1.5, "spend:team:team": None} if available else {}
        )
    assert await read_batched_spend_counters(keys) == {}


@pytest.mark.asyncio
async def test_bulk_reads_observe_recorded_writes_forgets_and_a_closed_scope() -> None:
    handle: Final = FakeSpendCounters(redis={"spend:key:hashed": 1.5})
    batch: Final = SpendCounterBatch(handle.store())
    known: Final = frozenset(("spend:key:hashed", "spend:team:team"))
    requested: Final = known | frozenset(("spend:user:unbound",))
    batch.bind(known)
    assert await batch.read_many(requested) == {"spend:key:hashed": 1.5, "spend:team:team": None}
    batch.record("spend:key:hashed", 2.0)
    batch.forget("spend:team:team")
    assert await batch.read_many(requested) == {"spend:key:hashed": 2.0}
    batch.close()
    assert await batch.read_many(requested) == {}
    assert handle.commands == ["MGET spend:key:hashed spend:team:team"]


@pytest.mark.asyncio
async def test_bound_counters_share_one_mget_and_a_clean_miss_is_authoritative():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.5, "spend:team:team": 2.5})
    batch = SpendCounterBatch(spend.store())
    batch.bind(TOKEN_KEYS)

    reads = await asyncio.gather(*(batch.read(key) for key in sorted(TOKEN_KEYS)))

    assert len(spend.commands) == 1
    assert set(spend.commands[0].split()[1:]) == TOKEN_KEYS
    assert dict(zip(sorted(TOKEN_KEYS), reads)) == {
        "spend:end_user:eu": (None, True),
        "spend:key:hashed": (1.5, True),
        "spend:org:org": (None, True),
        "spend:team:team": (2.5, True),
        "spend:team_member:user:team": (None, True),
        "spend:user:user": (None, True),
    }


@pytest.mark.asyncio
async def test_counters_bound_at_auth_join_the_native_pipeline_the_armed_cooldown_read_flushes():
    """One round trip: whichever native read the request awaits first flushes the native pre-call batch, and the
    batch's flush hook puts every bound counter on it before it goes out, so the budget read and the armed cooldown
    read leave together and the budget read costs no trip of its own."""
    natives = FakeNatives()
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})

    with (
        request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request,  # pyright: ignore[reportArgumentType]  # test double
        spend_counter_batch_scope(spend.store()),
    ):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        native = request.native()
        native.events.append("declare cooldowns")
        armed_cooldowns = request.native_result(FakeNativePending(native, lambda: ()))
        batch = active_spend_counter_batch()
        assert batch is not None

        assert await armed_cooldowns == ()
        budget = await batch.read("spend:key:hashed")

    assert budget == (3.0, True)
    assert native.events[:3] == ["declare cooldowns", "declare spend_counters", "flush_pre"]
    assert native.events.count("declare spend_counters") == 1
    assert [declared.counters for declared in spend.declared] == [tuple(sorted(TOKEN_KEYS - {"spend:end_user:eu"}))]
    assert spend.commands == [f"MGET {' '.join(sorted(TOKEN_KEYS - {'spend:end_user:eu'}))}"]


@pytest.mark.asyncio
async def test_counters_bound_at_auth_but_never_read_leave_no_coroutine_behind():
    """A key without budgets binds its admission counters, the armed cooldown read flushes them onto the pre-call
    pipeline, and no budget check ever reads them. The unread declaration must not warn when it is collected."""
    natives = FakeNatives()
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with (
            request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request,  # pyright: ignore[reportArgumentType]  # test double
            spend_counter_batch_scope(spend.store()),
        ):
            bind_admission_counter_keys(TOKEN, end_user_id=None)
            assert await request.native_result(FakeNativePending(request.native(), lambda: ())) == ()
        gc.collect()

    assert [declared.counters for declared in spend.declared] == [tuple(sorted(TOKEN_KEYS - {"spend:end_user:eu"}))]
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []


@pytest.mark.asyncio
async def test_counters_bound_at_auth_cost_no_redis_read_when_the_request_ends_before_reading_them():
    """A request denied between binding its admission counters and reading them closes its scope with the
    counters unread. Closing must not send them on a trip of their own that nothing will read."""
    natives = FakeNatives()
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})

    with (
        request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request,  # pyright: ignore[reportArgumentType]  # test double
        spend_counter_batch_scope(spend.store()),
    ):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        await request.flush_pre_call()

    assert spend.declared == []
    assert spend.commands == []


@pytest.mark.asyncio
async def test_counters_bound_after_the_native_batch_went_out_ride_the_next_one():
    natives = FakeNatives()
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0, "spend:org:org": 4.0})

    with (
        request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send),  # pyright: ignore[reportArgumentType]  # test double
        spend_counter_batch_scope(spend.store(), counter_keys=frozenset({"spend:key:hashed"})),
    ):
        batch = active_spend_counter_batch()
        assert batch is not None
        assert await batch.read("spend:key:hashed") == (3.0, True)
        batch.bind(frozenset({"spend:org:org"}))
        assert await batch.read("spend:org:org") == (4.0, True)

    assert [declared.counters for declared in spend.declared] == [("spend:key:hashed",), ("spend:org:org",)]


@pytest.mark.asyncio
async def test_unbound_counter_and_closed_batch_leave_the_read_to_the_caller():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(spend.store())
    batch.bind(frozenset({"spend:key:hashed"}))

    assert await batch.read("spend:tag:prod") is None
    assert spend.commands == []
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.close()
    batch.bind(frozenset({"spend:org:org"}))
    assert await batch.read("spend:key:hashed") is None
    assert await batch.read("spend:org:org") is None
    assert len(spend.commands) == 1


@pytest.mark.asyncio
async def test_keys_bound_after_the_first_read_join_one_more_mget_for_only_the_new_keys():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0, "spend:org:org": 9.0})
    batch = SpendCounterBatch(spend.store())
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.bind(frozenset({"spend:org:org", "spend:key:hashed"}))
    assert await batch.read("spend:org:org") == (9.0, True)
    assert await batch.read("spend:key:hashed") == (1.0, True)

    assert spend.commands == ["MGET spend:key:hashed", "MGET spend:org:org"]


@pytest.mark.asyncio
async def test_a_recorded_write_result_answers_later_reads_without_another_redis_read():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(spend.store())
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (1.0, True)

    batch.record("spend:key:hashed", 3.5)
    batch.record("spend:org:org", 7.0)

    assert await batch.read("spend:key:hashed") == (3.5, True)
    assert await batch.read("spend:org:org") == (7.0, True)
    assert spend.commands == ["MGET spend:key:hashed"]


@pytest.mark.asyncio
async def test_a_forgotten_counter_is_read_fresh_from_redis_when_it_is_bound_again():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0})
    batch = SpendCounterBatch(spend.store())
    batch.bind(frozenset({"spend:key:hashed"}))
    batch.record("spend:key:hashed", 3.5)

    batch.forget("spend:key:hashed")
    assert await batch.read("spend:key:hashed") is None

    spend.redis["spend:key:hashed"] = 9.0
    batch.bind(frozenset({"spend:key:hashed"}))
    assert await batch.read("spend:key:hashed") == (9.0, True)
    assert spend.commands == ["MGET spend:key:hashed"]


@pytest.mark.asyncio
async def test_failed_mget_hands_every_counter_back_to_the_caller():
    batch = SpendCounterBatch(FakeSpendCounters(fail=True).store())
    batch.bind(TOKEN_KEYS)

    assert await batch.read("spend:key:hashed") is None


@pytest.mark.asyncio
async def test_non_numeric_counter_payload_hands_the_batch_back_to_the_caller():
    spend = FakeSpendCounters()
    spend.redis["spend:key:hashed"] = "garbage"
    batch = SpendCounterBatch(spend.store())
    batch.bind(frozenset({"spend:key:hashed"}))

    assert await batch.read("spend:key:hashed") is None


def test_scope_installs_a_batch_only_when_redis_exists_and_release_closes_without_clearing():
    assert active_spend_counter_batch() is None
    with spend_counter_batch_scope(FakeSpendCounters(redis=None).store()):
        assert active_spend_counter_batch() is None
        bind_admission_counter_keys(TOKEN, end_user_id=None)

    with spend_counter_batch_scope(FakeSpendCounters().store()):
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
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0, "spend:team:team": 4.0, "spend:org:org": 5.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id="eu")
        key_spend = await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0)
        team_spend = await ps.get_current_spend(counter_key="spend:team:team", fallback_spend=0.0)
        org_spend = await ps.get_current_spend(counter_key="spend:org:org", fallback_spend=0.0)
        user_spend = await ps.get_current_spend(counter_key="spend:user:user", fallback_spend=7.0)

    assert (key_spend, team_spend, org_spend, user_spend) == (3.0, 4.0, 5.0, 7.0)
    assert [c for c in spend.commands if c.startswith("GET ")] == [], "the cold reseed reuses the MGET miss"
    assert [c for c in spend.commands if c.startswith("MGET ")] == [f"MGET {' '.join(sorted(TOKEN_KEYS))}"]


@pytest.mark.asyncio
async def test_get_current_spend_outside_the_scope_still_reads_redis_per_counter(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())

    assert await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0) == 3.0
    assert spend.commands == ["GET spend:key:hashed"]


@pytest.mark.asyncio
async def test_after_release_a_read_goes_to_redis_directly_so_read_then_write_sees_fresh_values(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (3.0, True)
        spend.redis["spend:key:hashed"] = 8.0
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (3.0, True)
        release_spend_counter_batch()
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (8.0, True)

    assert [c.split()[0] for c in spend.commands] == ["MGET", "GET"]


@pytest.mark.asyncio
async def test_batched_clean_miss_does_not_fall_back_to_the_per_pod_in_memory_copy(monkeypatch):
    spend = FakeSpendCounters(memory={"spend:key:hashed": 99.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (None, True)


@pytest.mark.asyncio
async def test_batched_redis_failure_falls_back_to_the_per_pod_in_memory_copy(monkeypatch):
    spend = FakeSpendCounters(fail=True, memory={"spend:key:hashed": 99.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.read_spend_counter_cache_value("spend:key:hashed") == (99.0, False)


@pytest.mark.asyncio
async def test_scope_is_per_task_so_concurrent_requests_do_not_share_a_batch():
    spend = FakeSpendCounters(redis={"spend:key:a": 1.0, "spend:key:b": 2.0})

    async def request(token: str) -> tuple[float | None, bool] | None:
        with spend_counter_batch_scope(spend.store()):
            bind_admission_counter_keys(UserAPIKeyAuth(token=token), end_user_id=None)
            batch = active_spend_counter_batch()
            assert batch is not None
            await asyncio.sleep(0)
            return await batch.read(f"spend:key:{token}")

    assert await asyncio.gather(request("a"), request("b")) == [(1.0, True), (2.0, True)]
    assert sorted(spend.commands) == ["MGET spend:key:a", "MGET spend:key:b"]


@pytest.mark.asyncio
async def test_batch_reads_never_touch_a_prisma_client_when_redis_answers(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 3.0})
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock()
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", prisma)

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        assert await ps.get_current_spend(counter_key="spend:key:hashed", fallback_spend=0.0) == 3.0

    assert prisma.db.mock_calls == []


def _reseed_prisma(spend: float) -> MagicMock:
    prisma = MagicMock()
    prisma.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=MagicMock(spend=spend))
    return prisma


@pytest.mark.asyncio
async def test_reseed_reuses_the_admission_mget_instead_of_its_own_get():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 7.5})
    prisma = _reseed_prisma(spend=1.0)
    cache = spend.store()

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        value = await SpendCounterReseed.coalesced(prisma, cache, counter_key="spend:key:hashed")

    assert value == 7.5
    assert spend.commands == [
        "MGET spend:key:hashed spend:org:org spend:team:team spend:team_member:user:team spend:user:user"
    ]
    assert prisma.db.mock_calls == []


@pytest.mark.asyncio
async def test_reseed_treats_a_batched_clean_miss_as_authoritative_and_seeds_from_the_db():
    spend = FakeSpendCounters(memory={"spend:key:hashed": 99.0})
    prisma = _reseed_prisma(spend=2.25)
    cache = spend.store()

    with spend_counter_batch_scope(spend.store()):
        bind_admission_counter_keys(TOKEN, end_user_id=None)
        value = await SpendCounterReseed.coalesced(prisma, cache, counter_key="spend:key:hashed")

    assert value == 2.25
    assert [c for c in spend.commands if c.startswith("GET")] == []
    assert [c for c in spend.commands if c.startswith("SETNX")] == ["SETNX spend:key:hashed 2.25"]


@pytest.mark.asyncio
async def test_reseed_outside_the_scope_still_re_checks_redis_itself():
    spend = FakeSpendCounters(redis={"spend:key:hashed": 4.0})

    value = await SpendCounterReseed.coalesced(_reseed_prisma(spend=1.0), spend.store(), "spend:key:hashed")

    assert value == 4.0
    assert spend.commands == ["GET spend:key:hashed"]


POST_CALL_KEYS = TOKEN_KEYS | {"spend:tag:prod", "spend:model_access_group:premium"}


@pytest.mark.asyncio
async def test_post_call_increment_for_every_entity_costs_one_mget_and_one_pipeline(monkeypatch):
    spend = FakeSpendCounters(redis={key: 1.0 for key in POST_CALL_KEYS})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
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

    assert [c.split()[0] for c in spend.commands] == ["MGET", "PIPELINE"], spend.commands
    assert set(spend.commands[0].split()[1:]) == POST_CALL_KEYS
    assert set(spend.commands[1].split()[1:]) == POST_CALL_KEYS
    assert {key: spend.redis[key] for key in POST_CALL_KEYS} == {key: 1.5 for key in POST_CALL_KEYS}


@pytest.mark.asyncio
async def test_post_call_cold_counters_seed_from_the_mget_miss_without_a_second_read(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0})
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=MagicMock(spend=4.0))
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", prisma)

    await ps.increment_spend_counters(token="hashed", team_id="team", user_id=None, response_cost=0.5)

    assert [c.split()[0] for c in spend.commands] == ["MGET", "SETNX", "PIPELINE"], spend.commands
    assert spend.commands[1] == "SETNX spend:team:team 4.0"
    assert spend.redis["spend:key:hashed"] == 1.5


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
async def test_post_call_with_a_reservation_costs_one_mget_and_one_pipeline_for_reconcile_and_increments(monkeypatch):
    spend = FakeSpendCounters(redis={key: 1.0 for key in POST_CALL_KEYS})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
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

    assert [c.split()[0] for c in spend.commands] == ["MGET", "PIPELINE"], spend.commands
    assert set(spend.commands[0].split()[1:]) == POST_CALL_KEYS, "reconcile and warm checks share the MGET"
    assert set(spend.commands[1].split()[1:]) == POST_CALL_KEYS, "reconcile adjustments ride the increment pipeline"
    assert {key: round(spend.redis[key], 6) for key in POST_CALL_KEYS} == {
        key: (1.1 if key in RESERVED_KEYS else 1.5) for key in POST_CALL_KEYS
    }
    assert [round(entry["applied_adjustment"], 6) for entry in reservation["entries"]] == [0.1] * len(RESERVED_KEYS)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_a_stale_counter_repair_updates_the_open_batch_instead_of_forcing_a_second_mget(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0, "spend:team:team": 1.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())

    with spend_counter_batch_scope(spend.store(), counter_keys=frozenset({"spend:key:hashed", "spend:team:team"})):
        assert await ps.read_spend_counter_cache_value(counter_key="spend:team:team") == (1.0, True)
        await ps._repair_stale_spend_counter(counter_key="spend:team:team", db_spend=4.0)
        assert await ps.read_spend_counter_cache_value(counter_key="spend:team:team") == (4.0, True)
        assert await ps.read_spend_counter_cache_value(counter_key="spend:key:hashed") == (1.0, True)

    assert [c.split()[0] for c in spend.commands] == ["MGET", "SETMAX"], spend.commands


@pytest.mark.asyncio
async def test_reconcile_settles_a_flushed_counter_on_its_own_after_the_shared_pipeline(monkeypatch):
    from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation

    spend = FakeSpendCounters(redis={key: 1.0 for key in RESERVED_KEYS - {"spend:team:team"}})
    prisma = MagicMock()
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=MagicMock(spend=4.0))
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", prisma)
    reservation = _reservation(reserved_cost=0.4)

    await reconcile_budget_reservation(budget_reservation=reservation, actual_cost=0.5)

    assert [c.split()[0] for c in spend.commands] == ["MGET", "PIPELINE", "SETMAX", "INCRBYFLOAT"], spend.commands
    assert set(spend.commands[1].split()[1:]) == RESERVED_KEYS - {"spend:team:team"}
    assert spend.commands[2] == "SETMAX spend:team:team 4.0"
    assert spend.commands[3] == "INCRBYFLOAT spend:team:team 0.5"
    assert all(round(entry["applied_adjustment"], 6) == 0.1 for entry in reservation["entries"])


@pytest.mark.asyncio
async def test_pre_call_resize_against_an_inconsistent_counter_writes_nothing_and_denies(monkeypatch):
    from litellm.proxy.spend_tracking.budget_reservation import _resize_applied_reservation

    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    entries = _reservation(reserved_cost=0.4, counter_keys=frozenset({"spend:key:hashed", "spend:team:team"}))[
        "entries"
    ]

    with pytest.raises(RuntimeError, match="spend:team:team"):
        await _resize_applied_reservation(entries=entries, current_reserved_cost=0.4, new_reserved_cost=0.9)

    assert [c.split()[0] for c in spend.commands] == ["MGET"], spend.commands
    assert spend.redis["spend:key:hashed"] == 1.0
    assert all("applied_adjustment" not in entry for entry in entries)


@pytest.mark.asyncio
async def test_a_failed_post_call_pipeline_invalidates_every_counter_it_carried_and_stamps_nothing(monkeypatch):
    spend = FakeSpendCounters(redis={key: 1.0 for key in POST_CALL_KEYS}, fail_pipelines=True)
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation = _reservation(reserved_cost=0.4)

    with pytest.raises(NativeRedisStoreError):
        await ps.increment_spend_counters(
            token="hashed",
            team_id="team",
            user_id="user",
            org_id="org",
            end_user_id="eu",
            response_cost=0.5,
            budget_reservation=reservation,
        )

    assert [c.split()[0] for c in spend.commands if not c.startswith("DEL ")] == ["MGET"], spend.commands
    assert {c.split()[1] for c in spend.commands if c.startswith("DEL ")} == RESERVED_KEYS | {"spend:user:user"}
    assert all("applied_adjustment" not in entry for entry in reservation["entries"])
    assert spend.redis == {"spend:tag:prod": 1.0, "spend:model_access_group:premium": 1.0}


def test_a_scope_opened_inside_an_open_scope_joins_its_batch_and_a_closed_one_gets_its_own():
    spend = FakeSpendCounters()
    with spend_counter_batch_scope(spend.store(), counter_keys=frozenset({"spend:key:a"})):
        outer = active_spend_counter_batch()
        assert outer is not None
        with spend_counter_batch_scope(spend.store(), counter_keys=frozenset({"spend:key:b"})):
            assert active_spend_counter_batch() is outer
        assert outer.counter_keys == {"spend:key:a", "spend:key:b"}
        release_spend_counter_batch()
        with spend_counter_batch_scope(spend.store(), counter_keys=frozenset({"spend:key:c"})):
            inner = active_spend_counter_batch()
            assert inner is not outer
            assert inner is not None and inner.counter_keys == {"spend:key:c"}
        assert active_spend_counter_batch() is outer


@pytest.mark.asyncio
async def test_reservation_inside_the_admission_scope_reuses_its_mget_and_reserves_in_one_pipeline(monkeypatch):
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.spend_tracking.budget_reservation import reserve_budget_for_request

    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0, "spend:team:team": 2.0})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr("litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost", lambda **_: 0.5)
    token = UserAPIKeyAuth(token="hashed", team_id="team", max_budget=10.0)

    with spend_counter_batch_scope(spend.store(), counter_keys=admission_counter_keys(token, end_user_id=None)):
        reservation = await reserve_budget_for_request(
            request_body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            route="/chat/completions",
            llm_router=None,
            valid_token=token,
            team_object=LiteLLM_TeamTableCachedObj(team_id="team", max_budget=20.0),
            user_object=None,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            proxy_logging_obj=MagicMock(),
        )

    assert reservation is not None
    assert [c.split()[0] for c in spend.commands] == ["MGET", "PIPELINE"], spend.commands
    assert set(spend.commands[0].split()[1:]) == {"spend:key:hashed", "spend:team:team"}
    assert spend.commands[1] == "PIPELINE spend:key:hashed spend:team:team"
    assert spend.redis == {"spend:key:hashed": 1.5, "spend:team:team": 2.5}
    assert [entry["counter_key"] for entry in reservation["entries"]] == ["spend:key:hashed", "spend:team:team"]


@pytest.mark.asyncio
async def test_a_failed_reservation_pipeline_drops_every_counter_and_reserves_nothing(monkeypatch):
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.spend_tracking.budget_reservation import reserve_budget_for_request

    spend = FakeSpendCounters(redis={"spend:key:hashed": 1.0, "spend:team:team": 2.0}, fail_pipelines=True)
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr("litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost", lambda **_: 0.5)
    token = UserAPIKeyAuth(token="hashed", team_id="team", max_budget=10.0)

    reservation = await reserve_budget_for_request(
        request_body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        route="/chat/completions",
        llm_router=None,
        valid_token=token,
        team_object=LiteLLM_TeamTableCachedObj(team_id="team", max_budget=20.0),
        user_object=None,
        prisma_client=None,
        user_api_key_cache=DualCache(),
        proxy_logging_obj=MagicMock(),
    )

    assert reservation is None
    assert {c.split()[1] for c in spend.commands if c.startswith("DEL ")} == {"spend:key:hashed", "spend:team:team"}
    assert spend.redis == {}


@pytest.mark.asyncio
async def test_post_call_lifecycle_reads_the_counters_after_the_db_update_and_writes_one_pipeline(monkeypatch):
    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters

    spend = FakeSpendCounters(redis={key: 1.0 for key in POST_CALL_KEYS})
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    proxy_logging_obj = MagicMock()

    async def _update_database(**kwargs: object) -> bool:
        spend.commands.append("DB")
        return True

    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(side_effect=_update_database)
    reservation = _reservation(reserved_cost=0.4)

    charged = await _update_database_and_spend_counters(
        proxy_logging_obj=proxy_logging_obj,
        increment_spend_counters=ps.increment_spend_counters,
        user_api_key="hashed",
        user_id="user",
        end_user_id="eu",
        team_id="team",
        org_id="org",
        kwargs={},
        completion_response=None,
        start_time=None,
        end_time=None,
        response_cost=0.5,
        budget_reservation=reservation,
        request_tags=["prod"],
        model_access_groups=["premium"],
    )

    assert charged is True
    proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()
    assert [c.split()[0] for c in spend.commands] == ["MGET", "DB", "MGET", "PIPELINE"], spend.commands
    assert set(spend.commands[0].split()[1:]) == RESERVED_KEYS
    assert set(spend.commands[2].split()[1:]) == POST_CALL_KEYS
    assert set(spend.commands[3].split()[1:]) == POST_CALL_KEYS
    assert {key: round(spend.redis[key], 6) for key in POST_CALL_KEYS} == {
        key: (1.1 if key in RESERVED_KEYS else 1.5) for key in POST_CALL_KEYS
    }
    assert [round(entry["applied_adjustment"], 6) for entry in reservation["entries"]] == [0.1] * len(RESERVED_KEYS)
    assert reservation["finalized"] is True
    assert active_spend_counter_batch() is None


def _reservation_fixture(monkeypatch, spend: FakeSpendCounters) -> None:
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr("litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost", lambda **_: 0.5)


async def _reserve(spend: FakeSpendCounters, token: UserAPIKeyAuth, team_max_budget: float) -> dict[str, object] | None:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj
    from litellm.proxy.spend_tracking.budget_reservation import reserve_budget_for_request

    with spend_counter_batch_scope(spend.store(), counter_keys=admission_counter_keys(token, end_user_id=None)):
        return await reserve_budget_for_request(
            request_body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            route="/chat/completions",
            llm_router=None,
            valid_token=token,
            team_object=LiteLLM_TeamTableCachedObj(team_id="team", max_budget=team_max_budget),
            user_object=None,
            prisma_client=None,
            user_api_key_cache=DualCache(),
            proxy_logging_obj=MagicMock(),
        )


@pytest.mark.asyncio
async def test_a_rejected_counter_is_charged_alone_so_the_counters_after_it_are_never_touched(monkeypatch):
    """Only counters the admission MGET says still fit the estimate share the reservation pipeline; a counter that
    does not is charged on its own first, so its rejection never inflates a sibling counter, not even briefly."""
    spend = FakeSpendCounters(redis={"spend:key:hashed": 10.0, "spend:team:team": 2.0})
    _reservation_fixture(monkeypatch, spend)
    token = UserAPIKeyAuth(token="hashed", team_id="team", max_budget=10.0)

    with pytest.raises(litellm.BudgetExceededError):
        await _reserve(spend, token, team_max_budget=20.0)

    writes = [c for c in spend.commands if not c.startswith("MGET")]
    assert writes and all("spend:team:team" not in c for c in writes), spend.commands
    assert spend.redis == {"spend:key:hashed": 10.0, "spend:team:team": 2.0}


@pytest.mark.asyncio
async def test_a_resized_reservation_is_carried_at_its_resized_cost_to_the_counters_charged_after_it(monkeypatch):
    spend = FakeSpendCounters(redis={"spend:key:hashed": 9.8, "spend:team:team": 2.0})
    _reservation_fixture(monkeypatch, spend)
    token = UserAPIKeyAuth(token="hashed", team_id="team", max_budget=10.0)

    reservation = await _reserve(spend, token, team_max_budget=2.1)

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.1)
    assert spend.redis["spend:key:hashed"] == pytest.approx(9.9)
    assert spend.redis["spend:team:team"] == pytest.approx(2.1)
