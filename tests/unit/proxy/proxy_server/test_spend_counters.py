"""Behavior pins for spend-counter helpers in proxy_server.

Pins covered:
- ``get_current_spend``
- ``increment_spend_counters``
- ``_reconcile_budget_reservation_for_counter_update``
- ``_prepare_end_user_and_tag_spend_increments``
- ``_prepare_org_spend_increment``
- ``_prepare_unreserved_spend_counter_increment``
- ``_prepare_spend_counter_increment``
- ``_prepare_window_spend_counter_increment``
- ``_apply_spend_counter_increments``
- ``_ensure_spend_counter_initialized``
- ``_get_source_cache_base_spend``
- ``_ensure_window_spend_counter_initialized``
- ``_is_spend_counter_cache_warm``
- ``_increment_spend_counter_cache``
- ``_invalidate_spend_counter``
- ``update_cache``
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Final, Literal, cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm._internal_context import current_service_target
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team import LiteLLM_TeamTable, LiteLLM_TeamTableCachedObj
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_local
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, end_user_cache_key, tag_cache_key
from litellm.proxy.management_endpoints.team_endpoints import refresh_cached_team

from .conftest import normalize
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


def _spend_counters(monkeypatch: pytest.MonkeyPatch, **fake: object) -> FakeSpendCounters:
    spend: Final = FakeSpendCounters(**fake)  # pyright: ignore[reportArgumentType]  # test double fields
    monkeypatch.setattr(ps, "spend_counter_store", spend.store())
    return spend


def _make_user_api_key_cache(get_value=None, get_side_effect=None):
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=get_value, side_effect=get_side_effect)
    cache.async_batch_get_cache = AsyncMock(side_effect=lambda keys, **_: [get_value for _ in keys])
    cache.async_set_cache_pipeline = AsyncMock()
    return cache


# ---------------------------------------------------------------------------
# get_current_spend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_spend_reads_redis_first(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:key:abc": 42.5}, memory={"spend:key:abc": 1.0})

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=0.0)

    assert result == 42.5
    assert spend.commands == ["GET spend:key:abc"]


@pytest.mark.asyncio
async def test_get_current_spend_redis_error_falls_back_to_in_memory(monkeypatch):
    _spend_counters(monkeypatch, fail=True, memory={"spend:key:abc": 17.0})

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=99.0)
    assert result == 17.0


@pytest.mark.asyncio
async def test_get_current_spend_floors_stale_low_counter_against_db(monkeypatch):
    """A Redis counter left stale-low by a Redis restart must not admit a key
    whose authoritative DB spend is already over budget. With max_budget set,
    get_current_spend re-checks the DB and returns the higher recorded spend."""
    spend = _spend_counters(monkeypatch, redis={"spend:key:abc": 2.0})
    from_db = AsyncMock(return_value=12.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    result = await ps.get_current_spend(
        counter_key="spend:key:abc",
        fallback_spend=12.0,
        max_budget=10.0,
    )

    assert result == 12.0
    assert from_db.await_count == 1
    # the stale counter is repaired up to the authoritative DB value via a
    # monotonic set-max so other workers read the corrected total, and a
    # concurrent increment cannot be clobbered
    assert spend.commands == ["GET spend:key:abc", "SETMAX spend:key:abc 12.0"]
    assert spend.redis == {"spend:key:abc": 12.0}


@pytest.mark.asyncio
async def test_get_current_spend_no_db_recheck_when_counter_healthy(monkeypatch):
    """A healthy counter (at or above the caller's recorded spend) is trusted
    without a DB read, so under-budget traffic stays off the DB path."""
    _spend_counters(monkeypatch, redis={"spend:key:abc": 5.0})
    from_db = AsyncMock(return_value=99.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    result = await ps.get_current_spend(
        counter_key="spend:key:abc",
        fallback_spend=3.0,
        max_budget=10.0,
    )

    assert result == 5.0
    assert from_db.await_count == 0


@pytest.mark.asyncio
async def test_get_current_spend_no_floor_without_max_budget(monkeypatch):
    """Without max_budget the read-time DB floor is skipped: callers that only
    read spend (alerts, soft budgets) keep the cheap counter-only behavior."""
    _spend_counters(monkeypatch, redis={"spend:key:abc": 2.0})
    from_db = AsyncMock(return_value=12.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=12.0)

    assert result == 2.0
    assert from_db.await_count == 0


@pytest.mark.asyncio
async def test_get_current_spend_floor_admits_after_reset(monkeypatch):
    """Right after a weekly reset the counter is 0 while the per-worker cached
    spend can still be last week's value. The DB floor reads the reset spend (0)
    and admits, so reset keys are not over-blocked."""
    spend = _spend_counters(monkeypatch, redis={"spend:key:abc": 0.0})
    from_db = AsyncMock(return_value=0.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    result = await ps.get_current_spend(
        counter_key="spend:key:abc",
        fallback_spend=12.0,
        max_budget=10.0,
    )

    assert result == 0.0
    assert from_db.await_count == 1
    # counter already matches the DB (reset to 0); nothing to repair, so no write
    assert spend.commands == ["GET spend:key:abc"]


@pytest.mark.asyncio
async def test_get_current_spend_floor_caches_db_read(monkeypatch):
    """A persistently stale-low counter must not drive a DB read per request:
    the authoritative spend is cached in-process and reused within the window."""
    _spend_counters(monkeypatch, redis={"spend:key:abc": 2.0})
    from_db = AsyncMock(return_value=12.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    first = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=12.0, max_budget=10.0)
    second = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=12.0, max_budget=10.0)

    assert first == 12.0
    assert second == 12.0
    assert from_db.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("counter_key", ("spend:end_user:e1", "spend:tag:t1"))
async def test_get_current_spend_floors_end_user_tag_against_fallback(monkeypatch, counter_key):
    """Tag counters have no DB row (from_db returns None), and an end-user counter has
    none to read without a DB client. When such a counter is stale-low, enforcement
    falls back to the caller's recorded spend (loaded fresh in auth) instead of
    trusting the stale counter."""
    spend: Final = _spend_counters(monkeypatch, redis_default=2.0)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=None))

    result: Final = await ps.get_current_spend(
        counter_key=counter_key,
        fallback_spend=20.0,
        max_budget=10.0,
    )

    assert result == 20.0
    # no DB row to repair against, so the shared counter is left untouched
    assert [command for command in spend.commands if command.startswith("SETMAX")] == []


def _make_prisma_with_end_user_row(spend: float | None):
    prisma: Final = MagicMock()
    prisma.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=None if spend is None else MagicMock(spend=spend)
    )
    return prisma


@pytest.mark.asyncio
async def test_get_current_spend_end_user_floor_admits_after_a_reset_on_a_stale_worker(monkeypatch):
    """The reset job zeroes LiteLLM_EndUserTable.spend and the shared counter, but it
    evicts the cached end-user object only on the worker that ran the reset. Every
    other worker still passes the pre-reset spend as fallback_spend, and that stale
    copy must not out-vote the reset row."""
    spend: Final = _spend_counters(monkeypatch, redis={"spend:end_user:customer-42": 0.0})
    prisma: Final = _make_prisma_with_end_user_row(spend=0.0)
    monkeypatch.setattr(ps, "prisma_client", prisma)

    result = await ps.get_current_spend(
        counter_key="spend:end_user:customer-42",
        fallback_spend=0.000032,
        max_budget=0.00003,
        fallback_authoritative=True,
    )

    assert result == 0.0
    prisma.db.litellm_endusertable.find_unique.assert_awaited_once_with(where={"user_id": "customer-42"})
    assert [command for command in spend.commands if command.startswith("SETMAX")] == []


@pytest.mark.asyncio
async def test_get_current_spend_end_user_floor_repairs_a_stale_low_counter(monkeypatch):
    """After a Redis restart the end-user counter can sit below the recorded spend;
    the row wins and the shared counter is raised so other workers stop admitting on
    the stale value."""
    spend: Final = _spend_counters(monkeypatch, redis={"spend:end_user:customer-42": 2.0})
    monkeypatch.setattr(ps, "prisma_client", _make_prisma_with_end_user_row(spend=12.0))

    result: Final = await ps.get_current_spend(
        counter_key="spend:end_user:customer-42",
        fallback_spend=12.0,
        max_budget=10.0,
    )

    assert result == 12.0
    assert spend.redis == {"spend:end_user:customer-42": 12.0}
    assert spend.commands[-1] == "SETMAX spend:end_user:customer-42 12.0"


@pytest.mark.asyncio
async def test_get_current_spend_end_user_without_a_row_keeps_the_cached_spend(monkeypatch):
    spend: Final = _spend_counters(monkeypatch, redis_default=0.0)
    monkeypatch.setattr(ps, "prisma_client", _make_prisma_with_end_user_row(spend=None))

    result: Final = await ps.get_current_spend(
        counter_key="spend:end_user:customer-42",
        fallback_spend=20.0,
        max_budget=10.0,
    )

    assert result == 20.0
    assert [command for command in spend.commands if command.startswith("SETMAX")] == []


@pytest.mark.asyncio
async def test_get_current_spend_floors_window_against_spend_logs(monkeypatch):
    """Per-window counters have no DB row but aggregate from spend logs. A
    stale-low window counter is floored to (and repaired up to) the logged
    window spend, even though the caller's fallback is 0."""
    from datetime import datetime, timezone

    counter_key = "spend:key:tok:window:7d"
    spend = _spend_counters(monkeypatch, redis={counter_key: 2.0})
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=None))
    wfsl = AsyncMock(return_value=15.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "window_from_spend_logs", wfsl)

    result = await ps.get_current_spend(
        counter_key=counter_key,
        fallback_spend=0.0,
        max_budget=10.0,
        window_entity_type="Key",
        window_entity_id="tok",
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert result == 15.0
    assert wfsl.await_count == 1
    assert spend.redis == {counter_key: 15.0}
    assert spend.commands[-1] == f"SETMAX {counter_key} 15.0"


def _make_window_spend_prisma(row=None, spend_logs_total=0.0):
    prisma = MagicMock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_spendlogs.group_by = AsyncMock(
        return_value=[{"api_key": "tok", "_sum": {"spend": spend_logs_total}}]
    )
    return prisma


@pytest.mark.asyncio
async def test_get_current_spend_floors_window_against_maintained_row(monkeypatch):
    """The floor re-check runs every few seconds per pod, so the window branch
    must read the maintained row and leave the unindexed spend-logs scan alone."""
    from datetime import timezone
    from types import SimpleNamespace

    window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fake_prisma = _make_window_spend_prisma(
        row=SimpleNamespace(window_start=window_start, spend=15.0),
        spend_logs_total=100.0,
    )
    counter_key = "spend:key:tok:window:7d"
    spend = _spend_counters(monkeypatch, redis={counter_key: 2.0})
    monkeypatch.setattr(ps, "prisma_client", fake_prisma)

    result = await ps.get_current_spend(
        counter_key=counter_key,
        fallback_spend=0.0,
        max_budget=10.0,
        window_entity_type="Key",
        window_entity_id="tok",
        window_duration="7d",
        window_start=window_start,
    )

    assert result == 15.0
    fake_prisma.db.litellm_spendlogs.group_by.assert_not_awaited()
    assert spend.redis == {counter_key: 15.0}
    assert spend.commands[-1] == f"SETMAX {counter_key} 15.0"


@pytest.mark.asyncio
async def test_get_current_spend_floors_window_against_logs_when_row_stale(monkeypatch):
    """A row left behind at a crossed window boundary must not be read as the
    current window's spend; the aggregate stays the fallback."""
    from datetime import timedelta, timezone
    from types import SimpleNamespace

    window_start = datetime(2026, 1, 8, tzinfo=timezone.utc)
    fake_prisma = _make_window_spend_prisma(
        row=SimpleNamespace(window_start=window_start - timedelta(days=7), spend=999.0),
        spend_logs_total=15.0,
    )
    _spend_counters(monkeypatch, redis_default=2.0)
    monkeypatch.setattr(ps, "prisma_client", fake_prisma)

    result = await ps.get_current_spend(
        counter_key="spend:key:tok:window:7d",
        fallback_spend=0.0,
        max_budget=10.0,
        window_entity_type="Key",
        window_entity_id="tok",
        window_duration="7d",
        window_start=window_start,
    )

    assert result == 15.0
    fake_prisma.db.litellm_spendlogs.group_by.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_rejects_when_unverifiable(monkeypatch):
    """With fail_closed_budget_enforcement on, an admit decision backed only by a
    per-pod fallback (Redis unreachable and DB unreadable) is rejected with 503
    rather than admitted on an unverifiable budget."""
    from fastapi import HTTPException

    _spend_counters(monkeypatch, fail=True)
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc:
        await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_off_admits_when_unverifiable(monkeypatch):
    """Default (flag off): an unverifiable read keeps the existing behavior and
    admits using the cached fallback — no new rejection."""
    _spend_counters(monkeypatch, fail=True)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_admits_when_redis_verified(monkeypatch):
    """Fail-closed only rejects unverifiable reads: a value served by Redis is
    authoritative, so an under-budget request is admitted normally."""
    _spend_counters(monkeypatch, redis={"spend:key:abc": 1.0})
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_allows_authoritative_fallback(monkeypatch):
    """End-user/tag callers pass fallback_authoritative=True (their spend is
    loaded fresh from the DB in auth), so fail-closed does not reject them even
    when the counter path is unreadable."""
    _spend_counters(monkeypatch, fail=True)
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    result = await ps.get_current_spend(
        counter_key="spend:end_user:e1",
        fallback_spend=1.0,
        max_budget=10.0,
        fallback_authoritative=True,
    )
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_strict_floors_when_fallback_also_stale(monkeypatch):
    """Strict mode closes the both-stale gap: when the counter AND the caller's
    cached spend are both stale-low (cheap guard would skip), strict mode still
    re-checks the authoritative DB and enforces against it."""
    _spend_counters(monkeypatch, redis_default=0.00001)
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})
    from_db = AsyncMock(return_value=0.5)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", from_db)

    # fallback == current, so the default cheap guard would NOT re-check
    result = await ps.get_current_spend(
        counter_key="spend:team:t1",
        fallback_spend=0.00001,
        max_budget=0.0002,
    )

    assert result == 0.5
    assert from_db.await_count == 1


# ---------------------------------------------------------------------------
# increment_spend_counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_spend_counters_increments_all_buckets(monkeypatch):
    spend = _spend_counters(monkeypatch)
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    await ps.increment_spend_counters(
        token="hashed-tok",
        team_id="t1",
        user_id="u1",
        response_cost=5.0,
    )

    keys = {"spend:key:hashed-tok", "spend:team:t1", "spend:team_member:u1:t1", "spend:user:u1"}
    assert [command.split()[0] for command in spend.commands] == ["MGET", "PIPELINE"]
    assert set(spend.commands[1].split()[1:]) == keys
    assert spend.redis == dict.fromkeys(keys, 5.0)
    assert fake_user_cache.async_get_cache.called is True


class _ConcurrencyProbe:
    """Stand-in for the per-counter warm-check read that pins concurrency.

    Each warm-check read registers itself as in-flight and blocks on ``release``
    until the test lets it proceed. ``all_arrived`` fires once ``expected``
    distinct scope warm-checks are simultaneously suspended here, which can only
    happen if the per-scope prepares are gathered rather than awaited one after
    another.
    """

    def __init__(self, expected_concurrency: int):
        self.expected = expected_concurrency
        self.in_flight = 0
        self.max_in_flight = 0
        self.all_arrived = asyncio.Event()
        self.release = asyncio.Event()
        self.keys: list[str] = []

    async def read(self, counter: str) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.keys.append(counter)
        if self.in_flight >= self.expected:
            self.all_arrived.set()
        if not self.release.is_set():
            await self.release.wait()
        self.in_flight -= 1


@pytest.mark.asyncio
async def test_increment_spend_counters_runs_scopes_concurrently(monkeypatch):
    """The six independent scopes (key, team, team_member, user, end_user+tags,
    org) must prepare their increments concurrently. The probe only fires once
    all eight warm-check reads (one per counter: 6 scopes + 2 tags) are
    suspended at the same time, which is impossible if the awaits are chained
    sequentially. Batch reads are off so every counter reads on its own."""
    probe = _ConcurrencyProbe(expected_concurrency=8)
    keys = (
        "spend:key:hashed-tok",
        "spend:team:t1",
        "spend:team_member:u1:t1",
        "spend:user:u1",
        "spend:end_user:eu1",
        "spend:tag:a",
        "spend:tag:b",
        "spend:org:org1",
    )
    spend = _spend_counters(monkeypatch, redis=dict.fromkeys(keys, 1.0), batch_reads=False, read_hook=probe.read)
    monkeypatch.setattr(ps, "user_api_key_cache", _make_user_api_key_cache(get_value=None))
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    task = asyncio.create_task(
        ps.increment_spend_counters(
            token="hashed-tok",
            team_id="t1",
            user_id="u1",
            org_id="org1",
            end_user_id="eu1",
            tags=["a", "b"],
            response_cost=5.0,
        )
    )

    try:
        await asyncio.wait_for(probe.all_arrived.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        probe.release.set()
        await task
        pytest.fail(
            "scope prepares did not run concurrently; sequential awaits "
            f"detected (peak in-flight was {probe.max_in_flight}, expected 8)"
        )

    assert probe.in_flight == 8
    assert probe.max_in_flight == 8
    probe.release.set()
    await task

    assert spend.redis == dict.fromkeys(keys, 6.0)


@pytest.mark.asyncio
async def test_increment_spend_counters_skips_reserved_counter_keys(monkeypatch):
    """Counters already reserved by a budget reservation are skipped, every
    other scope is still incremented exactly once, and the reservation is
    finalized after the gathered work completes."""
    import litellm.proxy.spend_tracking.budget_reservation as br

    reserved = {"spend:key:hashed-tok", "spend:org:org1"}
    monkeypatch.setattr(br, "get_reserved_counter_keys", MagicMock(return_value=set(reserved)))
    monkeypatch.setattr(br, "reconcile_budget_reservation", AsyncMock(return_value=()))
    spend = _spend_counters(monkeypatch)
    monkeypatch.setattr(ps, "user_api_key_cache", _make_user_api_key_cache(get_value=None))
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    reservation = {"finalized": False}
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
    assert spend.redis == {
        "spend:team:t1": 5.0,
        "spend:team_member:u1:t1": 5.0,
        "spend:user:u1": 5.0,
        "spend:end_user:eu1": 5.0,
        "spend:tag:a": 5.0,
    }


@pytest.mark.asyncio
async def test_increment_spend_counters_failing_scope_propagates_after_siblings_settle(
    monkeypatch,
):
    """A failure in one scope must propagate to the caller (so it can invalidate
    reserved counters) while every other scope still settles rather than being
    left as an orphaned background task, and the reservation is not finalized.
    The surviving scopes' increments are still applied in the single pipeline:
    dropping them would under-count spend, the unsafe direction for budget
    enforcement."""
    warmed_keys: list[str] = []

    async def _warm_check(counter: str) -> None:
        warmed_keys.append(counter)

    async def _reseed_fails(*, counter_key, **kwargs):
        if counter_key == "spend:team:t1":
            raise RuntimeError("reseed failed")

    keys = (
        "spend:key:hashed-tok",
        "spend:team_member:u1:t1",
        "spend:user:u1",
        "spend:end_user:eu1",
        "spend:tag:a",
        "spend:org:org1",
    )
    spend = _spend_counters(monkeypatch, redis=dict.fromkeys(keys, 1.0), batch_reads=False, read_hook=_warm_check)
    monkeypatch.setattr(ps, "user_api_key_cache", _make_user_api_key_cache(get_value=None))
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(side_effect=_reseed_fails))

    reservation = {"finalized": False}
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

    assert reservation["finalized"] is False
    # every sibling scope settled (its warm-check ran) before the error propagated
    assert set(warmed_keys) == {*keys, "spend:team:t1"}
    # the surviving scopes' increments were still applied, in one pipeline call
    assert [command.split()[0] for command in spend.commands if not command.startswith("GET")] == ["PIPELINE"]
    assert spend.redis == dict.fromkeys(keys, 6.0)


@pytest.mark.asyncio
async def test_increment_spend_counters_zero_cost_is_noop_finalizes_reservation(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    reservation = {"finalized": False}

    await ps.increment_spend_counters(
        token="hashed-tok",
        team_id="t1",
        user_id="u1",
        response_cost=0,
        budget_reservation=reservation,
    )

    assert reservation == {"finalized": True}
    assert spend.commands == []


@pytest.mark.asyncio
async def test_increment_spend_counters_pipelines_all_scopes_in_one_redis_call(
    monkeypatch,
):
    """Every scope's increment must go out in one pipeline, not one INCRBYFLOAT
    round-trip per scope, and each new total lands in this worker's memory."""
    expected_keys = {
        "spend:key:hashed",
        "spend:team:team-1",
        "spend:team_member:user-1:team-1",
        "spend:user:user-1",
        "spend:end_user:eu-1",
        "spend:tag:tag-a",
        "spend:tag:tag-b",
        "spend:org:org-1",
    }
    spend = _spend_counters(monkeypatch, redis=dict.fromkeys(expected_keys, 1.0))
    monkeypatch.setattr(ps, "user_api_key_cache", ps.DualCache())
    monkeypatch.setattr(ps, "prisma_client", None)

    await ps.increment_spend_counters(
        token="hashed",
        team_id="team-1",
        user_id="user-1",
        response_cost=0.5,
        org_id="org-1",
        end_user_id="eu-1",
        tags=["tag-a", "tag-b"],
    )

    assert [command.split()[0] for command in spend.commands] == ["MGET", "PIPELINE"]
    assert set(spend.commands[1].split()[1:]) == expected_keys
    assert spend.memory == dict.fromkeys(expected_keys, 1.5)


_PENDING_SPEND_COUNTER_KEYS: tuple[str, ...] = (
    "spend:key:hashed",
    "spend:team:team-1",
    "spend:team_member:user-1:team-1",
    "spend:user:user-1",
    "spend:end_user:eu-1",
    "spend:tag:tag-a",
    "spend:tag:tag-b",
    "spend:org:org-1",
)


def _warm_counters_behind_a_failing_pipeline(monkeypatch, failure: str) -> FakeSpendCounters:
    spend: Final = _spend_counters(
        monkeypatch,
        redis=dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.0),
        memory=dict.fromkeys(_PENDING_SPEND_COUNTER_KEYS, 1.0),
        fail_pipelines=True,
        failure=failure,
    )
    monkeypatch.setattr(ps, "user_api_key_cache", ps.DualCache())
    monkeypatch.setattr(ps, "prisma_client", None)
    return spend


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


def _assert_every_pending_counter_invalidated(spend: FakeSpendCounters) -> None:
    assert [command.split()[0] for command in spend.commands if not command.startswith("DEL ")] == ["MGET"]
    assert {command.split()[1] for command in spend.commands if command.startswith("DEL ")} == set(
        _PENDING_SPEND_COUNTER_KEYS
    )
    assert spend.redis == {}
    assert spend.memory == {}


@pytest.mark.asyncio
async def test_increment_spend_counters_pipeline_failure_invalidates_all_counters(
    monkeypatch,
):
    """A failing pipeline must invalidate every pending counter so the next
    request reseeds from the DB (which already holds this request's cost)
    instead of trusting a value the write may have partially applied."""
    spend = _warm_counters_behind_a_failing_pipeline(monkeypatch, failure="transport")

    with pytest.raises(NativeRedisStoreError):
        await _increment_every_spend_counter()

    _assert_every_pending_counter_invalidated(spend)


# ---------------------------------------------------------------------------
# _reconcile_budget_reservation_for_counter_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_budget_reservation_for_counter_update_returns_empty_set_when_none():
    result = await ps._reconcile_budget_reservation_for_counter_update(budget_reservation=None, response_cost=1.0)
    assert result.reserved_counter_keys == frozenset()
    assert result.pending == ()


@pytest.mark.asyncio
async def test_reconcile_budget_reservation_for_counter_update_failure_invalidates(
    monkeypatch,
):
    """Reservation reconcile raising must invalidate reserved counters, swallow
    the exception, and return an empty set so the caller falls back to the
    direct spend-counter increment instead of skipping it."""
    import litellm.proxy.spend_tracking.budget_reservation as br

    monkeypatch.setattr(
        br,
        "get_reserved_counter_keys",
        MagicMock(return_value={"spend:key:abc"}),
    )
    monkeypatch.setattr(
        br,
        "reconcile_budget_reservation",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    fake_invalidate = AsyncMock()
    monkeypatch.setattr(br, "invalidate_budget_reservation_counters", fake_invalidate)

    result = await ps._reconcile_budget_reservation_for_counter_update(
        budget_reservation={"foo": "bar"}, response_cost=1.0
    )

    assert result.reserved_counter_keys == frozenset()
    assert result.pending == ()
    assert fake_invalidate.called is True


@pytest.mark.asyncio
async def test_reconcile_budget_reservation_for_counter_update_finalized_reservation_falls_back_to_direct_increment(
    monkeypatch,
):
    """A reservation already finalized before the counter update (the pre-persist
    reconcile failed and dropped its counters) must not shield its keys from the
    direct increment, or the settled cost is never added back after the drop."""
    import litellm.proxy.spend_tracking.budget_reservation as br

    fake_reconcile = AsyncMock()
    monkeypatch.setattr(br, "reconcile_budget_reservation", fake_reconcile)

    result = await ps._reconcile_budget_reservation_for_counter_update(
        budget_reservation={
            "finalized": True,
            "entries": [{"counter_key": "spend:key:abc"}],
        },
        response_cost=1.0,
    )

    assert result.reserved_counter_keys == frozenset()
    assert result.pending == ()
    fake_reconcile.assert_not_awaited()


# ---------------------------------------------------------------------------
# _prepare_end_user_and_tag_spend_increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_end_user_and_tag_spend_increments_returns_each_unique_tag(
    monkeypatch,
):
    _spend_counters(monkeypatch, redis_default=1.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    pending = await ps._prepare_end_user_and_tag_spend_increments(
        end_user_id="eu1",
        tags=["a", "b", "a", "", None],
        response_cost=3.0,
        reserved_counter_keys=set(),
    )

    assert {item.counter_key for item in pending} == {
        "spend:end_user:eu1",
        "spend:tag:a",
        "spend:tag:b",
    }
    assert all(item.increment == 3.0 for item in pending)


@pytest.mark.asyncio
async def test_prepare_end_user_and_tag_spend_increments_no_end_user_no_tags_invalid_input_noop(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)

    pending = await ps._prepare_end_user_and_tag_spend_increments(
        end_user_id=None,
        tags=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert pending == ()
    assert spend.commands == []


# ---------------------------------------------------------------------------
# _prepare_org_spend_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_org_spend_increment_returns_pending_when_org_present(monkeypatch):
    _spend_counters(monkeypatch, redis_default=1.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    pending = await ps._prepare_org_spend_increment(
        org_id="org-1",
        response_cost=10.0,
        reserved_counter_keys=set(),
    )

    assert len(pending) == 1
    assert pending[0].counter_key == "spend:org:org-1"
    assert pending[0].increment == 10.0


@pytest.mark.asyncio
async def test_prepare_org_spend_increment_no_org_is_noop_invalid_id(monkeypatch):
    spend = _spend_counters(monkeypatch)

    pending = await ps._prepare_org_spend_increment(
        org_id=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert pending == ()
    assert spend.commands == []


# ---------------------------------------------------------------------------
# _prepare_unreserved_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_unreserved_spend_counter_increment_skips_reserved_keys(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)

    pending = await ps._prepare_unreserved_spend_counter_increment(
        counter_key="spend:tag:x",
        source_cache_key="tag:x",
        increment=1.0,
        reserved_counter_keys={"spend:tag:x"},
    )

    assert pending is None
    assert spend.commands == []


@pytest.mark.asyncio
async def test_prepare_unreserved_spend_counter_increment_proceeds_when_not_reserved(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)
    monkeypatch.setattr(ps, "user_api_key_cache", _make_user_api_key_cache())
    monkeypatch.setattr(ps, "prisma_client", None)
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", reseed)

    pending = await ps._prepare_unreserved_spend_counter_increment(
        counter_key="spend:tag:y",
        source_cache_key="tag:y",
        increment=2.0,
        reserved_counter_keys=set(),
    )

    assert pending is not None
    assert pending.counter_key == "spend:tag:y"
    assert pending.increment == 2.0
    assert spend.commands == ["GET spend:tag:y"]
    assert reseed.called is True


# ---------------------------------------------------------------------------
# _prepare_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_spend_counter_increment_warm_cache_skips_reseed(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:key:k": 11.0})
    monkeypatch.setattr(ps, "user_api_key_cache", _make_user_api_key_cache())
    monkeypatch.setattr(ps, "prisma_client", None)
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", reseed)

    pending = await ps._prepare_spend_counter_increment(
        counter_key="spend:key:k",
        source_cache_key="k",
        increment=3.0,
    )

    assert pending.counter_key == "spend:key:k"
    assert pending.increment == 3.0
    assert reseed.called is False
    assert spend.commands == ["GET spend:key:k"]
    assert spend.memory == {"spend:key:k": 11.0}


# ---------------------------------------------------------------------------
# _prepare_window_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_window_spend_counter_increment_returns_pending_when_initialized(
    monkeypatch,
):
    _spend_counters(monkeypatch, redis_default=0.0)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed,
        "coalesced_window",
        AsyncMock(return_value=0.0),
    )

    pending = await ps._prepare_window_spend_counter_increment(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_duration="1d",
        window_start=datetime(2024, 1, 1),
        increment=5.0,
    )

    assert pending is not None
    assert pending.counter_key == "spend:key:k:window:1d"
    assert pending.increment == 5.0


@pytest.mark.asyncio
async def test_prepare_window_spend_counter_increment_missing_window_start_invalid_skips(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)

    pending = await ps._prepare_window_spend_counter_increment(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_duration="1d",
        window_start=None,
        increment=5.0,
    )

    assert pending is None
    assert spend.commands == []


# ---------------------------------------------------------------------------
# _apply_spend_counter_increments
# ---------------------------------------------------------------------------


def _two_pending_increments() -> tuple[ps.PendingSpendIncrement, ...]:
    return (
        ps.PendingSpendIncrement(counter_key="spend:key:k", increment=1.5),
        ps.PendingSpendIncrement(counter_key="spend:team:t", increment=1.5),
    )


@pytest.mark.asyncio
async def test_apply_spend_counter_increments_open_breaker_invalidates_and_returns(monkeypatch):
    """An open Redis circuit breaker is a known, already-logged state, not a per-request tracking failure.

    Re-raising the refusal sent every request through the cost callback's error path, which
    logged an ERROR and fired the failed-tracking alert once per request for the whole outage.
    """
    spend = _spend_counters(
        monkeypatch, memory={"spend:key:k": 1.0, "spend:team:t": 1.0}, fail_pipelines=True, failure="breaker_open"
    )

    await ps._apply_spend_counter_increments(_two_pending_increments())

    assert spend.memory == {}
    assert spend.commands == ["DEL spend:key:k", "DEL spend:team:t"]


@pytest.mark.asyncio
async def test_apply_spend_counter_increments_redis_timeout_invalidates_and_returns(monkeypatch):
    """A Redis timeout invalidates the counters and returns without reaching the cost callback's error path."""
    spend = _spend_counters(
        monkeypatch, memory={"spend:key:k": 1.0, "spend:team:t": 1.0}, fail_pipelines=True, failure="timeout"
    )

    await ps._apply_spend_counter_increments(_two_pending_increments())

    assert spend.memory == {}
    assert spend.commands == ["DEL spend:key:k", "DEL spend:team:t"]


@pytest.mark.asyncio
async def test_apply_spend_counter_increments_other_redis_error_invalidates_and_raises(monkeypatch):
    spend = _spend_counters(monkeypatch, memory={"spend:key:k": 1.0, "spend:team:t": 1.0}, fail_pipelines=True)

    with pytest.raises(NativeRedisStoreError, match="transport"):
        await ps._apply_spend_counter_increments(_two_pending_increments())

    assert spend.memory == {}
    assert spend.commands == ["DEL spend:key:k", "DEL spend:team:t"]


# ---------------------------------------------------------------------------
# _ensure_spend_counter_initialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_spend_counter_initialized_warm_skips_reseed_and_source(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch, redis={"spend:user:u": 20.0})
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", reseed)

    await ps._ensure_spend_counter_initialized(
        counter_key="spend:user:u",
        source_cache_key="u",
    )

    assert spend.commands == ["GET spend:user:u"]
    assert reseed.called is False
    assert fake_user_cache.async_get_cache.called is False


@pytest.mark.asyncio
async def test_ensure_spend_counter_initialized_cold_seeds_from_source_cache(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch)
    fake_user_cache = _make_user_api_key_cache(get_value={"spend": 7.0})
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    await ps._ensure_spend_counter_initialized(
        counter_key="spend:user:u",
        source_cache_key="u",
    )

    assert fake_user_cache.async_get_cache.called is True
    assert spend.commands == ["GET spend:user:u", "INCRBYFLOAT spend:user:u 7.0"]
    assert spend.redis == {"spend:user:u": 7.0}


# ---------------------------------------------------------------------------
# _get_source_cache_base_spend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_source_cache_base_spend_reads_first_hit_from_list(monkeypatch):
    fake_user_cache = MagicMock()

    async def _get(key, **kwargs):
        if key == "miss":
            return None
        if key == "hit-obj":
            obj = MagicMock()
            obj.spend = 12.0
            return obj
        return None

    fake_user_cache.async_get_cache = AsyncMock(side_effect=_get)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)

    result = await ps._get_source_cache_base_spend(source_cache_key=["miss", "hit-obj", "miss2"])

    observed = {
        "result": result,
        "calls": fake_user_cache.async_get_cache.call_count,
        "stopped_after_hit": fake_user_cache.async_get_cache.call_count == 2,
    }
    assert normalize(observed) == {
        "result": 12.0,
        "calls": 2,
        "stopped_after_hit": True,
    }


@pytest.mark.asyncio
async def test_get_source_cache_base_spend_no_hits_returns_zero_fallback(monkeypatch):
    """All cache lookups miss — function falls back to 0.0 (no error)."""
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)

    result = await ps._get_source_cache_base_spend(source_cache_key="missing-key")
    assert result == 0.0


# ---------------------------------------------------------------------------
# _ensure_window_spend_counter_initialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_window_spend_counter_initialized_warm_returns_true(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:key:k:window:1d": 3.0})
    monkeypatch.setattr(ps, "prisma_client", None)
    window_reseed = AsyncMock(return_value=0.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced_window", window_reseed)

    initialized = await ps._ensure_window_spend_counter_initialized(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_duration="1d",
        window_start=datetime(2024, 1, 1),
    )

    assert initialized is True
    assert window_reseed.called is False
    assert spend.commands == ["GET spend:key:k:window:1d"]


@pytest.mark.asyncio
async def test_ensure_window_spend_counter_initialized_db_failure_invalid_returns_false(
    monkeypatch,
):
    _spend_counters(monkeypatch)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed,
        "coalesced_window",
        AsyncMock(return_value=None),
    )

    initialized = await ps._ensure_window_spend_counter_initialized(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_duration="1d",
        window_start=datetime(2024, 1, 1),
    )

    assert initialized is False


# ---------------------------------------------------------------------------
# _is_spend_counter_cache_warm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_spend_counter_cache_warm_redis_hit_seeds_in_memory(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:user:u": 99.0})

    result = await ps._is_spend_counter_cache_warm(counter_key="spend:user:u")

    assert result is True
    assert spend.commands == ["GET spend:user:u"]
    assert spend.memory == {"spend:user:u": 99.0}


@pytest.mark.asyncio
async def test_is_spend_counter_cache_warm_redis_error_falls_back_to_in_memory(
    monkeypatch,
):
    _spend_counters(monkeypatch, fail=True)

    result = await ps._is_spend_counter_cache_warm(counter_key="spend:user:u")
    assert result is False


# ---------------------------------------------------------------------------
# _increment_spend_counter_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_spend_counter_cache_redis_path_returns_new_value(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:key:k": 40.0})

    result = await ps._increment_spend_counter_cache(counter_key="spend:key:k", increment=4.0)

    assert result == 44.0
    assert spend.commands == ["INCRBYFLOAT spend:key:k 4.0"]
    assert spend.memory == {"spend:key:k": 44.0}


@pytest.mark.asyncio
async def test_increment_spend_counter_cache_redis_error_raises_and_invalidates(
    monkeypatch,
):
    spend = _spend_counters(monkeypatch, memory={"spend:key:k": 3.0}, fail=True)

    with pytest.raises(NativeRedisStoreError):
        await ps._increment_spend_counter_cache(counter_key="spend:key:k", increment=1.0)

    assert spend.memory == {}


# ---------------------------------------------------------------------------
# _invalidate_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_spend_counter_deletes_in_memory_and_redis(monkeypatch):
    spend = _spend_counters(monkeypatch, redis={"spend:key:k": 3.0}, memory={"spend:key:k": 3.0})

    await ps._invalidate_spend_counter(counter_key="spend:key:k")

    assert spend.commands == ["DEL spend:key:k"]
    assert spend.redis == {}
    assert spend.memory == {}


@pytest.mark.asyncio
async def test_invalidate_spend_counter_swallows_redis_failure_no_raise(monkeypatch):
    spend = _spend_counters(monkeypatch, memory={"spend:key:k": 3.0}, fail=True)

    await ps._invalidate_spend_counter(counter_key="spend:key:k")

    assert spend.memory == {}


# ---------------------------------------------------------------------------
# update_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_cache_user_cache_failure_invalid_state_is_swallowed(monkeypatch):
    """An inner _update_user_cache raising must not propagate — update_cache
    catches and logs, the public coroutine still completes normally."""
    fake_user_cache = MagicMock()
    fake_user_cache.async_batch_get_cache = AsyncMock(side_effect=RuntimeError("cache down"))
    fake_user_cache.async_set_cache_pipeline = AsyncMock()
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)

    result = await ps.update_cache(
        token=None,
        user_id="u1",
        end_user_id=None,
        team_id=None,
        response_cost=1.0,
        parent_otel_span=None,
        tags=None,
    )

    assert result is None


async def _drain_spend_cache_writes() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def _update_team_spend(team_id: str, response_cost: float) -> None:
    await ps.update_cache(
        token=None,
        user_id=None,
        end_user_id=None,
        team_id=team_id,
        response_cost=response_cost,
        parent_otel_span=None,
    )


@pytest.mark.asyncio
async def test_update_cache_adds_spend_to_the_cached_team_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = UserApiKeyCache()
    cache.set_cache(key="team_id:warm", value=LiteLLM_TeamTableCachedObj(team_id="warm", spend=5.0, blocked=True))
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    await _update_team_spend(team_id="warm", response_cost=2.5)
    await _update_team_spend(team_id="cold", response_cost=2.5)
    await _drain_spend_cache_writes()

    warm: Final = cache.get_cache(key="team_id:warm", model_type=LiteLLM_TeamTableCachedObj)
    assert warm is not None
    assert (warm.spend, warm.blocked) == (7.5, True)
    assert cache.get_cache(key="team_id:cold") is None


async def _block_on_this_worker(cache: UserApiKeyCache, team_id: str) -> None:
    await refresh_cached_team(
        team_row=LiteLLM_TeamTable(team_id=team_id, blocked=True),
        user_api_key_cache=cache,
        proxy_logging_obj=ps.proxy_logging_obj,
    )


async def _invalidated_by_a_peer(cache: UserApiKeyCache, team_id: str) -> None:
    evict_local(cache_keys=(f"team_id:{team_id}",), user_api_key_cache=cache)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("management_write", "expected_blocked"),
    [(_block_on_this_worker, True), (_invalidated_by_a_peer, None)],
)
async def test_update_cache_never_restores_a_team_a_management_write_replaced(
    management_write: Callable[[UserApiKeyCache, str], Awaitable[None]],
    expected_blocked: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_id: Final = "kill-switch"
    cache: Final = UserApiKeyCache()
    cache.set_cache(key=f"team_id:{team_id}", value=LiteLLM_TeamTableCachedObj(team_id=team_id, blocked=False))
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    await _update_team_spend(team_id=team_id, response_cost=1.0)
    await management_write(cache, team_id)
    await _drain_spend_cache_writes()

    seen_by_auth: Final = cache.get_cache(key=f"team_id:{team_id}", model_type=LiteLLM_TeamTableCachedObj)
    assert (seen_by_auth.blocked if seen_by_auth is not None else None) == expected_blocked


_CachedSpendObject = LiteLLM_UserTable | LiteLLM_EndUserTable | LiteLLM_TeamTableCachedObj | LiteLLM_TagTable


def _cached_spend_object(
    kind: Literal["user", "end_user", "team", "tag"], models: list[str]
) -> tuple[str, _CachedSpendObject]:
    match kind:
        case "user":
            return "race-user", LiteLLM_UserTable(user_id="race-user", models=models)
        case "end_user":
            return end_user_cache_key("race-end-user"), LiteLLM_EndUserTable(
                user_id="race-end-user", blocked=False, models=models
            )
        case "team":
            return "team_id:race-team", LiteLLM_TeamTableCachedObj(team_id="race-team", models=models)
        case "tag":
            return tag_cache_key("race-tag"), LiteLLM_TagTable(tag_name="race-tag", models=models)


class _RunsOnKeyRead(InMemoryCache):
    def __init__(self, on_read: Callable[[], Awaitable[None]]) -> None:
        super().__init__()
        self._on_read: Final = on_read

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        await self._on_read()
        return await super().async_get_cache(key, **kwargs)


def _cache_that_runs_on_key_read(on_read: Callable[[], Awaitable[None]]) -> UserApiKeyCache:
    return UserApiKeyCache(in_memory_cache=InMemoryCache(), key_object_in_memory_cache=_RunsOnKeyRead(on_read))


@pytest.fixture
def proxy_max_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "max_budget", 100.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "end_user", "team", "tag"])
@pytest.mark.parametrize(
    ("management_write", "expected"),
    [("none", (["gpt-a"], 0.25)), ("evict", None), ("refresh", (["gpt-b"], 0.25))],
)
async def test_update_cache_adds_spend_to_the_entry_a_management_write_left_mid_flight(
    kind: Literal["user", "end_user", "team", "tag"],
    management_write: Literal["none", "evict", "refresh"],
    expected: tuple[list[str], float] | None,
) -> None:
    key, seeded = _cached_spend_object(kind, models=["gpt-a"])
    _, refreshed = _cached_spend_object(kind, models=["gpt-b"])

    async def _write() -> None:
        if management_write == "none":
            return
        evict_local(cache_keys=(key,), user_api_key_cache=cache)
        if management_write == "refresh":
            cache.set_cache(key=key, value=refreshed, local_only=True)

    cache: Final = _cache_that_runs_on_key_read(on_read=_write)
    cache.set_cache(key=key, value=seeded, local_only=True)

    await ps.update_cache(
        token="sk-race",
        user_id="race-user",
        end_user_id="race-end-user",
        team_id="race-team",
        response_cost=0.25,
        parent_otel_span=None,
        tags=["race-tag"],
        cache=cache,
    )
    await _drain_spend_cache_writes()

    stored: Final = cache.get_cache(key=key, local_only=True, model_type=type(seeded))
    assert ((stored.models, stored.spend) if stored is not None else None) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
@pytest.mark.parametrize("user", ["cached_on_this_worker", "not_cached_on_this_worker", "none"])
async def test_update_cache_adds_spend_to_the_global_proxy_spend_for_every_request(
    user: Literal["cached_on_this_worker", "not_cached_on_this_worker", "none"],
) -> None:
    cache: Final = UserApiKeyCache()
    cache.set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0)
    if user == "cached_on_this_worker":
        cache.set_cache(key="spender", value=LiteLLM_UserTable(user_id="spender", spend=1.0))

    await ps.update_cache(
        token=None,
        user_id=None if user == "none" else "spender",
        end_user_id=None,
        team_id=None,
        response_cost=0.25,
        parent_otel_span=None,
        cache=cache,
    )
    await _drain_spend_cache_writes()

    assert cache.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY) == 10.25


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
async def test_update_cache_never_brings_back_a_global_proxy_spend_a_budget_reset_cleared() -> None:
    async def _budget_reset() -> None:
        await cache.async_delete_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)

    cache: Final = _cache_that_runs_on_key_read(on_read=_budget_reset)
    cache.set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0)

    await ps.update_cache(
        token="sk-spender",
        user_id="spender",
        end_user_id=None,
        team_id=None,
        response_cost=0.25,
        parent_otel_span=None,
        cache=cache,
    )
    await _drain_spend_cache_writes()

    assert cache.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY) is None


class _SharedSpend(FakeAuthCache):
    """The shared auth cache's Redis as the global spend scalar sees it; ``after_increment`` runs once Redis applied
    an increment, before its reply reaches the caller, and an increment equal to ``held`` waits for ``released``."""

    def __init__(
        self,
        total: float | None,
        after_increment: Callable[[], object] = lambda: None,
        held: float | None = None,
        applied: asyncio.Event | None = None,
        released: asyncio.Event | None = None,
    ) -> None:
        super().__init__(redis={} if total is None else {GLOBAL_PROXY_SPEND_CACHE_KEY: total})
        self._after_increment: Final = after_increment
        self._held: Final = held
        self._applied: Final = applied
        self._released: Final = released

    async def increment_if_present(self, key: str, amount: float) -> float | None:
        total: Final = await super().increment_if_present(key, amount)
        self._after_increment()
        if amount == self._held and self._applied is not None and self._released is not None:
            self._applied.set()
            await self._released.wait()
        return total


_SharedGlobalSpendOutcome = Literal["peers_added_spend", "reset_elsewhere", "reset_while_redis_answers", "redis_down"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
@pytest.mark.parametrize(
    ("outcome", "shared_total", "expected"),
    [
        ("peers_added_spend", 11.75, 12.0),
        ("reset_elsewhere", None, 10.25),
        ("reset_while_redis_answers", 9.75, None),
        ("redis_down", 10.0, 10.25),
    ],
)
async def test_update_cache_moves_this_workers_global_proxy_spend_to_the_shared_total(
    outcome: _SharedGlobalSpendOutcome, shared_total: float | None, expected: float | None
) -> None:
    def _reset_this_worker() -> None:
        if outcome == "reset_while_redis_answers":
            cache.delete_local(GLOBAL_PROXY_SPEND_CACHE_KEY)

    redis: Final = _SharedSpend(shared_total, after_increment=_reset_this_worker)
    if outcome == "redis_down":
        redis.fail = ConnectionError("redis down")
    cache, _ = shared_cache(redis)
    cache.set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0, local_only=True)

    await ps.update_cache(
        token=None,
        user_id=None,
        end_user_id=None,
        team_id=None,
        response_cost=0.25,
        parent_otel_span=None,
        cache=cache,
    )

    assert cache.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, local_only=True) == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
async def test_update_cache_keeps_the_newest_global_proxy_spend_when_redis_replies_out_of_order() -> None:
    held_increment_applied: Final = asyncio.Event()
    held_reply_released: Final = asyncio.Event()
    cache, _ = shared_cache(
        _SharedSpend(10.0, held=0.25, applied=held_increment_applied, released=held_reply_released)
    )
    cache.set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0, local_only=True)

    async def _request(response_cost: float) -> None:
        await ps.update_cache(
            token=None,
            user_id=None,
            end_user_id=None,
            team_id=None,
            response_cost=response_cost,
            parent_otel_span=None,
            cache=cache,
        )

    earlier: Final = asyncio.create_task(_request(0.25))
    await held_increment_applied.wait()
    await _request(0.5)
    held_reply_released.set()
    await earlier

    assert cache.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, local_only=True) == 10.75


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
async def test_update_cache_adds_the_global_spend_scalar_in_redis_as_spend_counters() -> None:
    target_during_increment: Final = MagicMock()
    redis: Final = _SharedSpend(10.0, after_increment=lambda: target_during_increment(current_service_target()))
    cache, _ = shared_cache(redis)

    await ps.update_cache(
        token=None,
        user_id="u1",
        end_user_id=None,
        team_id=None,
        response_cost=2.0,
        parent_otel_span=None,
        cache=cache,
    )

    assert target_during_increment.call_args_list == [call("spend_counters")]
    assert redis.redis[GLOBAL_PROXY_SPEND_CACHE_KEY] == 12.0
    assert current_service_target() is None


class _UnwritableKeyUserApiKeyCache(UserApiKeyCache):
    def __init__(self, unwritable_key: str) -> None:
        super().__init__(in_memory_cache=InMemoryCache())
        self._unwritable_key: Final = unwritable_key

    def set_cache(self, key: str | None, value: object, local_only: bool = False, **kwargs: object):
        if key == self._unwritable_key:
            raise RuntimeError("cache write failed")
        return super().set_cache(key, value, local_only=local_only, **kwargs)


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_max_budget")
async def test_update_cache_still_adds_spend_to_the_others_when_one_cached_object_fails() -> None:
    cache: Final = _UnwritableKeyUserApiKeyCache(unwritable_key="broken-user")
    cache.in_memory_cache.set_cache(
        key="broken-user",
        value=CacheCodec.serialize(LiteLLM_UserTable(user_id="broken-user", spend=1.0), model_type=LiteLLM_UserTable),
    )
    cache.set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0)
    cache.set_cache(key="team_id:still-billed", value=LiteLLM_TeamTableCachedObj(team_id="still-billed", spend=1.0))

    await ps.update_cache(
        token=None,
        user_id="broken-user",
        end_user_id=None,
        team_id="still-billed",
        response_cost=0.25,
        parent_otel_span=None,
        cache=cache,
    )
    await _drain_spend_cache_writes()

    team: Final = cache.get_cache(key="team_id:still-billed", model_type=LiteLLM_TeamTableCachedObj)
    assert (team.spend if team is not None else None, cache.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)) == (
        1.25,
        10.25,
    )


@pytest.mark.asyncio
async def test_increment_spend_counters_timeout_invalidates_all_counters_without_failing(
    monkeypatch,
):
    """A pipeline that timed out (a busy pool or a slow node) invalidates the counters and the request still
    succeeds."""
    spend = _warm_counters_behind_a_failing_pipeline(monkeypatch, failure="timeout")

    await _increment_every_spend_counter()

    _assert_every_pending_counter_invalidated(spend)
