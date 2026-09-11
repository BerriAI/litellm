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
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps

from .conftest import normalize


def _make_spend_counter_cache(
    *,
    redis_get_value=None,
    redis_get_side_effect=None,
    redis_increment_value=None,
    redis_increment_side_effect=None,
    in_memory_value=None,
    with_redis: bool = True,
):
    cache = MagicMock()
    cache.in_memory_cache = MagicMock()
    cache.in_memory_cache.get_cache = MagicMock(return_value=in_memory_value)
    cache.in_memory_cache.set_cache = MagicMock()
    cache.in_memory_cache.delete_cache = MagicMock()
    if with_redis:
        cache.redis_cache = MagicMock()
        cache.redis_cache.async_get_cache = AsyncMock(return_value=redis_get_value, side_effect=redis_get_side_effect)
        cache.redis_cache.async_increment = AsyncMock(
            return_value=redis_increment_value,
            side_effect=redis_increment_side_effect,
        )
        cache.redis_cache.async_delete_cache = AsyncMock()
        cache.redis_cache.async_set_cache = AsyncMock()
        cache.redis_cache.async_set_max = AsyncMock()
        cache.redis_cache.async_increment_pipeline = AsyncMock(return_value=None)
        cache.redis_cache.get_ttl = MagicMock(return_value=None)
    else:
        cache.redis_cache = None
    cache.async_increment_cache = AsyncMock(return_value=redis_increment_value)
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.async_set_cache = AsyncMock()
    cache.async_delete_cache = AsyncMock()
    cache.async_set_cache_pipeline = AsyncMock()
    return cache


def _make_user_api_key_cache(get_value=None, get_side_effect=None):
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=get_value, side_effect=get_side_effect)
    cache.async_set_cache_pipeline = AsyncMock()
    return cache


# ---------------------------------------------------------------------------
# get_current_spend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_spend_reads_redis_first(monkeypatch):
    fake_cache = _make_spend_counter_cache(redis_get_value=42.5)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=0.0)

    observed = {
        "value": result,
        "redis_called": fake_cache.redis_cache.async_get_cache.called,
        "in_memory_called": fake_cache.in_memory_cache.get_cache.called,
    }
    assert normalize(observed) == {
        "value": 42.5,
        "redis_called": True,
        "in_memory_called": False,
    }


@pytest.mark.asyncio
async def test_get_current_spend_redis_error_falls_back_to_in_memory(monkeypatch):
    fake_cache = _make_spend_counter_cache(
        redis_get_side_effect=RuntimeError("redis down"),
        in_memory_value=17.0,
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=99.0)
    assert result == 17.0


@pytest.mark.asyncio
async def test_get_current_spend_floors_stale_low_counter_against_db(monkeypatch):
    """A Redis counter left stale-low by a Redis restart must not admit a key
    whose authoritative DB spend is already over budget. With max_budget set,
    get_current_spend re-checks the DB and returns the higher recorded spend."""
    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(key="spend:key:abc", value=12.0)


@pytest.mark.asyncio
async def test_get_current_spend_no_db_recheck_when_counter_healthy(monkeypatch):
    """A healthy counter (at or above the caller's recorded spend) is trusted
    without a DB read, so under-budget traffic stays off the DB path."""
    fake_cache = _make_spend_counter_cache(redis_get_value=5.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache(redis_get_value=0.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache.redis_cache.async_set_max.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_spend_floor_caches_db_read(monkeypatch):
    """A persistently stale-low counter must not drive a DB read per request:
    the authoritative spend is cached in-process and reused within the window."""
    cache = ps.DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_get_cache = AsyncMock(return_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", cache)
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
    fake_cache: Final = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=None))

    result: Final = await ps.get_current_spend(
        counter_key=counter_key,
        fallback_spend=20.0,
        max_budget=10.0,
    )

    assert result == 20.0
    # no DB row to repair against, so the shared counter is left untouched
    fake_cache.redis_cache.async_set_max.assert_not_called()


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
    fake_cache: Final = _make_spend_counter_cache(redis_get_value=0.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache.redis_cache.async_set_max.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_spend_end_user_floor_repairs_a_stale_low_counter(monkeypatch):
    """After a Redis restart the end-user counter can sit below the recorded spend;
    the row wins and the shared counter is raised so other workers stop admitting on
    the stale value."""
    fake_cache: Final = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "prisma_client", _make_prisma_with_end_user_row(spend=12.0))

    result: Final = await ps.get_current_spend(
        counter_key="spend:end_user:customer-42",
        fallback_spend=12.0,
        max_budget=10.0,
    )

    assert result == 12.0
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(key="spend:end_user:customer-42", value=12.0)


@pytest.mark.asyncio
async def test_get_current_spend_end_user_without_a_row_keeps_the_cached_spend(monkeypatch):
    fake_cache: Final = _make_spend_counter_cache(redis_get_value=0.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "prisma_client", _make_prisma_with_end_user_row(spend=None))

    result: Final = await ps.get_current_spend(
        counter_key="spend:end_user:customer-42",
        fallback_spend=20.0,
        max_budget=10.0,
    )

    assert result == 20.0
    fake_cache.redis_cache.async_set_max.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_spend_floors_window_against_spend_logs(monkeypatch):
    """Per-window counters have no DB row but aggregate from spend logs. A
    stale-low window counter is floored to (and repaired up to) the logged
    window spend, even though the caller's fallback is 0."""
    from datetime import datetime, timezone

    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=None))
    wfsl = AsyncMock(return_value=15.0)
    monkeypatch.setattr(ps.SpendCounterReseed, "window_from_spend_logs", wfsl)

    counter_key = "spend:key:tok:window:7d"
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
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(key=counter_key, value=15.0)


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
    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "prisma_client", fake_prisma)

    counter_key = "spend:key:tok:window:7d"
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
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(key=counter_key, value=15.0)


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
    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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

    fake_cache = _make_spend_counter_cache(redis_get_side_effect=RuntimeError("redis down"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc:
        await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_off_admits_when_unverifiable(monkeypatch):
    """Default (flag off): an unverifiable read keeps the existing behavior and
    admits using the cached fallback — no new rejection."""
    fake_cache = _make_spend_counter_cache(redis_get_side_effect=RuntimeError("redis down"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_admits_when_redis_verified(monkeypatch):
    """Fail-closed only rejects unverifiable reads: a value served by Redis is
    authoritative, so an under-budget request is admitted normally."""
    fake_cache = _make_spend_counter_cache(redis_get_value=1.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": True})

    result = await ps.get_current_spend(counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0)
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_allows_authoritative_fallback(monkeypatch):
    """End-user/tag callers pass fallback_authoritative=True (their spend is
    loaded fresh from the DB in auth), so fail-closed does not reject them even
    when the counter path is unreadable."""
    fake_cache = _make_spend_counter_cache(redis_get_side_effect=RuntimeError("redis down"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache(redis_get_value=0.00001)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache(redis_get_value=None, redis_increment_value=5.0)
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)

    async def _fake_coalesced(**kwargs):
        return None

    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(side_effect=_fake_coalesced))

    await ps.increment_spend_counters(
        token="hashed-tok",
        team_id="t1",
        user_id="u1",
        response_cost=5.0,
    )

    pipeline = fake_cache.redis_cache.async_increment_pipeline
    pipeline.assert_awaited_once()
    increment_list = pipeline.await_args.kwargs["increment_list"]
    assert {op["key"] for op in increment_list} == {
        "spend:key:hashed-tok",
        "spend:team:t1",
        "spend:team_member:u1:t1",
        "spend:user:u1",
    }
    assert all(op["increment_value"] == 5.0 for op in increment_list)
    observed = {
        "redis_increment_called": fake_cache.redis_cache.async_increment.called,
        "pipeline_calls": pipeline.await_count,
        "user_cache_used": fake_user_cache.async_get_cache.called,
    }
    assert normalize(observed) == {
        "redis_increment_called": False,
        "pipeline_calls": 1,
        "user_cache_used": True,
    }


class _ConcurrencyProbe:
    """Stand-in for redis_cache.async_get_cache that pins concurrency.

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

    async def async_get_cache(self, *, key, **kwargs):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.keys.append(key)
        if self.in_flight >= self.expected:
            self.all_arrived.set()
        if not self.release.is_set():
            await self.release.wait()
        self.in_flight -= 1
        return 1.0


@pytest.mark.asyncio
async def test_increment_spend_counters_runs_scopes_concurrently(monkeypatch):
    """The six independent scopes (key, team, team_member, user, end_user+tags,
    org) must prepare their increments concurrently. The probe only fires once
    all eight warm-check reads (one per counter: 6 scopes + 2 tags) are
    suspended in async_get_cache at the same time, which is impossible if the
    awaits are chained sequentially."""
    probe = _ConcurrencyProbe(expected_concurrency=8)
    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_get_cache = probe.async_get_cache
    recorded: dict[str, float] = {}

    async def _record_pipeline(increment_list, **_):
        results = []
        for op in increment_list:
            recorded[op["key"]] = recorded.get(op["key"], 0.0) + op["increment_value"]
            results.append(recorded[op["key"]])
        return results

    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(side_effect=_record_pipeline)
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
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

    assert recorded == {
        "spend:key:hashed-tok": 5.0,
        "spend:team:t1": 5.0,
        "spend:team_member:u1:t1": 5.0,
        "spend:user:u1": 5.0,
        "spend:end_user:eu1": 5.0,
        "spend:tag:a": 5.0,
        "spend:tag:b": 5.0,
        "spend:org:org1": 5.0,
    }


@pytest.mark.asyncio
async def test_increment_spend_counters_skips_reserved_counter_keys(monkeypatch):
    """Counters already reserved by a budget reservation are skipped, every
    other scope is still incremented exactly once, and the reservation is
    finalized after the gathered work completes."""
    import litellm.proxy.spend_tracking.budget_reservation as br

    reserved = {"spend:key:hashed-tok", "spend:org:org1"}
    monkeypatch.setattr(br, "get_reserved_counter_keys", MagicMock(return_value=set(reserved)))
    monkeypatch.setattr(br, "reconcile_budget_reservation", AsyncMock())

    recorded: dict[str, float] = {}

    async def _record_pipeline(increment_list, **_):
        results = []
        for op in increment_list:
            recorded[op["key"]] = recorded.get(op["key"], 0.0) + op["increment_value"]
            results.append(recorded[op["key"]])
        return results

    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(side_effect=_record_pipeline)
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
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
    assert recorded == {
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

    async def _warm_check(*, key, **kwargs):
        warmed_keys.append(key)
        if key == "spend:team:t1":
            raise RuntimeError("redis get failed")
        return 1.0

    async def _reseed_fails(*, counter_key, **kwargs):
        if counter_key == "spend:team:t1":
            raise RuntimeError("reseed failed")

    applied: dict[str, float] = {}

    async def _record_pipeline(increment_list, **_):
        results = []
        for op in increment_list:
            applied[op["key"]] = op["increment_value"]
            results.append(op["increment_value"])
        return results

    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_get_cache = AsyncMock(side_effect=_warm_check)
    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(side_effect=_record_pipeline)
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed,
        "coalesced",
        AsyncMock(side_effect=_reseed_fails),
    )

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
    assert set(warmed_keys) == {
        "spend:key:hashed-tok",
        "spend:team:t1",
        "spend:team_member:u1:t1",
        "spend:user:u1",
        "spend:end_user:eu1",
        "spend:tag:a",
        "spend:org:org1",
    }
    # the surviving scopes' increments were still applied, in one pipeline call
    fake_cache.redis_cache.async_increment_pipeline.assert_awaited_once()
    assert applied == {
        "spend:key:hashed-tok": 5.0,
        "spend:team_member:u1:t1": 5.0,
        "spend:user:u1": 5.0,
        "spend:end_user:eu1": 5.0,
        "spend:tag:a": 5.0,
        "spend:org:org1": 5.0,
    }
    fake_cache.redis_cache.async_increment.assert_not_awaited()


@pytest.mark.asyncio
async def test_increment_spend_counters_zero_cost_is_noop_finalizes_reservation(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache()
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    assert fake_cache.redis_cache.async_increment.called is False
    fake_cache.redis_cache.async_increment_pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_increment_spend_counters_pipelines_all_scopes_in_one_redis_call(
    monkeypatch,
):
    """Every scope's increment must go out in a single async_increment_pipeline
    call, not one INCRBYFLOAT round-trip per scope."""
    counter_cache = ps.DualCache()
    fake_redis = AsyncMock()
    fake_redis.async_get_cache = AsyncMock(return_value=1.0)  # counters warm

    async def _pipeline(increment_list, **_):
        return [1.5] * len(increment_list)

    fake_redis.async_increment_pipeline = AsyncMock(side_effect=_pipeline)
    fake_redis.async_increment = AsyncMock()
    fake_redis.get_ttl = MagicMock(return_value=None)
    counter_cache.redis_cache = fake_redis
    monkeypatch.setattr(ps, "spend_counter_cache", counter_cache)
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

    fake_redis.async_increment_pipeline.assert_awaited_once()
    assert fake_redis.async_increment.await_count == 0
    increment_list = fake_redis.async_increment_pipeline.await_args.kwargs["increment_list"]
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
    assert {op["key"] for op in increment_list} == expected_keys
    assert all(op["increment_value"] == 0.5 for op in increment_list)
    for key in expected_keys:
        assert counter_cache.in_memory_cache.get_cache(key=key) == 1.5


@pytest.mark.asyncio
async def test_increment_spend_counters_pipeline_failure_invalidates_all_counters(
    monkeypatch,
):
    """A failing pipeline must invalidate every pending counter so the next
    request reseeds from the DB (which already holds this request's cost)
    instead of trusting a value the write may have partially applied."""
    from redis.exceptions import MaxConnectionsError

    counter_cache = ps.DualCache()
    pending_keys = (
        "spend:key:hashed",
        "spend:team:team-1",
        "spend:team_member:user-1:team-1",
        "spend:user:user-1",
        "spend:end_user:eu-1",
        "spend:tag:tag-a",
        "spend:tag:tag-b",
        "spend:org:org-1",
    )
    for key in pending_keys:
        counter_cache.in_memory_cache.set_cache(key=key, value=1.0)
    fake_redis = AsyncMock()
    fake_redis.async_get_cache = AsyncMock(return_value=1.0)  # counters warm
    fake_redis.async_increment_pipeline = AsyncMock(side_effect=MaxConnectionsError())
    fake_redis.async_increment = AsyncMock()
    fake_redis.async_delete_cache = AsyncMock()
    fake_redis.get_ttl = MagicMock(return_value=None)
    counter_cache.redis_cache = fake_redis
    monkeypatch.setattr(ps, "spend_counter_cache", counter_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", ps.DualCache())
    monkeypatch.setattr(ps, "prisma_client", None)

    with pytest.raises(MaxConnectionsError):
        await ps.increment_spend_counters(
            token="hashed",
            team_id="team-1",
            user_id="user-1",
            response_cost=0.5,
            org_id="org-1",
            end_user_id="eu-1",
            tags=["tag-a", "tag-b"],
        )

    assert fake_redis.async_increment.await_count == 0
    deleted_keys = {call.kwargs["key"] for call in fake_redis.async_delete_cache.await_args_list}
    assert deleted_keys == set(pending_keys)
    for key in pending_keys:
        assert counter_cache.in_memory_cache.get_cache(key=key) is None


# ---------------------------------------------------------------------------
# _reconcile_budget_reservation_for_counter_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_budget_reservation_for_counter_update_returns_empty_set_when_none():
    result = await ps._reconcile_budget_reservation_for_counter_update(budget_reservation=None, response_cost=1.0)
    assert result == set()


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

    assert result == set()
    assert fake_invalidate.called is True


# ---------------------------------------------------------------------------
# _prepare_end_user_and_tag_spend_increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_end_user_and_tag_spend_increments_returns_each_unique_tag(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=1.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    pending = await ps._prepare_end_user_and_tag_spend_increments(
        end_user_id=None,
        tags=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert pending == ()
    assert fake_cache.redis_cache.async_increment.called is False


# ---------------------------------------------------------------------------
# _prepare_org_spend_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_org_spend_increment_returns_pending_when_org_present(monkeypatch):
    fake_cache = _make_spend_counter_cache(redis_get_value=1.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    pending = await ps._prepare_org_spend_increment(
        org_id=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert pending == ()
    assert fake_cache.redis_cache.async_increment.called is False


# ---------------------------------------------------------------------------
# _prepare_unreserved_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_unreserved_spend_counter_increment_skips_reserved_keys(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    pending = await ps._prepare_unreserved_spend_counter_increment(
        counter_key="spend:tag:x",
        source_cache_key="tag:x",
        increment=1.0,
        reserved_counter_keys={"spend:tag:x"},
    )

    assert pending is None
    assert fake_cache.redis_cache.async_increment.called is False


@pytest.mark.asyncio
async def test_prepare_unreserved_spend_counter_increment_proceeds_when_not_reserved(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    fake_user_cache = _make_user_api_key_cache()
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
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
    assert fake_cache.redis_cache.async_get_cache.called is True
    assert reseed.called is True


# ---------------------------------------------------------------------------
# _prepare_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_spend_counter_increment_warm_cache_skips_reseed(monkeypatch):
    fake_cache = _make_spend_counter_cache(redis_get_value=11.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
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
    observed = {
        "reseed_called": reseed.called,
        "increment_called": fake_cache.redis_cache.async_increment.called,
        "in_memory_seeded_from_redis": fake_cache.in_memory_cache.set_cache.called,
    }
    assert normalize(observed) == {
        "reseed_called": False,
        "increment_called": False,
        "in_memory_seeded_from_redis": True,
    }


# ---------------------------------------------------------------------------
# _prepare_window_spend_counter_increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_window_spend_counter_increment_returns_pending_when_initialized(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=0.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    pending = await ps._prepare_window_spend_counter_increment(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_duration="1d",
        window_start=None,
        increment=5.0,
    )

    assert pending is None
    assert fake_cache.redis_cache.async_increment.called is False


# ---------------------------------------------------------------------------
# _apply_spend_counter_increments
# ---------------------------------------------------------------------------


def _two_pending_increments() -> tuple[ps._PendingSpendIncrement, ...]:
    return (
        ps._PendingSpendIncrement(counter_key="spend:key:k", increment=1.5),
        ps._PendingSpendIncrement(counter_key="spend:team:t", increment=1.5),
    )


@pytest.mark.asyncio
async def test_apply_spend_counter_increments_open_breaker_invalidates_and_returns(monkeypatch):
    """An open Redis circuit breaker is a known, already-logged state, not a per-request tracking failure.

    Re-raising the refusal sent every request through the cost callback's error path, which
    logged an ERROR and fired the failed-tracking alert once per request for the whole outage.
    """
    from litellm.caching.redis_cache import RedisCircuitBreakerOpenError

    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(
        side_effect=RedisCircuitBreakerOpenError("Redis circuit breaker is open")
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._apply_spend_counter_increments(_two_pending_increments())

    deleted_keys = sorted(call.kwargs["key"] for call in fake_cache.in_memory_cache.delete_cache.call_args_list)
    assert deleted_keys == ["spend:key:k", "spend:team:t"]
    fake_cache.in_memory_cache.set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_apply_spend_counter_increments_other_redis_error_invalidates_and_raises(monkeypatch):
    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_increment_pipeline = AsyncMock(side_effect=ConnectionError("redis down"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    with pytest.raises(ConnectionError, match="redis down"):
        await ps._apply_spend_counter_increments(_two_pending_increments())

    deleted_keys = sorted(call.kwargs["key"] for call in fake_cache.in_memory_cache.delete_cache.call_args_list)
    assert deleted_keys == ["spend:key:k", "spend:team:t"]
    fake_cache.in_memory_cache.set_cache.assert_not_called()


# ---------------------------------------------------------------------------
# _ensure_spend_counter_initialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_spend_counter_initialized_warm_skips_reseed_and_source(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=20.0)
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", reseed)

    await ps._ensure_spend_counter_initialized(
        counter_key="spend:user:u",
        source_cache_key="u",
    )

    observed = {
        "warm_check_redis": fake_cache.redis_cache.async_get_cache.called,
        "reseed_called": reseed.called,
        "source_cache_called": fake_user_cache.async_get_cache.called,
    }
    assert normalize(observed) == {
        "warm_check_redis": True,
        "reseed_called": False,
        "source_cache_called": False,
    }


@pytest.mark.asyncio
async def test_ensure_spend_counter_initialized_cold_seeds_from_source_cache(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=None, redis_increment_value=7.0)
    fake_user_cache = _make_user_api_key_cache(get_value={"spend": 7.0})
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None))

    await ps._ensure_spend_counter_initialized(
        counter_key="spend:user:u",
        source_cache_key="u",
    )

    observed = {
        "source_cache_called": fake_user_cache.async_get_cache.called,
        "seed_increment_called": fake_cache.redis_cache.async_increment.called,
        "warm_check_done": fake_cache.redis_cache.async_get_cache.called,
    }
    assert normalize(observed) == {
        "source_cache_called": True,
        "seed_increment_called": True,
        "warm_check_done": True,
    }


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
    fake_cache = _make_spend_counter_cache(redis_get_value=3.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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

    observed = {
        "initialized": initialized,
        "reseed_called": window_reseed.called,
        "redis_get_called": fake_cache.redis_cache.async_get_cache.called,
    }
    assert normalize(observed) == {
        "initialized": True,
        "reseed_called": False,
        "redis_get_called": True,
    }


@pytest.mark.asyncio
async def test_ensure_window_spend_counter_initialized_db_failure_invalid_returns_false(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
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
    fake_cache = _make_spend_counter_cache(redis_get_value=99.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    result = await ps._is_spend_counter_cache_warm(counter_key="spend:user:u")

    observed = {
        "result": result,
        "redis_get_called": fake_cache.redis_cache.async_get_cache.called,
        "in_memory_set_called": fake_cache.in_memory_cache.set_cache.called,
    }
    assert normalize(observed) == {
        "result": True,
        "redis_get_called": True,
        "in_memory_set_called": True,
    }


@pytest.mark.asyncio
async def test_is_spend_counter_cache_warm_redis_error_falls_back_to_in_memory(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(
        redis_get_side_effect=RuntimeError("redis err"),
        in_memory_value=None,
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    result = await ps._is_spend_counter_cache_warm(counter_key="spend:user:u")
    assert result is False


# ---------------------------------------------------------------------------
# _increment_spend_counter_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_spend_counter_cache_redis_path_returns_new_value(monkeypatch):
    fake_cache = _make_spend_counter_cache(redis_increment_value=44.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    result = await ps._increment_spend_counter_cache(counter_key="spend:key:k", increment=4.0)

    observed = {
        "result": result,
        "redis_increment_called": fake_cache.redis_cache.async_increment.called,
        "in_memory_set_called": fake_cache.in_memory_cache.set_cache.called,
    }
    assert normalize(observed) == {
        "result": 44.0,
        "redis_increment_called": True,
        "in_memory_set_called": True,
    }


@pytest.mark.asyncio
async def test_increment_spend_counter_cache_redis_error_raises_and_invalidates(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(redis_increment_side_effect=RuntimeError("incr fail"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    with pytest.raises(RuntimeError):
        await ps._increment_spend_counter_cache(counter_key="spend:key:k", increment=1.0)

    assert fake_cache.in_memory_cache.delete_cache.called is True
    assert fake_cache.redis_cache.async_delete_cache.called is True


# ---------------------------------------------------------------------------
# _invalidate_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_spend_counter_deletes_in_memory_and_redis(monkeypatch):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._invalidate_spend_counter(counter_key="spend:key:k")

    observed = {
        "in_memory_delete_called": fake_cache.in_memory_cache.delete_cache.called,
        "redis_delete_called": fake_cache.redis_cache.async_delete_cache.called,
        "delete_args_key": fake_cache.redis_cache.async_delete_cache.call_args.kwargs["key"],
    }
    assert normalize(observed) == {
        "in_memory_delete_called": True,
        "redis_delete_called": True,
        "delete_args_key": "spend:key:k",
    }


@pytest.mark.asyncio
async def test_invalidate_spend_counter_swallows_redis_failure_no_raise(monkeypatch):
    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_delete_cache = AsyncMock(side_effect=RuntimeError("redis down"))
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._invalidate_spend_counter(counter_key="spend:key:k")

    assert fake_cache.in_memory_cache.delete_cache.called is True


# ---------------------------------------------------------------------------
# update_cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_cache_no_cached_entities_schedules_pipeline_flush(monkeypatch):
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)

    await ps.update_cache(
        token=None,
        user_id="u1",
        end_user_id="eu1",
        team_id="t1",
        response_cost=1.0,
        parent_otel_span=None,
        tags=["x"],
    )

    observed = {
        "lookups": fake_user_cache.async_get_cache.call_count,
        "got_user": True,
        "got_team": True,
    }
    assert normalize(observed) == {
        "lookups": 4,
        "got_user": True,
        "got_team": True,
    }


@pytest.mark.asyncio
async def test_update_cache_user_cache_failure_invalid_state_is_swallowed(monkeypatch):
    """An inner _update_user_cache raising must not propagate — update_cache
    catches and logs, the public coroutine still completes normally."""
    fake_user_cache = MagicMock()
    fake_user_cache.async_get_cache = AsyncMock(side_effect=RuntimeError("cache down"))
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
