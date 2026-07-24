"""Behavior pins for spend-counter helpers in proxy_server.

Pins covered:
- ``get_current_spend``
- ``increment_spend_counters``
- ``_reconcile_budget_reservation_for_counter_update``
- ``_increment_end_user_and_tag_spend_counters``
- ``_increment_org_spend_counter``
- ``_init_and_increment_unreserved_spend_counter``
- ``_init_and_increment_spend_counter``
- ``_init_and_increment_window_spend_counter``
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
        cache.redis_cache.async_get_cache = AsyncMock(
            return_value=redis_get_value, side_effect=redis_get_side_effect
        )
        cache.redis_cache.async_increment = AsyncMock(
            return_value=redis_increment_value,
            side_effect=redis_increment_side_effect,
        )
        cache.redis_cache.async_delete_cache = AsyncMock()
        cache.redis_cache.async_set_cache = AsyncMock()
        cache.redis_cache.async_set_max = AsyncMock()
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
    cache.async_get_cache = AsyncMock(
        return_value=get_value, side_effect=get_side_effect
    )
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

    result = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=99.0
    )
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
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(
        key="spend:key:abc", value=12.0
    )


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

    result = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=12.0
    )

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

    first = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=12.0, max_budget=10.0
    )
    second = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=12.0, max_budget=10.0
    )

    assert first == 12.0
    assert second == 12.0
    assert from_db.await_count == 1


@pytest.mark.asyncio
async def test_get_current_spend_floors_end_user_tag_against_fallback(monkeypatch):
    """End-user and tag counters have no DB row (from_db returns None). When the
    counter is stale-low, enforcement falls back to the caller's recorded spend
    (loaded fresh in auth) instead of trusting the stale counter."""
    fake_cache = _make_spend_counter_cache(redis_get_value=2.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=None))

    result = await ps.get_current_spend(
        counter_key="spend:end_user:e1",
        fallback_spend=20.0,
        max_budget=10.0,
    )

    assert result == 20.0
    # no DB row to repair against, so the shared counter is left untouched
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
    fake_cache.redis_cache.async_set_max.assert_awaited_once_with(
        key=counter_key, value=15.0
    )


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_rejects_when_unverifiable(monkeypatch):
    """With fail_closed_budget_enforcement on, an admit decision backed only by a
    per-pod fallback (Redis unreachable and DB unreadable) is rejected with 503
    rather than admitted on an unverifiable budget."""
    from fastapi import HTTPException

    fake_cache = _make_spend_counter_cache(
        redis_get_side_effect=RuntimeError("redis down")
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(
        ps, "general_settings", {"fail_closed_budget_enforcement": True}
    )
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    with pytest.raises(HTTPException) as exc:
        await ps.get_current_spend(
            counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0
        )
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_off_admits_when_unverifiable(monkeypatch):
    """Default (flag off): an unverifiable read keeps the existing behavior and
    admits using the cached fallback — no new rejection."""
    fake_cache = _make_spend_counter_cache(
        redis_get_side_effect=RuntimeError("redis down")
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    result = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0
    )
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_admits_when_redis_verified(monkeypatch):
    """Fail-closed only rejects unverifiable reads: a value served by Redis is
    authoritative, so an under-budget request is admitted normally."""
    fake_cache = _make_spend_counter_cache(redis_get_value=1.0)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(
        ps, "general_settings", {"fail_closed_budget_enforcement": True}
    )

    result = await ps.get_current_spend(
        counter_key="spend:key:abc", fallback_spend=1.0, max_budget=10.0
    )
    assert result == 1.0


@pytest.mark.asyncio
async def test_get_current_spend_fail_closed_allows_authoritative_fallback(monkeypatch):
    """End-user/tag callers pass fallback_authoritative=True (their spend is
    loaded fresh from the DB in auth), so fail-closed does not reject them even
    when the counter path is unreadable."""
    fake_cache = _make_spend_counter_cache(
        redis_get_side_effect=RuntimeError("redis down")
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(
        ps, "general_settings", {"fail_closed_budget_enforcement": True}
    )
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

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
    monkeypatch.setattr(
        ps, "general_settings", {"fail_closed_budget_enforcement": True}
    )
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
    fake_cache = _make_spend_counter_cache(
        redis_get_value=None, redis_increment_value=5.0
    )
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)

    async def _fake_coalesced(**kwargs):
        return None

    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(side_effect=_fake_coalesced)
    )

    await ps.increment_spend_counters(
        token="hashed-tok",
        team_id="t1",
        user_id="u1",
        response_cost=5.0,
    )

    observed = {
        "redis_increment_called": fake_cache.redis_cache.async_increment.called,
        "increment_calls": fake_cache.redis_cache.async_increment.call_count,
        "user_cache_used": fake_user_cache.async_get_cache.called,
    }
    assert normalize(observed) == {
        "redis_increment_called": True,
        "increment_calls": 4,
        "user_cache_used": True,
    }


class _ConcurrencyProbe:
    """Stand-in for redis_cache.async_increment that pins concurrency.

    Each call registers itself as in-flight and blocks on ``release`` until the
    test lets it proceed. ``all_arrived`` fires once ``expected`` distinct scope
    increments are simultaneously suspended here, which can only happen if the
    per-scope increments are gathered rather than awaited one after another.
    """

    def __init__(self, expected_concurrency: int):
        self.expected = expected_concurrency
        self.in_flight = 0
        self.max_in_flight = 0
        self.all_arrived = asyncio.Event()
        self.release = asyncio.Event()
        self.values: dict[str, float] = {}

    async def async_increment(self, *, key, value, refresh_ttl=True):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= self.expected:
            self.all_arrived.set()
        if not self.release.is_set():
            await self.release.wait()
        self.in_flight -= 1
        self.values[key] = self.values.get(key, 0.0) + value
        return self.values[key]


@pytest.mark.asyncio
async def test_increment_spend_counters_runs_scopes_concurrently(monkeypatch):
    """The six independent scopes (key, team, team_member, user, end_user+tags,
    org) must be incremented concurrently. The probe only fires once all six are
    suspended in async_increment at the same time, which is impossible if the
    awaits are chained sequentially."""
    probe = _ConcurrencyProbe(expected_concurrency=6)
    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    fake_cache.redis_cache.async_increment = probe.async_increment
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

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
            "scope increments did not run concurrently; sequential awaits "
            f"detected (peak in-flight was {probe.max_in_flight}, expected 6)"
        )

    assert probe.in_flight == 6
    assert probe.max_in_flight == 6
    probe.release.set()
    await task

    assert probe.values == {
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
    monkeypatch.setattr(
        br, "get_reserved_counter_keys", MagicMock(return_value=set(reserved))
    )
    monkeypatch.setattr(br, "reconcile_budget_reservation", AsyncMock())

    recorded: dict[str, float] = {}

    async def _record_increment(*, key, value, refresh_ttl=True):
        recorded[key] = recorded.get(key, 0.0) + value
        return recorded[key]

    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    fake_cache.redis_cache.async_increment = _record_increment
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

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
    left as an orphaned background task, and the reservation is not finalized."""
    recorded: dict[str, float] = {}

    async def _increment(*, key, value, refresh_ttl=True):
        if key == "spend:team:t1":
            raise RuntimeError("redis increment failed")
        recorded[key] = recorded.get(key, 0.0) + value
        return recorded[key]

    fake_cache = _make_spend_counter_cache(redis_get_value=None)
    fake_cache.redis_cache.async_increment = _increment
    fake_user_cache = _make_user_api_key_cache(get_value=None)
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    reservation = {"finalized": False}
    with pytest.raises(RuntimeError, match="redis increment failed"):
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
    assert recorded == {
        "spend:key:hashed-tok": 5.0,
        "spend:team_member:u1:t1": 5.0,
        "spend:user:u1": 5.0,
        "spend:end_user:eu1": 5.0,
        "spend:tag:a": 5.0,
        "spend:org:org1": 5.0,
    }


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


# ---------------------------------------------------------------------------
# _reconcile_budget_reservation_for_counter_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_budget_reservation_for_counter_update_returns_empty_set_when_none():
    result = await ps._reconcile_budget_reservation_for_counter_update(
        budget_reservation=None, response_cost=1.0
    )
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
# _increment_end_user_and_tag_spend_counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_end_user_and_tag_spend_counters_increments_each_unique_tag(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(
        redis_get_value=None, redis_increment_value=3.0
    )
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    await ps._increment_end_user_and_tag_spend_counters(
        end_user_id="eu1",
        tags=["a", "b", "a", "", None],
        response_cost=3.0,
        reserved_counter_keys=set(),
    )

    observed = {
        "increment_calls": fake_cache.redis_cache.async_increment.call_count,
        "in_memory_set_calls": fake_cache.in_memory_cache.set_cache.call_count,
        "called": fake_cache.redis_cache.async_increment.called,
    }
    assert normalize(observed) == {
        "increment_calls": 3,
        "in_memory_set_calls": 3,
        "called": True,
    }


@pytest.mark.asyncio
async def test_increment_end_user_and_tag_spend_counters_no_end_user_no_tags_invalid_input_noop(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._increment_end_user_and_tag_spend_counters(
        end_user_id=None,
        tags=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert fake_cache.redis_cache.async_increment.called is False


# ---------------------------------------------------------------------------
# _increment_org_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_increment_org_spend_counter_increments_when_org_present(monkeypatch):
    fake_cache = _make_spend_counter_cache(
        redis_get_value=None, redis_increment_value=10.0
    )
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    await ps._increment_org_spend_counter(
        org_id="org-1",
        response_cost=10.0,
        reserved_counter_keys=set(),
    )

    observed = {
        "increment_called": fake_cache.redis_cache.async_increment.called,
        "increment_calls": fake_cache.redis_cache.async_increment.call_count,
        "counter_key_arg": fake_cache.redis_cache.async_increment.call_args.kwargs[
            "key"
        ],
    }
    assert normalize(observed) == {
        "increment_called": True,
        "increment_calls": 1,
        "counter_key_arg": "spend:org:org-1",
    }


@pytest.mark.asyncio
async def test_increment_org_spend_counter_no_org_is_noop_invalid_id(monkeypatch):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._increment_org_spend_counter(
        org_id=None,
        response_cost=1.0,
        reserved_counter_keys=set(),
    )

    assert fake_cache.redis_cache.async_increment.called is False


# ---------------------------------------------------------------------------
# _init_and_increment_unreserved_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_and_increment_unreserved_spend_counter_skips_reserved_keys(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._init_and_increment_unreserved_spend_counter(
        counter_key="spend:tag:x",
        source_cache_key="tag:x",
        increment=1.0,
        reserved_counter_keys={"spend:tag:x"},
    )

    assert fake_cache.redis_cache.async_increment.called is False


@pytest.mark.asyncio
async def test_init_and_increment_unreserved_spend_counter_proceeds_when_not_reserved(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(
        redis_get_value=None, redis_increment_value=2.0
    )
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

    await ps._init_and_increment_unreserved_spend_counter(
        counter_key="spend:tag:y",
        source_cache_key="tag:y",
        increment=2.0,
        reserved_counter_keys=set(),
    )

    observed = {
        "increment_called": fake_cache.redis_cache.async_increment.called,
        "redis_get_called": fake_cache.redis_cache.async_get_cache.called,
        "reseed_consulted": True,
    }
    assert observed == {
        "increment_called": True,
        "redis_get_called": True,
        "reseed_consulted": True,
    }


# ---------------------------------------------------------------------------
# _init_and_increment_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_and_increment_spend_counter_warm_cache_skips_reseed(monkeypatch):
    fake_cache = _make_spend_counter_cache(
        redis_get_value=11.0, redis_increment_value=14.0
    )
    fake_user_cache = _make_user_api_key_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    reseed = AsyncMock(return_value=None)
    monkeypatch.setattr(ps.SpendCounterReseed, "coalesced", reseed)

    await ps._init_and_increment_spend_counter(
        counter_key="spend:key:k",
        source_cache_key="k",
        increment=3.0,
    )

    observed = {
        "reseed_called": reseed.called,
        "increment_called": fake_cache.redis_cache.async_increment.called,
        "in_memory_seeded_from_redis": fake_cache.in_memory_cache.set_cache.called,
    }
    assert normalize(observed) == {
        "reseed_called": False,
        "increment_called": True,
        "in_memory_seeded_from_redis": True,
    }


# ---------------------------------------------------------------------------
# _init_and_increment_window_spend_counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_and_increment_window_spend_counter_increments_when_initialized(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache(
        redis_get_value=0.0, redis_increment_value=5.0
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed,
        "coalesced_window",
        AsyncMock(return_value=0.0),
    )

    await ps._init_and_increment_window_spend_counter(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_start=datetime(2024, 1, 1),
        increment=5.0,
    )

    observed = {
        "redis_increment_called": fake_cache.redis_cache.async_increment.called,
        "increment_calls": fake_cache.redis_cache.async_increment.call_count,
        "in_memory_set_calls": fake_cache.in_memory_cache.set_cache.call_count,
    }
    assert normalize(observed) == {
        "redis_increment_called": True,
        "increment_calls": 1,
        "in_memory_set_calls": 2,
    }


@pytest.mark.asyncio
async def test_init_and_increment_window_spend_counter_missing_window_start_invalid_skips(
    monkeypatch,
):
    fake_cache = _make_spend_counter_cache()
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    await ps._init_and_increment_window_spend_counter(
        counter_key="spend:key:k:window:1d",
        entity_type="Key",
        entity_id="k",
        window_start=None,
        increment=5.0,
    )

    assert fake_cache.redis_cache.async_increment.called is False


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
    fake_cache = _make_spend_counter_cache(
        redis_get_value=None, redis_increment_value=7.0
    )
    fake_user_cache = _make_user_api_key_cache(get_value={"spend": 7.0})
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)
    monkeypatch.setattr(ps, "user_api_key_cache", fake_user_cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(
        ps.SpendCounterReseed, "coalesced", AsyncMock(return_value=None)
    )

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

    result = await ps._get_source_cache_base_spend(
        source_cache_key=["miss", "hit-obj", "miss2"]
    )

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

    result = await ps._increment_spend_counter_cache(
        counter_key="spend:key:k", increment=4.0
    )

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
    fake_cache = _make_spend_counter_cache(
        redis_increment_side_effect=RuntimeError("incr fail")
    )
    monkeypatch.setattr(ps, "spend_counter_cache", fake_cache)

    with pytest.raises(RuntimeError):
        await ps._increment_spend_counter_cache(
            counter_key="spend:key:k", increment=1.0
        )

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
        "delete_args_key": fake_cache.redis_cache.async_delete_cache.call_args.kwargs[
            "key"
        ],
    }
    assert normalize(observed) == {
        "in_memory_delete_called": True,
        "redis_delete_called": True,
        "delete_args_key": "spend:key:k",
    }


@pytest.mark.asyncio
async def test_invalidate_spend_counter_swallows_redis_failure_no_raise(monkeypatch):
    fake_cache = _make_spend_counter_cache()
    fake_cache.redis_cache.async_delete_cache = AsyncMock(
        side_effect=RuntimeError("redis down")
    )
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
