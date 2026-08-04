from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.spend_tracking.auto_router_benchmarks import (
    BENCHMARKS_MAX_WINDOW_DAYS,
    clamp_window,
    compute_benchmarks,
)

GROUP_KINDS = {"claude-auto": "semantic"}


class _FakeTable:
    """Records every upsert so a test can assert on what would be written."""

    def __init__(self):
        self.upserts = []

    async def upsert(self, where, data):
        self.upserts.append((where, data))


class _FakeDb:
    """Returns rows the way prisma really does: a list of plain dicts."""

    def __init__(self, rows, table=None):
        self._rows = rows
        self.queries = []
        self.litellm_autoroutersession = table or _FakeTable()

    async def query_raw(self, sql, *args):
        self.queries.append((sql, args))
        return self._rows


class _FakePrisma:
    def __init__(self, rows, table=None):
        self.db = _FakeDb(rows, table)


def group_row(**overrides):
    row = {
        "model_group": "claude-auto",
        "baseline_model": "anthropic/claude-opus-4-8",
        "sessions": 10,
        "turns": 100,
        "total_session_seconds": 36000.0,
        "total_tokens": 1_000_000,
        "actual_spend": 10.0,
        "baseline_spend": 100.0,
        "turns_with_usage": 100,
        "ephemeral_5m_tokens": 0,
        "ephemeral_1h_tokens": 5000,
        "same_model_turns": 60,
        "same_model_hits": 57,
        "first_visit_turns": 10,
        "first_visit_hits": 2,
        "return_turns": 30,
        "return_hits": 24,
        "stale_return_misses": 4,
        "savable_return_misses": 2,
        "rescued_spend": 6.76,
        "replay_spend": 3.91,
    }
    row.update(overrides)
    return row


async def benchmarks_for(**overrides):
    prisma = _FakePrisma([group_row(**overrides)])
    return await compute_benchmarks(prisma, GROUP_KINDS, "2026-07-02", "2026-08-01")


class TestWindowClamping:
    def test_a_wider_request_is_clamped_to_the_maximum_window(self):
        window = clamp_window("2020-01-01", "2026-08-01")
        expected = (datetime(2026, 8, 1, tzinfo=timezone.utc) - timedelta(days=BENCHMARKS_MAX_WINDOW_DAYS)).date()
        assert window.start == expected.isoformat()

    def test_a_narrower_request_is_served_as_asked(self):
        assert clamp_window("2026-07-25", "2026-08-01").start == "2026-07-25"

    def test_the_response_echoes_the_window_actually_served(self):
        window = clamp_window("2020-01-01", "2026-08-01")
        assert window.end == "2026-08-01"


@pytest.mark.asyncio
class TestSessionShape:
    async def test_turns_per_session_divides_turns_by_sessions(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_turns_per_session == pytest.approx(10.0)

    async def test_session_length_averages_the_summed_durations(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_session_length_seconds == pytest.approx(3600.0)

    async def test_tokens_per_session_divides_tokens_by_sessions(self):
        result = await benchmarks_for()
        assert result.groups[0].avg_tokens_per_session == pytest.approx(100_000.0)

    async def test_a_group_with_no_sessions_is_omitted_rather_than_zeroed(self):
        result = await benchmarks_for(sessions=0)
        assert result.groups == ()


@pytest.mark.asyncio
class TestSavings:
    async def test_savings_is_baseline_minus_actual(self):
        result = await benchmarks_for()
        assert result.groups[0].savings == pytest.approx(90.0)
        assert result.groups[0].savings_pct == pytest.approx(90.0)

    async def test_a_route_that_cost_more_than_the_baseline_reports_a_loss(self):
        """Signed on purpose: a cache-thrashing router must not read as zero."""
        result = await benchmarks_for(actual_spend=120.0, baseline_spend=100.0)
        assert result.groups[0].savings == pytest.approx(-20.0)
        assert result.groups[0].savings_pct == pytest.approx(-20.0)

    async def test_an_unpriced_baseline_reports_no_percentage_instead_of_dividing_by_zero(self):
        result = await benchmarks_for(baseline_spend=0.0)
        assert result.groups[0].savings_pct == 0.0


@pytest.mark.asyncio
class TestCacheBuckets:
    async def test_the_three_buckets_sum_to_the_reported_turn_count(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.same_model_turns + cache.first_visit_turns + cache.return_turns == cache.turns

    async def test_the_headline_rate_is_weighted_by_turns_not_an_average_of_buckets(self):
        """57+2+24 hits over 100 turns is 83%, not the 61% mean of the three rates."""
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.hit_rate_pct == pytest.approx(83.0)

    async def test_each_bucket_reports_its_own_hit_rate(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.same_model_hit_rate_pct == pytest.approx(95.0)
        assert cache.first_visit_hit_rate_pct == pytest.approx(20.0)
        assert cache.return_hit_rate_pct == pytest.approx(80.0)

    async def test_stale_share_is_measured_against_return_misses_only(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.stale_miss_share_pct == pytest.approx(100.0 * 4 / 6)

    async def test_savable_share_is_measured_against_every_miss(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.warming_savable_miss_pct == pytest.approx(100.0 * 2 / 17)

    async def test_cache_is_omitted_when_nothing_reported_usage(self):
        result = await benchmarks_for(turns_with_usage=0)
        assert result.groups[0].cache is None

    async def test_coverage_is_the_share_of_turns_that_reported_usage(self):
        cache = (await benchmarks_for(turns_with_usage=50)).groups[0].cache
        assert cache is not None
        assert cache.usage_coverage_pct == pytest.approx(50.0)


@pytest.mark.asyncio
class TestWarmingEstimate:
    async def test_net_is_rescued_less_replays(self):
        cache = (await benchmarks_for()).groups[0].cache
        assert cache is not None
        assert cache.warming_net_spend == pytest.approx(6.76 - 3.91)

    async def test_break_even_follows_the_ttl_in_use(self):
        one_hour = (await benchmarks_for()).groups[0].cache
        five_min = (await benchmarks_for(ephemeral_1h_tokens=0, ephemeral_5m_tokens=5000)).groups[0].cache
        assert one_hour is not None and five_min is not None
        assert one_hour.ttl_seconds == 3600
        assert one_hour.warming_break_even_pct == 5.0
        assert five_min.ttl_seconds == 300
        assert five_min.warming_break_even_pct == 9.0

    async def test_no_ephemeral_evidence_reads_as_the_five_minute_tier(self):
        cache = (await benchmarks_for(ephemeral_1h_tokens=0, ephemeral_5m_tokens=0)).groups[0].cache
        assert cache is not None
        assert cache.ttl_seconds == 300


@pytest.mark.asyncio
class TestReadPathSource:
    async def test_the_dashboard_query_never_touches_the_spend_logs(self):
        prisma = _FakePrisma([group_row()])
        await compute_benchmarks(prisma, GROUP_KINDS, "2026-07-02", "2026-08-01")
        sql = prisma.db.queries[0][0]
        assert "LiteLLM_SpendLogs" not in sql
        assert "LiteLLM_AutoRouterSession" in sql

    async def test_one_query_covers_every_configured_auto_router(self):
        prisma = _FakePrisma([group_row(), group_row(model_group="claude-router-2")])
        result = await compute_benchmarks(
            prisma, {"claude-auto": "semantic", "claude-router-2": "complexity"}, "2026-07-02", "2026-08-01"
        )
        assert len(prisma.db.queries) == 1
        assert {g.model_group for g in result.groups} == {"claude-auto", "claude-router-2"}

    async def test_each_group_is_labelled_with_its_router_kind(self):
        prisma = _FakePrisma([group_row(model_group="claude-router-2")])
        result = await compute_benchmarks(prisma, {"claude-router-2": "complexity"}, "2026-07-02", "2026-08-01")
        assert result.groups[0].router_kind == "complexity"


class TestEveryCounterSurvivesToTheDashboard:
    """A counter declared on TurnDelta must reach the row and come back out.

    The failure this guards is silent: a metric gets written on every request,
    the read query never selects it, and the card shows zero forever with a green
    diff and passing tests. Both ends are checked against the one declaration.
    """

    def test_the_read_query_aggregates_every_counter(self):
        from litellm.proxy.spend_tracking.auto_router_benchmarks import _GROUP_SQL
        from litellm.proxy.spend_tracking.auto_router_sessions import COUNTER_FIELDS

        missing = [name for name in COUNTER_FIELDS if f"SUM({name})" not in _GROUP_SQL]
        assert missing == [], f"counters written but never read: {missing}"

    def test_the_response_row_carries_every_counter(self):
        from litellm.proxy.spend_tracking.auto_router_benchmarks import _GroupRow
        from litellm.proxy.spend_tracking.auto_router_sessions import COUNTER_FIELDS

        # `spend` is the one deliberate rename; the row calls it actual_spend
        expected = {"actual_spend" if name == "spend" else name for name in COUNTER_FIELDS}
        assert expected <= set(_GroupRow.model_fields)

    def test_the_flush_payload_carries_every_counter(self):
        from litellm.proxy.spend_tracking.auto_router_sessions import (
            COUNTER_FIELDS,
            EMPTY_SESSION_STATE,
            TurnDelta,
            counters_of,
        )

        assert set(counters_of(TurnDelta(state=EMPTY_SESSION_STATE))) == set(COUNTER_FIELDS)

    def test_counter_fields_is_derived_not_hand_listed(self):
        """Adding a field to TurnDelta must extend COUNTER_FIELDS with no other edit."""
        from dataclasses import fields

        from litellm.proxy.spend_tracking.auto_router_sessions import COUNTER_FIELDS, TurnDelta

        assert set(COUNTER_FIELDS) == {f.name for f in fields(TurnDelta)} - {"state"}
