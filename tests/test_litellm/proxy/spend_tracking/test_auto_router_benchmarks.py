"""Read-path derivations for the auto-router benchmarks dashboard.

Every figure the dashboard shows is a pure function of the rollup counters, so all of it
is exercised here without a database.
"""

import datetime as dt
from dataclasses import fields

import pytest

from litellm.proxy.spend_tracking.auto_router_benchmarks import (
    MAX_WINDOW_DAYS,
    _Counters,
    build_response,
    clamp_window,
    summarize,
)


def _row(model_group: str, **overrides: object) -> dict[str, object]:
    """Every counter at zero unless the case under test says otherwise."""
    return {
        "model_group": model_group,
        "router_kind": "complexity",
        "baseline_model": "anthropic/claude-opus-4-8",
        **{field.name: 0 for field in fields(_Counters)},
        "sessions": 1,
        **overrides,
    }


class TestSummarize:
    def test_savings_are_the_difference_between_the_two_arms(self):
        result = summarize(_Counters(sessions=4, turns=40, spend=364.59, baseline_spend=414.63))
        assert result.savings == pytest.approx(50.04)
        assert result.savings_pct == pytest.approx(100 * 50.04 / 414.63)
        assert result.saved_per_session == pytest.approx(50.04 / 4)
        assert result.avg_turns_per_session == pytest.approx(10.0)

    def test_a_cache_thrashing_router_reports_a_signed_loss(self):
        result = summarize(_Counters(sessions=1, turns=1, spend=5.0, baseline_spend=4.0))
        assert result.savings == pytest.approx(-1.0)
        assert result.savings_pct < 0

    def test_session_shape_averages_over_sessions_not_turns(self):
        result = summarize(
            _Counters(sessions=2, turns=64, total_tokens=10_000, total_session_seconds=7200.0)
        )
        assert result.avg_turns_per_session == pytest.approx(32.0)
        assert result.avg_session_seconds == pytest.approx(3600.0)
        assert result.avg_tokens_per_session == pytest.approx(5000.0)

    def test_an_empty_window_divides_by_nothing(self):
        result = summarize(_Counters())
        assert result.sessions == 0
        assert result.savings_pct == 0.0
        assert result.saved_per_session == 0.0
        assert result.avg_turns_per_session == 0.0
        assert result.cache is None


class TestCacheView:
    def test_hit_rate_is_weighted_by_turn_count_not_averaged_across_buckets(self):
        cache = summarize(
            _Counters(
                sessions=1,
                turns=3145,
                turns_with_usage=3145,
                warm_turns=2560,
                warm_hits=2491,
                first_visit_turns=146,
                first_visit_hits=15,
                expired_turns=439,
                expired_hits=348,
            )
        ).cache
        assert cache is not None
        hits = 2491 + 15 + 348
        assert cache.hit_rate_pct == pytest.approx(100 * hits / 3145)
        mean_of_bucket_rates = (
            cache.warm_hit_rate_pct + cache.first_visit_hit_rate_pct + cache.expired_hit_rate_pct
        ) / 3
        assert cache.hit_rate_pct != pytest.approx(mean_of_bucket_rates)
        assert cache.hit_rate_pct > mean_of_bucket_rates

    def test_traffic_that_never_touched_the_cache_is_left_out_of_the_hit_rate(self):
        """A model with caching off would otherwise read as a wall of misses. It is absent
        from the buckets, and coverage says how much of the traffic that was."""
        cache = summarize(
            _Counters(sessions=1, turns=100, turns_with_usage=40, warm_turns=40, warm_hits=36)
        ).cache
        assert cache is not None
        assert cache.turns == 40
        assert cache.coverage_pct == pytest.approx(40.0)
        assert cache.hit_rate_pct == pytest.approx(90.0)
        assert cache.misses == 4

    def test_the_three_buckets_partition_every_turn(self):
        counters = _Counters(
            sessions=1, turns=10, turns_with_usage=10, warm_turns=6, first_visit_turns=2, expired_turns=2
        )
        cache = summarize(counters).cache
        assert cache is not None
        assert cache.warm_turns + cache.first_visit_turns + cache.expired_turns == cache.turns

    def test_every_miss_has_one_cause_and_they_stack_to_the_whole(self):
        """A miss is cold by design, a changed prefix, an expiry, or a turn whose cache
        state could not be established."""
        cache = summarize(
            _Counters(
                sessions=1,
                turns=24,
                turns_with_usage=24,
                first_visit_turns=5,
                first_visit_hits=1,
                warm_turns=10,
                warm_hits=8,
                expired_turns=5,
                expired_hits=2,
                unordered_turns=4,
                unordered_hits=3,
            )
        ).cache
        assert cache is not None
        assert cache.hits == 14
        assert cache.misses == 10
        assert (cache.cold_misses, cache.prefix_change_misses, cache.expired_misses) == (4, 2, 3)
        assert cache.unattributed_misses == 1
        assert (
            cache.cold_misses + cache.prefix_change_misses + cache.expired_misses + cache.unattributed_misses
            == cache.misses
        )
        assert (
            cache.cold_miss_pct
            + cache.prefix_change_miss_pct
            + cache.expired_miss_pct
            + cache.unattributed_miss_pct
            == pytest.approx(100.0)
        )

    def test_an_unordered_turn_still_counts_toward_the_headline_hit_rate(self):
        """Its cause is unknowable, but the provider still said whether it hit, so the
        rate a reader looks at stays exact and only the attribution abstains."""
        cache = summarize(
            _Counters(sessions=1, turns=10, turns_with_usage=10, warm_turns=6, warm_hits=6, unordered_turns=4, unordered_hits=2)
        ).cache
        assert cache is not None
        assert cache.hits == 8
        assert cache.hit_rate_pct == pytest.approx(80.0)

    def test_a_router_with_no_cache_evidence_has_no_cache_view(self):
        assert summarize(_Counters(sessions=1, turns=5, turns_with_usage=0)).cache is None

    def test_ttl_follows_the_tier_the_majority_of_turns_used(self):
        five_minute = summarize(_Counters(sessions=1, turns=10, turns_with_usage=10, ephemeral_1h_turns=4)).cache
        one_hour = summarize(_Counters(sessions=1, turns=10, turns_with_usage=10, ephemeral_1h_turns=6)).cache
        assert five_minute is not None and one_hour is not None
        assert five_minute.ttl_seconds == 300.0
        assert one_hour.ttl_seconds == 3600.0


class TestTotals:
    def test_totals_sum_the_counters_rather_than_averaging_group_rates(self):
        """A big cheap router and a small expensive one must not be weighted equally."""
        response = build_response(
            rows=[
                _row(
                    "big",
                    turns=1000,
                    turns_with_usage=1000,
                    warm_turns=1000,
                    warm_hits=900,
                    first_visit_turns=0,
                    first_visit_hits=0,
                    expired_turns=0,
                    expired_hits=0,
                ),
                _row(
                    "small",
                    turns=10,
                    turns_with_usage=10,
                    warm_turns=10,
                    warm_hits=1,
                    first_visit_turns=0,
                    first_visit_hits=0,
                    expired_turns=0,
                    expired_hits=0,
                ),
            ],
            start_date=dt.date(2026, 7, 5),
            end_date=dt.date(2026, 8, 3),
        )
        assert response.routers_in_scope == 2
        assert response.totals.cache is not None
        assert response.totals.cache.hit_rate_pct == pytest.approx(100 * 901 / 1010)
        group_rates = [g.benchmark.cache.hit_rate_pct for g in response.groups if g.benchmark.cache]
        assert response.totals.cache.hit_rate_pct != pytest.approx(sum(group_rates) / len(group_rates))

    def test_totals_dollars_are_the_sum_of_every_router(self):
        response = build_response(
            rows=[_row("a", spend=3.0, baseline_spend=5.0), _row("b", spend=1.0, baseline_spend=9.0)],
            start_date=dt.date(2026, 7, 5),
            end_date=dt.date(2026, 8, 3),
        )
        assert response.totals.spend == pytest.approx(4.0)
        assert response.totals.baseline_spend == pytest.approx(14.0)
        assert response.totals.savings == pytest.approx(10.0)
        assert response.totals.sessions == 2

    def test_the_response_echoes_the_window_actually_read(self):
        """A caller asking for a year is told it got a month, not handed month-sized
        numbers under year-sized dates."""
        response = build_response(rows=[], start_date=dt.date(2026, 7, 5), end_date=dt.date(2026, 8, 3))
        assert (response.start_date, response.end_date) == (dt.date(2026, 7, 5), dt.date(2026, 8, 3))

    def test_an_empty_window_still_answers_with_zeroed_totals(self):
        response = build_response(rows=[], start_date=dt.date(2026, 8, 1), end_date=dt.date(2026, 8, 3))
        assert response.routers_in_scope == 0
        assert response.groups == ()
        assert response.totals.turns == 0

    def test_group_identity_is_carried_through(self):
        response = build_response(
            rows=[_row("claude-auto")], start_date=dt.date(2026, 8, 1), end_date=dt.date(2026, 8, 3)
        )
        assert response.groups[0].model_group == "claude-auto"
        assert response.groups[0].router_kind == "complexity"
        assert response.groups[0].baseline_model == "anthropic/claude-opus-4-8"


class TestWindow:
    def test_end_date_is_inclusive(self):
        start, end = clamp_window(dt.date(2026, 8, 3), dt.date(2026, 8, 3))
        assert start == dt.datetime(2026, 8, 3, tzinfo=dt.timezone.utc)
        assert end == dt.datetime(2026, 8, 4, tzinfo=dt.timezone.utc)

    def test_a_wider_request_is_clamped_to_the_cap_measured_in_dates_spanned(self):
        start, end = clamp_window(dt.date(2020, 1, 1), dt.date(2026, 8, 3))
        assert (end.date() - start.date()).days == MAX_WINDOW_DAYS
        assert start == dt.datetime(2026, 7, 5, tzinfo=dt.timezone.utc)

    def test_a_window_inside_the_cap_is_left_alone(self):
        start, end = clamp_window(dt.date(2026, 8, 1), dt.date(2026, 8, 3))
        assert start == dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
        assert end == dt.datetime(2026, 8, 4, tzinfo=dt.timezone.utc)
