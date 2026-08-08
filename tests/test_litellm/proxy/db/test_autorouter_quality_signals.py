"""
Unit tests for the auto-router quality signal computation (escalation, abandonment).

These test the pure logic in litellm.proxy.db.autorouter_quality_signals against
plain Turn objects and a fixed rank table, so they exercise the actual escalation/
abandonment/eligibility rules without touching Postgres or a real Router.
"""

import pytest

from litellm.proxy.db.autorouter_quality_signals import (
    MIN_COHORT_SESSIONS,
    MIN_SESSION_ID_COVERAGE,
    Turn,
    baseline_unavailable_reason,
    could_escalate,
    session_escalated,
    signals_for_cohort,
)
from litellm.router import Router

# haiku < sonnet < opus, matching how rank_models_by_cost would order real deployments
RANKS = {"haiku": 0, "sonnet": 1, "opus": 2}
REACHABLE_BY_KEY = {"key1": ("haiku", "sonnet", "opus")}


def _turn(
    session_id="s1",
    model="sonnet",
    started_at=0.0,
    client_disconnected=False,
    has_client_session_id=True,
    api_key="key1",
):
    return Turn(
        session_id=session_id,
        api_key=api_key,
        model=model,
        started_at=started_at,
        client_disconnected=client_disconnected,
        router_name="auto-router",
        has_client_session_id=has_client_session_id,
    )


class TestRankModelsByCost:
    def test_equal_cost_models_receive_equal_rank(self, monkeypatch: pytest.MonkeyPatch):
        import litellm.proxy.db.autorouter_quality_signals as module

        def _fake_priced(router, candidate):
            cost = {"cheap-a": 1.0, "cheap-b": 1.0, "dear": 2.0}[candidate.model]
            return (cost, candidate)

        monkeypatch.setattr("litellm.router_strategy.savings_baseline._priced", _fake_priced)
        ranks = module.rank_models_by_cost(Router(model_list=[]), ["cheap-a", "cheap-b", "dear"])
        assert ranks["cheap-a"] == ranks["cheap-b"], "same-cost models must share the same rank"
        assert ranks["dear"] > ranks["cheap-a"], "costlier model must rank higher"


class TestSessionEscalated:
    def test_upward_move_is_escalation(self):
        turns = [_turn(model="sonnet", started_at=1), _turn(model="opus", started_at=2)]
        assert session_escalated(turns, RANKS) is True

    def test_downward_move_is_not_escalation(self):
        # This is the exact case the plan explicitly excludes: sonnet -> haiku is a cost
        # or latency choice, not a quality miss, and must not be counted.
        turns = [_turn(model="sonnet", started_at=1), _turn(model="haiku", started_at=2)]
        assert session_escalated(turns, RANKS) is False

    def test_same_model_every_turn_is_not_escalation(self):
        turns = [_turn(model="sonnet", started_at=1), _turn(model="sonnet", started_at=2)]
        assert session_escalated(turns, RANKS) is False

    def test_single_turn_session_cannot_escalate(self):
        assert session_escalated([_turn(model="sonnet", started_at=1)], RANKS) is False

    def test_escalation_detected_regardless_of_input_order(self):
        # Turns can arrive from the DB in any order; escalation must be judged on
        # started_at, not on list position.
        turns = [_turn(model="opus", started_at=2), _turn(model="sonnet", started_at=1)]
        assert session_escalated(turns, RANKS) is True

    def test_unrankable_model_does_not_count_as_escalation(self):
        turns = [_turn(model="sonnet", started_at=1), _turn(model="mystery-model", started_at=2)]
        assert session_escalated(turns, RANKS) is False

    def test_late_escalation_is_still_detected(self):
        turns = [
            _turn(model="sonnet", started_at=1),
            _turn(model="sonnet", started_at=2),
            _turn(model="sonnet", started_at=3),
            _turn(model="opus", started_at=4),
        ]
        assert session_escalated(turns, RANKS) is True


class TestCouldEscalate:
    def test_session_on_cheapest_model_with_pricier_model_reachable_could_escalate(self):
        turns = [_turn(model="haiku")]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet", "opus"]) is True

    def test_session_already_on_most_expensive_reachable_model_could_not_escalate(self):
        # This is the case the plan calls out: a session pinned to the priciest model it
        # could reach had nowhere to go, so it must not silently count as "no miss".
        turns = [_turn(model="opus")]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet", "opus"]) is False

    def test_reachable_set_without_anything_costlier_cannot_escalate(self):
        turns = [_turn(model="sonnet")]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet"]) is False

    def test_no_rankable_turns_cannot_escalate(self):
        turns = [_turn(model="unknown-model")]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet", "opus"]) is False

    def test_eligibility_is_judged_from_the_opening_turn_not_the_ceiling_reached(self):
        # A session that started on sonnet and already escalated to opus must still count
        # as "could escalate" -- judging from the max model used would make every session
        # that actually escalated look ineligible, driving the measured rate toward zero
        # exactly as the true rate rises.
        turns = [_turn(model="sonnet", started_at=1), _turn(model="opus", started_at=2)]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet", "opus"]) is True

    def test_eligibility_uses_first_turn_by_time_not_by_list_order(self):
        turns = [_turn(model="opus", started_at=2), _turn(model="sonnet", started_at=1)]
        assert could_escalate(turns, RANKS, reachable=["haiku", "sonnet", "opus"]) is True


class TestSignalsForCohort:
    def test_escalation_rate_counts_only_eligible_sessions(self):
        # s1 escalates and could have; s2 sits on opus (the ceiling) so it is excluded from
        # the denominator entirely, not counted as a non-escalating session.
        turns = [
            _turn(session_id="s1", model="sonnet", started_at=1),
            _turn(session_id="s1", model="opus", started_at=2),
            _turn(session_id="s2", model="opus", started_at=1),
        ]
        result = signals_for_cohort(turns, RANKS, REACHABLE_BY_KEY)
        assert result.sessions == 1
        assert result.escalation_rate_pct == 100.0

    def test_abandonment_counted_over_eligible_turns_only(self):
        turns = [
            _turn(session_id="s1", model="haiku", started_at=1, client_disconnected=True),
            _turn(session_id="s1", model="haiku", started_at=2, client_disconnected=False),
            # s2 is on the ceiling model and ineligible; its disconnect must not be counted.
            _turn(session_id="s2", model="opus", started_at=1, client_disconnected=True),
        ]
        result = signals_for_cohort(turns, RANKS, REACHABLE_BY_KEY)
        assert result.sessions == 1
        assert result.abandonment_rate_pct == 50.0

    def test_no_eligible_sessions_returns_none_rates_not_zero(self):
        # Every session already on the ceiling model: zero would misleadingly claim
        # "no miss detected" when in fact nothing could be measured.
        turns = [_turn(session_id="s1", model="opus")]
        result = signals_for_cohort(turns, RANKS, REACHABLE_BY_KEY)
        assert result.sessions == 0
        assert result.escalation_rate_pct is None
        assert result.abandonment_rate_pct is None

    def test_downward_switch_never_inflates_escalation_rate(self):
        # Opens on sonnet (eligible: opus is reachable and costlier), then moves down to
        # haiku. The downward move must not be read as an escalation.
        turns = [
            _turn(session_id="s1", model="sonnet", started_at=1),
            _turn(session_id="s1", model="haiku", started_at=2),
        ]
        result = signals_for_cohort(turns, RANKS, REACHABLE_BY_KEY)
        assert result.sessions == 1
        assert result.escalation_rate_pct == 0.0

    def test_same_session_id_from_two_keys_is_never_spliced_into_one_escalation(self):
        # Two different callers' keys reuse the same session id. Splicing them into one
        # session would read key1's haiku turn followed by key2's opus turn as one caller
        # escalating -- an escalation neither caller ever made. Kept separate, each is its
        # own single-turn session and neither can escalate within itself.
        turns = [
            _turn(session_id="shared", api_key="key1", model="haiku", started_at=1),
            _turn(session_id="shared", api_key="key2", model="opus", started_at=2),
        ]
        reachable = {"key1": ("haiku", "sonnet", "opus"), "key2": ("haiku", "sonnet", "opus")}
        result = signals_for_cohort(turns, RANKS, reachable)
        assert result.escalation_rate_pct == 0.0

    def test_reachability_is_scoped_to_each_session_own_key(self):
        # key1 can only ever reach haiku; key2 can reach up to opus. A pooled reachable set
        # would let key1's session borrow key2's ceiling and count as eligible when it never
        # had anywhere to escalate to.
        turns = [
            _turn(session_id="s1", api_key="key1", model="haiku", started_at=1),
            _turn(session_id="s2", api_key="key2", model="haiku", started_at=1),
            _turn(session_id="s2", api_key="key2", model="opus", started_at=2),
        ]
        reachable = {"key1": ("haiku",), "key2": ("haiku", "sonnet", "opus")}
        result = signals_for_cohort(turns, RANKS, reachable)
        assert result.sessions == 1
        assert result.escalation_rate_pct == 100.0


class TestBaselineUnavailableReason:
    def test_low_session_id_coverage_reported_before_size(self):
        turns = [_turn(has_client_session_id=False) for _ in range(100)]
        from litellm.proxy.db.autorouter_quality_signals import CohortSignals

        cohort = CohortSignals(sessions=100, escalation_rate_pct=1.0, abandonment_rate_pct=1.0)
        assert baseline_unavailable_reason(turns, cohort) == "no_session_ids"

    def test_high_session_id_coverage_with_too_few_sessions_reports_size(self):
        from litellm.proxy.db.autorouter_quality_signals import CohortSignals

        turns = [_turn(has_client_session_id=True) for _ in range(10)]
        cohort = CohortSignals(sessions=MIN_COHORT_SESSIONS - 1, escalation_rate_pct=1.0, abandonment_rate_pct=1.0)
        assert baseline_unavailable_reason(turns, cohort) == "insufficient_sessions"

    def test_sufficient_coverage_and_size_reports_available(self):
        from litellm.proxy.db.autorouter_quality_signals import CohortSignals

        turns = [_turn(has_client_session_id=True) for _ in range(10)]
        cohort = CohortSignals(sessions=MIN_COHORT_SESSIONS, escalation_rate_pct=1.0, abandonment_rate_pct=1.0)
        assert baseline_unavailable_reason(turns, cohort) is None

    def test_coverage_exactly_at_floor_is_available(self):
        from litellm.proxy.db.autorouter_quality_signals import CohortSignals

        n = 100
        with_id = int(n * MIN_SESSION_ID_COVERAGE)
        turns = [_turn(has_client_session_id=True) for _ in range(with_id)] + [
            _turn(has_client_session_id=False) for _ in range(n - with_id)
        ]
        cohort = CohortSignals(sessions=MIN_COHORT_SESSIONS, escalation_rate_pct=1.0, abandonment_rate_pct=1.0)
        assert baseline_unavailable_reason(turns, cohort) is None

    def test_no_turns_at_all_only_checked_against_size_floor(self):
        from litellm.proxy.db.autorouter_quality_signals import CohortSignals

        cohort = CohortSignals(sessions=0, escalation_rate_pct=None, abandonment_rate_pct=None)
        assert baseline_unavailable_reason([], cohort) == "insufficient_sessions"
