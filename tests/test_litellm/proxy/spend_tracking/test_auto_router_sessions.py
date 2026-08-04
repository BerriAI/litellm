from dataclasses import replace
from datetime import datetime, timezone

import pytest

from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    PROMPT_CACHE_TTL_SECONDS,
    SessionState,
    TurnFacts,
    fold_turn,
    merge_deltas,
    state_from_row,
    state_to_json,
    turn_ttl_seconds,
)

MODEL_A = "anthropic/claude-haiku-4-5"
MODEL_B = "anthropic/claude-sonnet-4-5"

READ_RATE = 1e-6
WRITE_RATE = 5e-6
FIVE_MIN = PROMPT_CACHE_TTL_SECONDS["5m"]
ONE_HOUR = PROMPT_CACHE_TTL_SECONDS["1h"]


def rates(model: str, ttl_seconds: int) -> tuple[float, float]:
    """Fixed prices so dollar assertions do not move with the cost map."""
    return READ_RATE, WRITE_RATE


def turn(
    model: str = MODEL_A,
    at: float = 0.0,
    read: int = 0,
    created: int = 0,
    spend: float = 0.01,
    savings: float = 0.0,
    ephemeral_5m: int | None = None,
    ephemeral_1h: int = 0,
) -> TurnFacts:
    return TurnFacts(
        model=model,
        started_at=at,
        prompt_tokens=1000,
        completion_tokens=100,
        total_tokens=1100,
        cache_read_tokens=read,
        cache_creation_tokens=created,
        ephemeral_5m_tokens=created if ephemeral_5m is None else ephemeral_5m,
        ephemeral_1h_tokens=ephemeral_1h,
        spend=spend,
        autorouter_savings=savings,
        has_usage=True,
    )


def fold_all(turns: tuple[TurnFacts, ...]):
    """Fold a whole session, returning every delta in order plus the final state."""
    state = EMPTY_SESSION_STATE
    deltas = []
    for one in turns:
        delta = fold_turn(state, one, rates=rates)
        deltas.append(delta)
        state = delta.state
    return tuple(deltas), state


def buckets(delta) -> int:
    return delta.same_model_turns + delta.first_visit_turns + delta.return_turns


class TestBucketExhaustiveness:
    def test_every_turn_lands_in_exactly_one_bucket(self):
        deltas, _ = fold_all(
            (
                turn(MODEL_A, at=0, created=5000),
                turn(MODEL_A, at=60, read=5000),
                turn(MODEL_B, at=120, created=5000),
                turn(MODEL_A, at=180, read=5000),
                turn(MODEL_B, at=240, read=5000),
            )
        )
        assert [buckets(d) for d in deltas] == [1, 1, 1, 1, 1]
        assert sum(buckets(d) for d in deltas) == sum(d.turns for d in deltas)

    def test_the_opening_turn_of_a_session_is_a_first_visit(self):
        """The old split left it in no bucket, so bucket totals undercounted."""
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates)
        assert delta.first_visit_turns == 1
        assert buckets(delta) == delta.turns == 1

    def test_arriving_at_an_unused_model_is_a_first_visit_not_a_return(self):
        deltas, _ = fold_all((turn(MODEL_A, at=0), turn(MODEL_B, at=60)))
        assert deltas[1].first_visit_turns == 1
        assert deltas[1].return_turns == 0

    def test_staying_on_the_same_model_is_not_a_return(self):
        deltas, _ = fold_all((turn(MODEL_A, at=0), turn(MODEL_A, at=60)))
        assert deltas[1].same_model_turns == 1
        assert deltas[1].return_turns == 0


class TestHits:
    def test_a_turn_that_read_from_cache_is_a_hit_in_its_own_bucket(self):
        deltas, _ = fold_all((turn(MODEL_A, at=0, created=5000), turn(MODEL_A, at=60, read=5000)))
        assert deltas[1].same_model_hits == 1
        assert deltas[0].first_visit_hits == 0

    def test_a_hit_is_read_tokens_not_a_provider_flag(self):
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, read=1, created=0), rates=rates)
        assert delta.first_visit_hits == 1


class TestStaleAndSavableReturns:
    def _return_after(self, idle: float):
        state = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates).state
        state = fold_turn(state, turn(MODEL_B, at=1, created=5000), rates=rates).state
        return fold_turn(state, turn(MODEL_A, at=idle, created=5000), rates=rates)

    def test_return_inside_the_ttl_is_neither_stale_nor_savable(self):
        delta = self._return_after(FIVE_MIN - 10)
        assert delta.return_turns == 1
        assert delta.stale_return_misses == 0
        assert delta.savable_return_misses == 0
        assert delta.rescued_spend == 0.0

    def test_return_past_the_ttl_within_two_ttls_is_savable_and_rescues_the_write(self):
        delta = self._return_after(FIVE_MIN + 10)
        assert delta.stale_return_misses == 1
        assert delta.savable_return_misses == 1
        assert delta.rescued_spend == pytest.approx(5000 * (WRITE_RATE - READ_RATE))

    def test_return_past_two_ttls_is_stale_but_not_savable(self):
        delta = self._return_after(2 * FIVE_MIN + 10)
        assert delta.stale_return_misses == 1
        assert delta.savable_return_misses == 0
        assert delta.rescued_spend == 0.0

    def test_a_return_that_hit_is_never_counted_as_a_miss(self):
        state = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates).state
        state = fold_turn(state, turn(MODEL_B, at=1, created=5000), rates=rates).state
        delta = fold_turn(state, turn(MODEL_A, at=FIVE_MIN + 10, read=5000), rates=rates)
        assert delta.return_hits == 1
        assert delta.stale_return_misses == 0
        assert delta.savable_return_misses == 0


class TestWarmingReplayEconomics:
    def test_replay_is_withdrawn_when_the_session_returns_inside_the_ttl(self):
        """No refresher would have fired, so the provisional charge comes back off."""
        deltas, _ = fold_all((turn(MODEL_A, at=0, created=5000), turn(MODEL_A, at=60, read=5000)))
        assert deltas[0].replay_spend == pytest.approx(5000 * READ_RATE)
        assert deltas[1].replay_spend == pytest.approx(0.0)

    def test_replay_is_kept_when_the_session_stays_away_past_the_ttl(self):
        deltas, _ = fold_all(
            (turn(MODEL_A, at=0, created=5000), turn(MODEL_A, at=FIVE_MIN + 10, created=5000))
        )
        assert sum(d.replay_spend for d in deltas) == pytest.approx(2 * 5000 * READ_RATE)

    def test_total_replay_is_each_bridged_gap_plus_one_final_abandon(self):
        """Reproduces what the window-function query summed, without the window."""
        deltas, _ = fold_all(
            (
                turn(MODEL_A, at=0, created=5000),
                turn(MODEL_A, at=60, read=5000),
                turn(MODEL_A, at=1400, created=5000),
            )
        )
        bridged_gap = 5000 * READ_RATE
        final_abandon = 5000 * READ_RATE
        assert sum(d.replay_spend for d in deltas) == pytest.approx(bridged_gap + final_abandon)

    def test_each_model_a_session_touches_carries_its_own_abandon_charge(self):
        deltas, _ = fold_all((turn(MODEL_A, at=0, created=5000), turn(MODEL_B, at=60, created=3000)))
        assert sum(d.replay_spend for d in deltas) == pytest.approx((5000 + 3000) * READ_RATE)

    def test_prefix_for_replay_counts_read_and_written_tokens(self):
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, read=2000, created=3000), rates=rates)
        assert delta.replay_spend == pytest.approx(5000 * READ_RATE)


class TestTtlSelection:
    def test_defaults_to_five_minutes_without_ephemeral_evidence(self):
        """Both counters zero must not read as the one hour tier."""
        assert turn_ttl_seconds(turn(MODEL_A, created=0, ephemeral_5m=0, ephemeral_1h=0)) == FIVE_MIN

    def test_one_hour_when_the_turn_wrote_mostly_to_the_long_cache(self):
        assert turn_ttl_seconds(turn(MODEL_A, ephemeral_5m=100, ephemeral_1h=5000)) == ONE_HOUR

    def test_five_minutes_when_the_turn_wrote_mostly_to_the_short_cache(self):
        assert turn_ttl_seconds(turn(MODEL_A, ephemeral_5m=5000, ephemeral_1h=100)) == FIVE_MIN

    def test_staleness_follows_the_turns_own_ttl(self):
        state = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates).state
        state = fold_turn(state, turn(MODEL_B, at=1, created=5000), rates=rates).state
        inside_the_hour = fold_turn(
            state, turn(MODEL_A, at=1800, created=5000, ephemeral_5m=0, ephemeral_1h=5000), rates=rates
        )
        assert inside_the_hour.stale_return_misses == 0


class TestOutOfOrderTurns:
    def test_a_late_turn_keeps_its_spend_but_not_its_classification(self):
        first = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=100, created=5000, spend=0.02), rates=rates)
        late = fold_turn(first.state, turn(MODEL_B, at=50, created=5000, spend=0.03), rates=rates)
        assert late.turns == 1
        assert late.spend == 0.03
        assert buckets(late) == 0
        assert late.replay_spend == 0.0

    def test_a_late_turn_does_not_rewrite_the_sessions_state(self):
        first = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=100), rates=rates)
        late = fold_turn(first.state, turn(MODEL_B, at=50), rates=rates)
        assert late.state is first.state
        assert late.state.last_model == MODEL_A


class TestBaselineSpend:
    def test_baseline_is_what_was_paid_plus_what_routing_saved(self):
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, spend=0.01, savings=0.09), rates=rates)
        assert delta.baseline_spend == pytest.approx(0.10)

    def test_a_route_that_lost_money_reports_a_baseline_below_actual_spend(self):
        """Savings are signed, so a cache-thrashing route must stay visible as a loss."""
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, spend=0.05, savings=-0.02), rates=rates)
        assert delta.baseline_spend == pytest.approx(0.03)
        assert delta.baseline_spend < delta.spend


class TestCoverage:
    def test_a_turn_without_cache_reporting_still_counts_as_a_turn(self):
        """Coverage separates "logging is off" from "the cache was cold"."""
        facts = replace(turn(MODEL_A, at=0), has_usage=False)
        delta = fold_turn(EMPTY_SESSION_STATE, facts, rates=rates)
        assert delta.turns == 1
        assert delta.turns_with_usage == 0

    def test_a_turn_that_reported_usage_counts_toward_coverage(self):
        delta = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates)
        assert delta.turns_with_usage == 1


class TestStateRoundTrip:
    def test_state_survives_a_trip_through_the_row(self):
        _, state = fold_all((turn(MODEL_A, at=0, created=5000), turn(MODEL_B, at=60, created=3000)))
        restored = state_from_row(
            state.last_model,
            datetime.fromtimestamp(state.last_turn_at, tz=timezone.utc),
            state_to_json(state),
        )
        assert restored.last_model == state.last_model
        assert restored.last_turn_at == pytest.approx(state.last_turn_at)
        assert set(restored.model_marks) == set(state.model_marks)
        for model, mark in state.model_marks.items():
            assert restored.model_marks[model].last_used_at == pytest.approx(mark.last_used_at)
            assert restored.model_marks[model].provisioned_replay_spend == pytest.approx(
                mark.provisioned_replay_spend
            )

    def test_a_restored_session_classifies_the_next_turn_the_same_way(self):
        """This is the property that makes a pod hop or a restart harmless."""
        _, state = fold_all((turn(MODEL_A, at=0, created=5000), turn(MODEL_B, at=60, created=5000)))
        restored = state_from_row(
            state.last_model, datetime.fromtimestamp(state.last_turn_at, tz=timezone.utc), state_to_json(state)
        )
        next_turn = turn(MODEL_A, at=FIVE_MIN + 100, created=5000)
        assert fold_turn(restored, next_turn, rates=rates) == fold_turn(state, next_turn, rates=rates)

    def test_an_unreadable_state_blob_resets_history_instead_of_raising(self):
        restored = state_from_row("m", datetime.now(timezone.utc), {"bad": "shape"})
        assert restored is EMPTY_SESSION_STATE

    def test_a_naive_timestamp_is_read_as_utc(self):
        naive = datetime(2026, 8, 1, 12, 0, 0)
        aware = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert state_from_row(None, naive, {}).last_turn_at == state_from_row(None, aware, {}).last_turn_at


class TestMergeDeltas:
    def test_counters_add_and_the_later_state_wins(self):
        first = fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates)
        second = fold_turn(first.state, turn(MODEL_A, at=60, read=5000), rates=rates)
        merged = merge_deltas(first, second)
        assert merged.turns == 2
        assert merged.first_visit_turns == 1
        assert merged.same_model_turns == 1
        assert merged.spend == pytest.approx(first.spend + second.spend)
        assert merged.replay_spend == pytest.approx(first.replay_spend + second.replay_spend)
        assert merged.state == second.state

    def test_merging_a_whole_session_matches_folding_it_turn_by_turn(self):
        deltas, _ = fold_all(
            (
                turn(MODEL_A, at=0, created=5000),
                turn(MODEL_A, at=60, read=5000),
                turn(MODEL_B, at=400, created=5000),
                turn(MODEL_A, at=800, created=5000),
            )
        )
        merged = deltas[0]
        for delta in deltas[1:]:
            merged = merge_deltas(merged, delta)
        assert merged.turns == 4
        assert merged.same_model_turns + merged.first_visit_turns + merged.return_turns == 4
        assert merged.replay_spend == pytest.approx(sum(d.replay_spend for d in deltas))


class TestEmptySessionState:
    def test_a_fresh_session_has_no_model_history(self):
        assert EMPTY_SESSION_STATE.last_model is None
        assert EMPTY_SESSION_STATE.model_marks == {}

    def test_folding_never_mutates_the_state_it_was_given(self):
        before = state_to_json(EMPTY_SESSION_STATE)
        fold_turn(EMPTY_SESSION_STATE, turn(MODEL_A, at=0, created=5000), rates=rates)
        assert state_to_json(EMPTY_SESSION_STATE) == before

    def test_session_state_is_hashable_free_of_shared_mutation(self):
        _, state = fold_all((turn(MODEL_A, at=0, created=5000),))
        snapshot = dict(state.model_marks)
        fold_turn(state, turn(MODEL_B, at=60, created=5000), rates=rates)
        assert dict(state.model_marks) == snapshot


def test_session_state_type_is_frozen():
    _, state = fold_all((turn(MODEL_A, at=0),))
    with pytest.raises(Exception):
        state.last_model = "other"  # pyright: ignore[reportAttributeAccessIssue]  # asserting frozen


def test_state_from_row_without_a_timestamp_starts_at_the_epoch():
    assert state_from_row(None, None, {}) == SessionState(
        last_model=None, last_turn_at=0.0, model_marks={}
    )


class _RecordingTable:
    """A session-rollup table that can be told to fail."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.upserts: list = []

    async def find_unique(self, where):
        return None

    async def upsert(self, where, data):
        if self.fail:
            raise RuntimeError("transient database fault")
        self.upserts.append((where, data))


class _RecordingPrisma:
    def __init__(self, table):
        self.db = type("_Db", (), {"litellm_autoroutersession": table})()


def _turn_at(at: float, model: str = MODEL_A) -> TurnFacts:
    return turn(model, at=at, created=5000)


@pytest.mark.asyncio
class TestFlushDurability:
    """A transient write fault must not silently delete an interval of traffic."""

    async def test_a_failed_write_is_restaged_rather_than_dropped(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail=True)
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0), prisma)

        assert await queue.flush(prisma) == 0

        table.fail = False
        assert await queue.flush(prisma) == 1
        assert table.upserts[0][1]["create"]["turns"] == 1

    async def test_a_restaged_batch_merges_under_turns_staged_since(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail=True)
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0), prisma)
        await queue.flush(prisma)

        table.fail = False
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(60), prisma)
        assert await queue.flush(prisma) == 1
        # Both turns land, once each
        assert table.upserts[0][1]["create"]["turns"] == 2

    async def test_a_successful_flush_stages_nothing_back(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        prisma = _RecordingPrisma(_RecordingTable())
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0), prisma)
        assert await queue.flush(prisma) == 1
        assert await queue.flush(prisma) == 0


@pytest.mark.asyncio
class TestPendingIsBounded:
    """`session_id` is caller-controlled, so the staged aggregate needs a ceiling."""

    async def test_new_sessions_are_refused_once_the_aggregate_is_full(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        prisma = _RecordingPrisma(_RecordingTable())
        queue = AutoRouterSessionQueue(max_tracked_sessions=2)
        for i in range(5):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(0), prisma)

        assert await queue.flush(prisma) == 2

    async def test_a_session_already_staged_keeps_accumulating_at_the_cap(self):
        """Refusing new keys must not stall the conversations already in flight."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue(max_tracked_sessions=1)
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0), prisma)
        await queue.record_turn(("s2", "g"), "complexity", None, _turn_at(0), prisma)
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(60), prisma)

        await queue.flush(prisma)
        assert len(table.upserts) == 1
        assert table.upserts[0][1]["create"]["turns"] == 2
