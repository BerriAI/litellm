from dataclasses import replace
from datetime import datetime, timezone

import pytest

from litellm.proxy.spend_tracking.auto_router_sessions import (
    EMPTY_SESSION_STATE,
    PROMPT_CACHE_TTL_SECONDS,
    SessionState,
    TurnFacts,
    fold_session,
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


class TestFoldSession:
    """An interval's turns are folded in start order, not in arrival order."""

    def test_turns_that_arrived_out_of_order_still_all_classify(self):
        """Two turns in flight together finish in whichever order the providers answer."""
        arrival_order = (turn(MODEL_B, at=60, created=5000), turn(MODEL_A, at=0, created=5000))
        folded = fold_session(EMPTY_SESSION_STATE, arrival_order, rates=rates)
        assert folded.turns == 2
        assert buckets(folded) == 2
        assert folded.state.last_model == MODEL_B

    def test_folding_an_interval_matches_folding_its_turns_one_at_a_time(self):
        turns = (
            turn(MODEL_A, at=0, created=5000),
            turn(MODEL_A, at=60, read=5000),
            turn(MODEL_B, at=400, created=5000),
            turn(MODEL_A, at=800, created=5000),
        )
        deltas, state = fold_all(turns)
        folded = fold_session(EMPTY_SESSION_STATE, turns, rates=rates)
        assert folded.turns == sum(d.turns for d in deltas)
        assert folded.return_turns == sum(d.return_turns for d in deltas)
        assert folded.replay_spend == pytest.approx(sum(d.replay_spend for d in deltas))
        assert folded.state == state

    def test_an_empty_interval_folds_to_the_state_it_was_given(self):
        _, state = fold_all((turn(MODEL_A, at=0, created=5000),))
        assert fold_session(state, (), rates=rates).state is state


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


class _StoredRow:
    """A session rollup as prisma hands it back."""

    def __init__(self, session_id: str, model_group: str, last_model: str, last_turn_at: float, model_state: dict):
        self.session_id = session_id
        self.model_group = model_group
        self.last_model = last_model
        self.last_turn_at = datetime.fromtimestamp(last_turn_at, tz=timezone.utc)
        self.model_state = model_state


class _RecordingTable:
    """A session-rollup table that can be told to fail on either side.

    Counts round trips the way prisma issues them: one `find_many` however many
    keys it selects, and one transaction however many upserts it carries.
    """

    def __init__(self, fail_read: bool = False, fail_write: bool = False, rows=()):
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.rows = rows
        self.reads = 0
        self.transactions = 0
        self.upserts: list = []

    async def find_many(self, where):
        self.reads += 1
        if self.fail_read:
            raise RuntimeError("transient database fault")
        wanted = {(pair["session_id"], pair["model_group"]) for pair in where["OR"]}
        return [row for row in self.rows if (row.session_id, row.model_group) in wanted]

    def commit(self, statements):
        self.transactions += 1
        if self.fail_write:
            raise RuntimeError("transient database fault")
        self.upserts.extend(statements)


class _RefillingTable(_RecordingTable):
    """Lets turns arrive in the middle of a flush, between the drain and the replay."""

    def __init__(self, arrive, **kwargs):
        super().__init__(**kwargs)
        self._arrive = arrive

    async def find_many(self, where):
        await self._arrive()
        return await super().find_many(where)


class _RecordingBatchActions:
    """Prisma's batcher: statements queue up and land only when the batch commits."""

    def __init__(self):
        self.queued: list = []

    def upsert(self, where, data):
        self.queued.append((where, data))


class _RecordingBatch:
    def __init__(self, table):
        self._table = table
        self.litellm_autoroutersession = _RecordingBatchActions()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc is None:
            self._table.commit(tuple(self.litellm_autoroutersession.queued))
        return False


class _RecordingTransaction:
    def __init__(self, table):
        self._table = table

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def batch_(self):
        return _RecordingBatch(self._table)


class _RecordingDb:
    def __init__(self, table):
        self.litellm_autoroutersession = table

    def tx(self, timeout=None):
        return _RecordingTransaction(self.litellm_autoroutersession)


class _RecordingPrisma:
    def __init__(self, table):
        self.db = _RecordingDb(table)


def _turn_at(at: float, model: str = MODEL_A) -> TurnFacts:
    return turn(model, at=at, created=5000)


def _stored_session(session_id: str = "s1", model_group: str = "g") -> _StoredRow:
    """A session last served on MODEL_B that has already used MODEL_A."""
    return _StoredRow(
        session_id=session_id,
        model_group=model_group,
        last_model=MODEL_B,
        last_turn_at=60.0,
        model_state={
            MODEL_A: {"last_used_at": 0.0, "provisioned_replay_spend": 0.0},
            MODEL_B: {"last_used_at": 60.0, "provisioned_replay_spend": 0.0},
        },
    )


@pytest.mark.asyncio
class TestTheLoggingPathNeverTouchesTheDatabase:
    """Staging a turn must not put a database round trip in front of spend tracking."""

    async def test_turns_can_be_recorded_with_no_database_in_sight(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        queue = AutoRouterSessionQueue()
        for at in (0, 60, 120):
            await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(at))

        table = _RecordingTable()
        assert await queue.flush(_RecordingPrisma(table)) == 1
        assert table.upserts[0][1]["create"]["turns"] == 3

    async def test_a_session_costs_one_read_per_flush_at_most_not_one_per_turn(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        for at in (0, 60, 120):
            await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(at))
        await queue.flush(prisma)

        assert table.reads == 1

        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(180))
        await queue.flush(prisma)
        assert table.reads == 1


@pytest.mark.asyncio
class TestFlushDurability:
    """A transient database fault must not delete an interval of traffic or its history."""

    async def test_a_failed_write_is_restaged_rather_than_dropped(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail_write=True)
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0))

        assert await queue.flush(prisma) == 0

        table.fail_write = False
        assert await queue.flush(prisma) == 1
        assert table.upserts[0][1]["create"]["turns"] == 1

    async def test_a_restaged_batch_merges_under_turns_staged_since(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail_write=True)
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0))
        await queue.flush(prisma)

        table.fail_write = False
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(60))
        assert await queue.flush(prisma) == 1
        assert table.upserts[0][1]["create"]["turns"] == 2

    async def test_a_failed_state_read_writes_nothing_at_all(self):
        """The corrupting move was folding onto an empty state and persisting it."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail_read=True, rows=(_stored_session(),))
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(120))

        assert await queue.flush(prisma) == 0
        assert table.upserts == []

    async def test_history_survives_a_failed_read_and_still_classifies_the_turn(self):
        """Returning to a tier this session already used is a return, not a first visit."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail_read=True, rows=(_stored_session(),))
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(120))
        await queue.flush(prisma)

        table.fail_read = False
        assert await queue.flush(prisma) == 1
        update = table.upserts[0][1]["update"]
        assert update["return_turns"] == {"increment": 1}
        assert update["first_visit_turns"] == {"increment": 0}
        assert update["turns"] == {"increment": 1}

    async def test_a_successful_flush_stages_nothing_back(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        prisma = _RecordingPrisma(_RecordingTable())
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0))
        assert await queue.flush(prisma) == 1
        assert await queue.flush(prisma) == 0


@pytest.mark.asyncio
class TestTheRowRecordsTheFoldsAnswer:
    """Session progress on the row comes from the folded state, not the staged turns.

    They agree until a turn arrives late, at which point `fold_turn` refuses to
    advance the state and the staged turns still carry the late timestamp.
    """

    async def test_a_late_turn_does_not_rewind_the_rows_last_activity(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(rows=(_stored_session(),))
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(30))

        assert await queue.flush(_RecordingPrisma(table)) == 1
        update = table.upserts[0][1]["update"]
        assert update["last_turn_at"] == datetime.fromtimestamp(60.0, tz=timezone.utc)

    async def test_a_turn_that_did_advance_the_session_moves_last_activity_forward(self):
        """The guard must not freeze the column for ordinary in-order traffic."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(rows=(_stored_session(),))
        queue = AutoRouterSessionQueue()
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(300))

        assert await queue.flush(_RecordingPrisma(table)) == 1
        update = table.upserts[0][1]["update"]
        assert update["last_turn_at"] == datetime.fromtimestamp(300.0, tz=timezone.utc)


@pytest.mark.asyncio
class TestStagingCostsTheSamePerTurn:
    """Staging must not re-copy the turns already staged for that session.

    Rebuilding the buffer per turn is quadratic in the length of a session, so a
    caller reusing one `session_id` pays for the whole interval on every request
    while holding the queue's lock.
    """

    async def test_a_turn_is_appended_to_the_existing_buffer_not_a_fresh_copy(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        queue = AutoRouterSessionQueue()
        key = ("s1", "g")
        await queue.record_turn(key, "complexity", None, _turn_at(0))
        buffer = queue._pending[key].turns

        for at in (60, 120, 180):
            await queue.record_turn(key, "complexity", None, _turn_at(at))

        assert queue._pending[key].turns is buffer
        assert [turn.started_at for turn in buffer] == [0, 60, 120, 180]

    async def test_a_long_session_still_folds_every_turn_it_staged(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        queue = AutoRouterSessionQueue()
        for at in range(500):
            await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(at * 60))

        assert await queue.flush(_RecordingPrisma(table)) == 1
        assert table.upserts[0][1]["create"]["turns"] == 500


@pytest.mark.asyncio
class TestStagingIsBounded:
    """`session_id` is caller-controlled, so what is held between flushes needs a ceiling."""

    async def test_turns_are_refused_once_the_staging_is_full(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        queue = AutoRouterSessionQueue(max_staged_turns=2)
        for i in range(5):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(0))

        assert await queue.flush(_RecordingPrisma(table)) == 2

    async def test_the_ceiling_counts_turns_not_sessions(self):
        """One caller replaying a long session must not evade the memory bound."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        queue = AutoRouterSessionQueue(max_staged_turns=2)
        for at in (0, 60, 120, 180):
            await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(at))

        await queue.flush(_RecordingPrisma(table))
        assert table.upserts[0][1]["create"]["turns"] == 2

    async def test_staging_reopens_once_it_has_drained(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue(max_staged_turns=1)
        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(0))
        await queue.flush(prisma)

        await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(60))
        assert await queue.flush(prisma) == 1
        assert table.upserts[1][1]["update"]["turns"] == {"increment": 1}

    async def test_a_replayed_batch_is_bounded_by_the_same_ceiling(self):
        """A database that stays down must not grow the staging one failed flush at a time.

        The window that matters is inside the flush: the drain has already zeroed
        the counter, turns keep arriving against it, and only then is the failed
        batch put back. A ceiling checked by the arriving path alone lets the
        replay land on top of a staging that is already full.
        """
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        queue = AutoRouterSessionQueue(max_staged_turns=2)

        async def arrive_mid_flush():
            for at in (200, 260):
                await queue.record_turn(("s2", "g"), "complexity", None, _turn_at(at))

        table = _RefillingTable(arrive_mid_flush, fail_write=True)
        for at in (0, 60):
            await queue.record_turn(("s1", "g"), "complexity", None, _turn_at(at))

        await queue.flush(_RecordingPrisma(table))

        held = sum(len(pending.turns) for pending in queue._pending.values())
        assert held <= 2, f"staging grew past its ceiling to {held} turns"


def test_chunking_caps_what_one_statement_carries_and_keeps_it_in_key_order():
    """Key order is the lock order two pods draining the same sessions have to agree on."""
    from litellm.proxy.spend_tracking.auto_router_session_queue import SESSIONS_PER_STATEMENT, _chunked

    batch = {(f"s{i:05d}", "g"): i for i in range(2500)}
    chunks = _chunked(batch, SESSIONS_PER_STATEMENT)

    assert [len(chunk) for chunk in chunks] == [1000, 1000, 500]
    assert [key for chunk in chunks for key in chunk] == sorted(batch)


@pytest.mark.asyncio
class TestFlushCostDoesNotGrowWithSessionCount:
    """A read and an upsert per session is two round trips per session; a flush must not do that."""

    async def test_many_sessions_cost_one_read_and_one_transaction(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        queue = AutoRouterSessionQueue()
        for i in range(250):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(0))

        assert await queue.flush(_RecordingPrisma(table)) == 250
        assert (table.reads, table.transactions) == (1, 1)
        assert len(table.upserts) == 250

    async def test_sessions_this_pod_has_already_folded_cost_no_read_at_all(self):
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable()
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        for i in range(50):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(0))
        await queue.flush(prisma)

        for i in range(50):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(60))
        assert await queue.flush(prisma) == 50
        assert (table.reads, table.transactions) == (1, 2)

    async def test_a_chunk_that_could_not_be_written_restages_every_session_in_it(self):
        """The transaction is the failure unit, so nothing in it landed and nothing may be dropped."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(fail_write=True)
        prisma = _RecordingPrisma(table)
        queue = AutoRouterSessionQueue()
        for i in range(5):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(0))

        assert await queue.flush(prisma) == 0
        assert table.upserts == []

        table.fail_write = False
        assert await queue.flush(prisma) == 5
        written = {where["session_id_model_group"]["session_id"] for where, _ in table.upserts}
        assert written == {f"s{i}" for i in range(5)}

    async def test_one_read_serves_a_chunk_of_sessions_that_all_have_history(self):
        """Every session's own row has to come back from the batched read, not just the first."""
        from litellm.proxy.spend_tracking.auto_router_session_queue import AutoRouterSessionQueue

        table = _RecordingTable(rows=tuple(_stored_session(session_id=f"s{i}") for i in range(5)))
        queue = AutoRouterSessionQueue()
        for i in range(5):
            await queue.record_turn((f"s{i}", "g"), "complexity", None, _turn_at(120))

        assert await queue.flush(_RecordingPrisma(table)) == 5
        assert table.reads == 1
        assert all(data["update"]["return_turns"] == {"increment": 1} for _, data in table.upserts)


def _router_with_auto_routers():
    """A real router carrying one auto-router of each kind that builds without extra packages.

    A semantic auto-router needs the ``semantic_router`` package to initialize, so
    it is exercised against the registries directly in
    ``TestResolvingOneGroupsKind`` rather than here.
    """
    from litellm import Router

    return Router(
        model_list=[
            {
                "model_name": "adaptive-complexity-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": ["cheap"], "MEDIUM": ["cheap"], "COMPLEX": ["pricey"]},
                        "adaptive": True,
                    },
                    "complexity_router_default_model": "cheap",
                },
            },
            {
                "model_name": "quality-router",
                "litellm_params": {
                    "model": "auto_router/quality_router",
                    "quality_router_config": {"complexity_to_quality": {"SIMPLE": 1, "MEDIUM": 2, "COMPLEX": 3}},
                    "quality_router_default_model": "cheap",
                },
            },
            {"model_name": "cheap", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"}},
            {"model_name": "pricey", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"}},
        ]
    )


class _RegistriesOnly:
    """Just the four registries a router keys its pre-routing strategies by."""

    def __init__(self, complexity=(), quality=(), adaptive=(), semantic=()):
        self.complexity_routers = dict.fromkeys(complexity, ())
        self.quality_routers = dict.fromkeys(quality, ())
        self.adaptive_routers = dict.fromkeys(adaptive, ())
        self.auto_routers = dict.fromkeys(semantic, ())


class TestTheGateRecognisesAnAutoRouterGroup:
    """The write path gates on membership only. Which strategy ran is the router's
    answer, recorded on the routing decision, not something re-derived here."""

    @pytest.mark.parametrize("registry", ["complexity", "quality", "adaptive", "semantic"])
    def test_a_group_in_any_registry_passes_the_gate(self, registry):
        from litellm.proxy.spend_tracking.auto_router_sessions import serves_an_auto_router

        assert serves_an_auto_router(_RegistriesOnly(**{registry: ("a-router",)}), "a-router")

    def test_a_plain_model_group_does_not(self):
        from litellm.proxy.spend_tracking.auto_router_sessions import serves_an_auto_router

        assert not serves_an_auto_router(_RegistriesOnly(complexity=("a-router",)), "gpt-4o")


class TestTheWriteAndReadPathsAgreeOnWhatAnAutoRouterIs:
    """The rollup is written per request and read per group, off two different
    derivations of the same fact. A group the writer files under one kind and the
    dashboard labels another, or filters out entirely, is a benchmark that reads
    empty for traffic that really happened."""

    def test_every_group_the_dashboard_asks_about_passes_the_write_paths_gate(self):
        from litellm.proxy.spend_tracking.auto_router_sessions import auto_router_group_kinds, serves_an_auto_router

        router = _router_with_auto_routers()
        group_kinds = auto_router_group_kinds(router)

        assert dict(group_kinds) == {"adaptive-complexity-router": "complexity", "quality-router": "quality"}
        assert all(serves_an_auto_router(router, group) for group in group_kinds)

    def test_a_complexity_router_running_the_bandit_is_still_a_complexity_router(self):
        """It owns an entry in both registries, so the lookup order decides, and only
        one of the two orders agrees with what the dashboard labels the group."""
        router = _router_with_auto_routers()

        assert "adaptive-complexity-router" in router.complexity_routers
        assert "adaptive-complexity-router" in router.adaptive_routers

    def test_a_group_no_auto_router_serves_is_left_out_of_both(self):
        from litellm.proxy.spend_tracking.auto_router_sessions import auto_router_group_kinds, serves_an_auto_router

        router = _router_with_auto_routers()

        assert "cheap" not in auto_router_group_kinds(router)
        assert not serves_an_auto_router(router, "cheap")
