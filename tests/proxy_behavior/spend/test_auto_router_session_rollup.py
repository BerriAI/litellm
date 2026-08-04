"""How the auto-router rollup buckets a turn, evaluated by a real Postgres.

The upsert classifies each turn against the session's own cache record, so these assertions
are about SQL. They cover what the dashboard depends on: the three buckets partition every
turn, a tier that aged out is told from one that is still warm, the warming estimate is
priced on the prefix that was actually cached, and one caller cannot write into another's
rollup by reusing a session id.
"""

import datetime as dt
from dataclasses import replace

import pytest

from litellm.proxy.spend_tracking.auto_router_sessions import AutoRouterSessionQueue, TurnFacts

pytestmark = pytest.mark.asyncio(loop_scope="session")

HAIKU = "anthropic/claude-haiku-4-5"
OPUS = "anthropic/claude-opus-4-8"
T0 = dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.timezone.utc).timestamp()

TURN = TurnFacts(
    api_key="key-a",
    session_id="sess-1",
    model_group="claude-auto",
    router_kind="complexity",
    baseline_model=OPUS,
    model=HAIKU,
    started_at=T0,
    total_tokens=1000,
    spend=0.01,
    baseline_spend=0.05,
    cache_hit=False,
    cache_creation_tokens=2000,
    cached_prefix_tokens=2000,
    ttl_seconds=300.0,
)


async def _flush(client, turns) -> None:
    queue = AutoRouterSessionQueue()
    for turn in turns:
        await queue.update_queue.put(turn)
    await queue.flush(prisma_client=client)


async def _rows(db) -> list[dict]:
    return await db.query_raw('SELECT * FROM "LiteLLM_AutoRouterSession" ORDER BY api_key, session_id')


def _epoch_of(stored: str) -> float:
    parsed = dt.datetime.fromisoformat(stored)
    return (parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed).timestamp()


async def test_a_sessions_opening_turn_on_a_tier_is_a_first_visit(rollup_client, prisma_db):
    await _flush(rollup_client, [TURN])
    row = (await _rows(prisma_db))[0]
    assert (row["turns"], row["first_visit_turns"], row["warm_turns"], row["expired_turns"]) == (1, 1, 0, 0)
    assert row["tiers"] == {HAIKU: [T0, 300.0, 2000]}


async def test_a_second_tier_is_its_own_first_visit(rollup_client, prisma_db):
    await _flush(rollup_client, [TURN, replace(TURN, model=OPUS, started_at=T0 + 10)])
    row = (await _rows(prisma_db))[0]
    assert (row["turns"], row["first_visit_turns"]) == (2, 2)
    assert set(row["tiers"]) == {HAIKU, OPUS}


async def test_a_tier_used_again_inside_its_ttl_is_warm(rollup_client, prisma_db):
    await _flush(rollup_client, [TURN, replace(TURN, started_at=T0 + 60, cache_hit=True)])
    row = (await _rows(prisma_db))[0]
    assert (row["warm_turns"], row["warm_hits"], row["expired_turns"]) == (1, 1, 0)


async def test_a_tier_used_again_past_its_ttl_has_expired(rollup_client, prisma_db):
    await _flush(rollup_client, [TURN, replace(TURN, started_at=T0 + 900)])
    row = (await _rows(prisma_db))[0]
    assert (row["warm_turns"], row["expired_turns"], row["expired_hits"]) == (0, 1, 0)


async def test_expiry_measures_against_the_ttl_the_cache_was_written_with(rollup_client, prisma_db):
    """A one-hour entry is still warm at 30 minutes, even though the turn reading it
    reports no one-hour evidence of its own and so carries the five minute default."""
    await _flush(
        rollup_client,
        [
            replace(TURN, ttl_seconds=3600.0, cache_creation_tokens=2000),
            replace(TURN, started_at=T0 + 1800, ttl_seconds=300.0, cache_hit=True, cache_creation_tokens=0),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert (row["warm_turns"], row["expired_turns"]) == (1, 0)


async def test_the_three_buckets_partition_every_turn(rollup_client, prisma_db):
    await _flush(
        rollup_client,
        [
            TURN,
            replace(TURN, started_at=T0 + 10),
            replace(TURN, model=OPUS, started_at=T0 + 20),
            replace(TURN, started_at=T0 + 5000),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert row["first_visit_turns"] + row["warm_turns"] + row["expired_turns"] == row["turns"] == 4


async def test_a_turn_that_only_read_the_cache_leaves_the_written_terms_alone(rollup_client, prisma_db):
    await _flush(
        rollup_client,
        [
            replace(TURN, ttl_seconds=3600.0, cache_creation_tokens=5000, cached_prefix_tokens=5000),
            replace(TURN, started_at=T0 + 10, cache_hit=True, cache_creation_tokens=0, cached_prefix_tokens=5000),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert row["tiers"][HAIKU] == [T0 + 10, 3600.0, 5000]


async def test_counters_accumulate_across_flushes(rollup_client, prisma_db):
    await _flush(rollup_client, [TURN])
    await _flush(rollup_client, [replace(TURN, started_at=T0 + 10)])
    row = (await _rows(prisma_db))[0]
    assert row["turns"] == 2
    assert row["spend"] == pytest.approx(0.02)
    assert row["baseline_spend"] == pytest.approx(0.10)
    assert row["total_tokens"] == 2000
    assert _epoch_of(row["first_turn_at"]) == pytest.approx(T0, abs=0.001)
    assert _epoch_of(row["last_turn_at"]) == pytest.approx(T0 + 10, abs=0.001)


async def test_a_late_turn_cannot_rewind_the_session(rollup_client, prisma_db):
    await _flush(rollup_client, [replace(TURN, started_at=T0 + 600)])
    await _flush(rollup_client, [replace(TURN, started_at=T0)])
    row = (await _rows(prisma_db))[0]
    assert row["turns"] == 2
    assert row["tiers"][HAIKU][0] == T0 + 600
    assert _epoch_of(row["last_turn_at"]) == pytest.approx(T0 + 600, abs=0.001)
    assert _epoch_of(row["first_turn_at"]) == pytest.approx(T0, abs=0.001)


async def test_two_callers_reusing_one_session_id_keep_separate_rollups(rollup_client, prisma_db):
    await _flush(
        rollup_client,
        [
            replace(TURN, api_key="key-a", model=HAIKU),
            replace(TURN, api_key="key-b", model=OPUS, started_at=T0 + 10),
        ],
    )
    rows = await _rows(prisma_db)
    assert [row["api_key"] for row in rows] == ["key-a", "key-b"]
    assert all(row["turns"] == 1 and row["first_visit_turns"] == 1 for row in rows)


async def test_a_turn_arriving_before_an_already_recorded_one_is_not_called_warm(rollup_client, prisma_db):
    """Its cache state at its own time is unknowable, so it abstains rather than being
    guessed at; a negative idle gap is not evidence of warmth."""
    await _flush(rollup_client, [replace(TURN, started_at=T0 + 600)])
    await _flush(rollup_client, [replace(TURN, started_at=T0)])
    row = (await _rows(prisma_db))[0]
    assert row["turns"] == 2
    assert row["unordered_turns"] == 1
    assert row["warm_turns"] == 0
    assert row["expired_turns"] == 0
    assert row["first_visit_turns"] == 1
    assert row["tiers"][HAIKU][0] == T0 + 600


async def test_every_turn_lands_in_exactly_one_of_the_four_buckets(rollup_client, prisma_db):
    await _flush(
        rollup_client,
        [
            TURN,
            replace(TURN, started_at=T0 + 10),
            replace(TURN, model=OPUS, started_at=T0 + 20),
            replace(TURN, started_at=T0 + 5000),
        ],
    )
    await _flush(rollup_client, [replace(TURN, started_at=T0 + 5)])
    row = (await _rows(prisma_db))[0]
    buckets = row["first_visit_turns"] + row["warm_turns"] + row["expired_turns"] + row["unordered_turns"]
    assert buckets == row["turns"] == 5
    assert row["unordered_turns"] == 1


async def test_a_first_visit_to_a_new_tier_keeps_its_own_ttl_even_with_no_cache_write(rollup_client, prisma_db):
    await _flush(
        rollup_client,
        [
            TURN,
            replace(
                TURN,
                model=OPUS,
                started_at=T0 + 10,
                ttl_seconds=3600.0,
                cache_creation_tokens=0,
                cached_prefix_tokens=0,
            ),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert row["tiers"][OPUS] == [T0 + 10, 3600.0, 0]


async def test_a_growing_conversation_records_the_whole_live_prefix(rollup_client, prisma_db):
    """A warm turn on a growing prompt writes only the new segment, so recording
    cache_creation_tokens alone would shrink the prefix and under-price later replays.
    The live prefix is what was read plus what was written."""
    await _flush(
        rollup_client,
        [
            replace(TURN, cache_creation_tokens=2000, cached_prefix_tokens=2000),
            replace(TURN, started_at=T0 + 60, cache_hit=True, cache_creation_tokens=500, cached_prefix_tokens=2500),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert row["warm_turns"] == 1
    assert row["tiers"][HAIKU] == [T0 + 60, 300.0, 2500]


async def test_a_turn_that_touched_no_cache_is_left_out_of_the_cache_view(rollup_client, prisma_db):
    """A model with caching off would otherwise read as a wall of first visits and drag the
    hit rate down; it counts as a turn and nothing else."""
    await _flush(
        rollup_client,
        [
            replace(TURN, cache_creation_tokens=0, cached_prefix_tokens=0),
            replace(TURN, started_at=T0 + 10, cache_creation_tokens=0, cached_prefix_tokens=0),
        ],
    )
    row = (await _rows(prisma_db))[0]
    assert row["turns"] == 2
    assert row["turns_with_usage"] == 0
    assert row["first_visit_turns"] + row["warm_turns"] + row["expired_turns"] + row["unordered_turns"] == 0
