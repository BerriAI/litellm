"""
Behavior tests for the LiteLLM_AutoRouterSession conditional upsert and the benchmarks
aggregate, against a real Postgres. The classification lives in SQL, so these tests are
the ones that exercise it; the builder and flush contracts are unit-tested in
tests/test_litellm/proxy/db/test_autorouter_session_rollup.py.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.db.autorouter_session_rollup import (
    AUTOROUTER_BENCHMARKS_SQL,
    UPSERT_AUTOROUTER_SESSION_SQL,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

T0 = datetime(2026, 8, 1, 12, 0, 0)


def _utc_epoch(moment: datetime) -> float:
    return moment.replace(tzinfo=timezone.utc).timestamp()


async def _turn(
    db,
    key: str,
    model: str,
    at: datetime,
    covered: int = 1,
    hit: int = 0,
    ttl: "int | None" = None,
    session_id: str = "s1",
    router: str = "auto-1",
    router_type: str = "complexity",
    tokens: int = 100,
    spend: float = 0.01,
    saved: float = 0.02,
    classifier_cost: float = 0.0,
    tier: "str | None" = None,
) -> None:
    touched: Final = 1 if (hit or ttl is not None or not covered) else 0
    await db.execute_raw(
        UPSERT_AUTOROUTER_SESSION_SQL,
        key,
        session_id,
        router,
        router_type,
        model,
        at.isoformat(),
        tokens,
        spend,
        saved,
        classifier_cost,
        covered,
        hit,
        ttl,
        touched,
        tier,
    )


async def _row(db, key: str, session_id: str = "s1", router: str = "auto-1") -> dict:
    rows = await db.query_raw(
        'SELECT * FROM "LiteLLM_AutoRouterSession" WHERE api_key = $1 AND session_id = $2 AND router_name = $3',
        key,
        session_id,
        router,
    )
    assert len(rows) == 1
    return rows[0]


async def test_every_turn_lands_in_exactly_one_bucket(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, ttl=300)
    await _turn(db, key, "A", T0 + timedelta(seconds=10), hit=1)
    await _turn(db, key, "B", T0 + timedelta(seconds=20), ttl=3600)
    await _turn(db, key, "A", T0 + timedelta(seconds=30), hit=1)
    await _turn(db, key, "B", T0 + timedelta(seconds=40))
    await _turn(db, key, "A", T0 + timedelta(seconds=500))
    await _turn(db, key, "A", T0 + timedelta(seconds=5))
    await _turn(db, key, "B", T0 + timedelta(seconds=600), covered=0)

    row = await _row(db, key)
    assert row["turns"] == 8
    assert row["same_model_turns"] == 1
    assert row["same_model_hits"] == 1
    assert row["first_visit_turns"] == 2
    assert row["first_visit_hits"] == 0
    assert row["return_turns"] == 4
    assert row["return_hits"] == 1
    assert row["unordered_turns"] == 1
    assert (
        row["same_model_turns"] + row["first_visit_turns"] + row["return_turns"] + row["unordered_turns"]
        == row["turns"]
    )
    assert row["covered_turns"] == 7
    assert row["cache_hits"] == 2
    assert row["ttl_5m_turns"] == 1
    assert row["ttl_1h_turns"] == 1


async def test_return_misses_attribute_against_the_recorded_ttl(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, ttl=300)
    await _turn(db, key, "B", T0 + timedelta(seconds=10), ttl=3600)
    await _turn(db, key, "A", T0 + timedelta(seconds=400))
    await _turn(db, key, "B", T0 + timedelta(seconds=410))

    row = await _row(db, key)
    assert row["return_expired_misses"] == 1
    assert row["return_within_ttl_misses"] == 1


async def test_a_return_miss_with_no_recorded_ttl_stays_unattributed(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0)
    await _turn(db, key, "B", T0 + timedelta(seconds=10))
    await _turn(db, key, "A", T0 + timedelta(seconds=20))

    row = await _row(db, key)
    assert row["return_turns"] == 1
    assert row["return_expired_misses"] == 0
    assert row["return_within_ttl_misses"] == 0


async def test_a_hit_refreshes_the_models_cache_clock(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, ttl=300)
    await _turn(db, key, "B", T0 + timedelta(seconds=250), ttl=3600)
    await _turn(db, key, "A", T0 + timedelta(seconds=290), hit=1)
    await _turn(db, key, "B", T0 + timedelta(seconds=300))
    await _turn(db, key, "A", T0 + timedelta(seconds=560))

    row = await _row(db, key)
    assert row["return_within_ttl_misses"] == 2
    assert row["return_expired_misses"] == 0
    assert row["models"]["A"]["at"] == pytest.approx(_utc_epoch(T0 + timedelta(seconds=290)), abs=1)


async def test_out_of_order_turns_do_not_rewind_the_session(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0 + timedelta(seconds=100), ttl=300)
    await _turn(db, key, "B", T0 + timedelta(seconds=200))
    await _turn(db, key, "A", T0)

    row = await _row(db, key)
    assert row["last_model"] == "B"
    assert row["unordered_turns"] == 1
    assert row["first_turn_at"].startswith("2026-08-01T12:00:00")
    assert row["last_turn_at"].startswith("2026-08-01T12:03:20")
    assert row["models"]["A"]["at"] == pytest.approx(_utc_epoch(T0 + timedelta(seconds=100)), abs=1)


async def test_concurrent_writers_compose_without_losing_turns(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, classifier_cost=0.001)
    await asyncio.gather(
        *(_turn(db, key, "A", T0 + timedelta(seconds=1 + offset), hit=1, classifier_cost=0.002) for offset in range(30))
    )
    row = await _row(db, key)
    assert row["turns"] == 31
    assert (
        row["same_model_turns"] + row["first_visit_turns"] + row["return_turns"] + row["unordered_turns"]
        == row["turns"]
    )
    assert row["spend"] == pytest.approx(0.31)
    assert row["saved_spend"] == pytest.approx(0.62)
    assert row["classifier_cost"] == pytest.approx(0.061)
    assert row["classifier_cost_recorded_turns"] == 31


async def _legacy_turn(db, key: str, at: datetime, session_id: str = "s1", router: str = "auto-1") -> None:
    await db.execute_raw(
        """INSERT INTO "LiteLLM_AutoRouterSession" AS t (
            api_key, session_id, router_name, router_type, first_turn_at, last_turn_at, last_model, turns, spend, saved_spend
        ) VALUES ($1, $2, $3, 'complexity', $4::timestamp, $4::timestamp, 'A', 1, 0.01, 0.02)
        ON CONFLICT (api_key, session_id, router_name) DO UPDATE SET
            turns = t.turns + 1, spend = t.spend + EXCLUDED.spend, saved_spend = t.saved_spend + EXCLUDED.saved_spend,
            last_turn_at = EXCLUDED.last_turn_at""",
        key,
        session_id,
        router,
        at.isoformat(),
    )


@pytest.mark.parametrize("writers", [(False,), (True,), (False, True), (True, False)])
async def test_subtotal_coverage_survives_legacy_and_rolling_writers(db, writers: tuple[bool, ...]):
    key: Final = f"k-{uuid.uuid4()}"
    for offset, records_cost in enumerate(writers):
        at: Final = T0 + timedelta(seconds=offset)
        if records_cost:
            await _turn(db, key, "A", at, classifier_cost=0.004)
        else:
            await _legacy_turn(db, key, at)

    row: Final = await _row(db, key)
    assert row["turns"] == len(writers)
    assert row["spend"] == pytest.approx(0.01 * len(writers))
    assert row["saved_spend"] == pytest.approx(0.02 * len(writers))
    assert row["classifier_cost"] == pytest.approx(0.004 * sum(writers))
    assert row["classifier_cost_recorded_turns"] == sum(writers)
    groups: Final = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL, T0.isoformat(), (T0 + timedelta(days=1)).isoformat(), key
    )
    assert len(groups) == 1
    assert groups[0]["classifier_cost"] == row["classifier_cost"]
    assert groups[0]["classifier_cost_recorded_turns"] == sum(writers)
    assert groups[0]["turns"] == len(writers)
    assert groups[0]["spend"] == row["spend"]
    assert groups[0]["saved_spend"] == row["saved_spend"]


async def test_the_benchmarks_aggregate_reads_only_overlapping_sessions(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    in_window = f"s-{uuid.uuid4()}"
    out_of_window = f"s-{uuid.uuid4()}"
    await _turn(
        db,
        key,
        "B",
        T0 + timedelta(seconds=60),
        session_id=in_window,
        router=router,
        saved=0.5,
        spend=0.25,
        classifier_cost=0.01,
    )
    await _turn(db, key, "A", T0, session_id=in_window, router=router, saved=0.5, spend=0.25, classifier_cost=0.02)
    await _turn(db, key, "A", T0 - timedelta(days=40), session_id=out_of_window, router=router, classifier_cost=9.0)

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        None,
    )
    matching = [row for row in rows if row["router_name"] == router]
    assert len(matching) == 1
    grouped = matching[0]
    assert grouped["router_type"] == "complexity"
    assert grouped["sessions"] == 1
    assert grouped["turns"] == 2
    assert grouped["spend"] == pytest.approx(0.5)
    assert grouped["saved_spend"] == pytest.approx(1.0)
    assert grouped["classifier_cost"] == pytest.approx(0.03)
    assert grouped["classifier_cost_recorded_turns"] == 2
    assert grouped["unordered_turns"] == 1
    assert grouped["session_seconds"] == pytest.approx(60.0)


async def test_the_benchmarks_aggregate_can_filter_to_one_key(db):
    router = f"r-{uuid.uuid4()}"
    first_key = f"k-{uuid.uuid4()}"
    second_key = f"k-{uuid.uuid4()}"
    await _turn(db, first_key, "A", T0, router=router, saved=0.5, classifier_cost=0.01)
    await _turn(db, second_key, "A", T0, router=router, saved=9.0, classifier_cost=0.09)

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        first_key,
    )
    matching = [row for row in rows if row["router_name"] == router]
    assert len(matching) == 1
    assert matching[0]["sessions"] == 1
    assert matching[0]["saved_spend"] == pytest.approx(0.5)
    assert matching[0]["classifier_cost"] == pytest.approx(0.01)
    assert matching[0]["classifier_cost_recorded_turns"] == 1

    unknown_key_rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        f"k-{uuid.uuid4()}",
    )
    assert [row for row in unknown_key_rows if row["router_name"] == router] == []


async def test_a_reconfigured_alias_reports_each_router_type_as_its_own_group(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, router_type="complexity")
    await _turn(
        db, key, "A", T0 + timedelta(seconds=10), session_id=f"s-{uuid.uuid4()}", router=router, router_type="quality"
    )

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        None,
    )
    matching = sorted(
        (row for row in rows if row["router_name"] == router),
        key=lambda row: row["router_type"],
    )
    assert [(row["router_type"], row["sessions"]) for row in matching] == [("complexity", 1), ("quality", 1)]


async def test_tier_turns_count_each_tier_that_served_a_turn(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, tier="simple")
    await _turn(db, key, "B", T0 + timedelta(seconds=10), tier="complex")
    await _turn(db, key, "A", T0 + timedelta(seconds=20), tier="simple")

    assert (await _row(db, key))["tier_turns"] == {"simple": 2, "complex": 1}


async def test_an_untiered_turn_increments_no_tier_counter(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, tier=None)
    assert (await _row(db, key))["tier_turns"] == {}

    await _turn(db, key, "A", T0 + timedelta(seconds=10), tier="medium")
    await _turn(db, key, "A", T0 + timedelta(seconds=20), tier=None)
    row = await _row(db, key)
    assert row["tier_turns"] == {"medium": 1}
    assert row["turns"] == 3


async def test_a_mid_session_router_type_change_keeps_foreign_tier_names_out_of_the_map(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, router_type="complexity", tier="medium")
    await _turn(db, key, "A", T0 + timedelta(seconds=10), router_type="quality", tier="2")
    await _turn(db, key, "A", T0 + timedelta(seconds=20), router_type="complexity", tier="medium")

    row = await _row(db, key)
    assert row["tier_turns"] == {"medium": 2}
    assert row["turns"] == 3


async def test_an_out_of_order_turn_still_counts_toward_its_tier(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0 + timedelta(seconds=60), tier="simple")
    await _turn(db, key, "A", T0, tier="simple")

    row = await _row(db, key)
    assert row["tier_turns"] == {"simple": 2}
    assert row["unordered_turns"] == 1


async def test_the_benchmarks_aggregate_sums_tier_turns_across_sessions(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, tier="simple")
    await _turn(db, key, "A", T0 + timedelta(seconds=10), session_id=f"s-{uuid.uuid4()}", router=router, tier="simple")
    await _turn(db, key, "B", T0 + timedelta(seconds=20), session_id=f"s-{uuid.uuid4()}", router=router, tier="complex")
    await _turn(db, key, "C", T0 + timedelta(seconds=30), session_id=f"s-{uuid.uuid4()}", router=router, tier=None)

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        None,
    )
    grouped = next(row for row in rows if row["router_name"] == router)
    assert grouped["tier_turns"] == {"simple": 2, "complex": 1}
    assert grouped["turns"] == 4


async def test_tier_maps_stay_separate_per_router_type_on_a_reconfigured_alias(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(
        db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, router_type="complexity", tier="medium"
    )
    await _turn(
        db,
        key,
        "A",
        T0 + timedelta(seconds=10),
        session_id=f"s-{uuid.uuid4()}",
        router=router,
        router_type="quality",
        tier="2",
    )

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        None,
    )
    by_type = {row["router_type"]: row["tier_turns"] for row in rows if row["router_name"] == router}
    assert by_type == {"complexity": {"medium": 1}, "quality": {"2": 1}}


async def test_a_window_with_no_tiered_turns_aggregates_to_an_empty_map(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, tier=None)

    rows = await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        (T0 - timedelta(days=1)).isoformat(),
        (T0 + timedelta(days=1)).isoformat(),
        None,
    )
    grouped = next(row for row in rows if row["router_name"] == router)
    assert grouped["tier_turns"] == {}


async def test_a_miss_that_touched_no_cache_does_not_advance_the_ttl_clock(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, ttl=300)
    await _turn(db, key, "B", T0 + timedelta(seconds=10), ttl=3600)
    await _turn(db, key, "A", T0 + timedelta(seconds=400))
    await _turn(db, key, "B", T0 + timedelta(seconds=410))
    await _turn(db, key, "A", T0 + timedelta(seconds=600))

    row = await _row(db, key)
    assert row["return_expired_misses"] == 2
    assert row["models"]["A"]["at"] == pytest.approx(_utc_epoch(T0), abs=1)


async def test_an_out_of_order_hit_still_counts_toward_the_overall_hit_rate(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0 + timedelta(seconds=100))
    await _turn(db, key, "A", T0 + timedelta(seconds=50), hit=1)

    row = await _row(db, key)
    assert row["unordered_turns"] == 1
    assert row["cache_hits"] == 1
    assert row["same_model_hits"] + row["first_visit_hits"] + row["return_hits"] == 0
