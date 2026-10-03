"""
Behavior tests for the LiteLLM_AutoRouterSession conditional upsert and the benchmarks
aggregate, against a real Postgres. The classification lives in SQL, so these tests are
the ones that exercise it; the builder and flush contracts are unit-tested in
tests/unit/proxy/db/test_autorouter_session_rollup.py.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final, TypedDict, cast

import pytest
from prisma import Prisma
from prisma.errors import RawQueryError
from typing_extensions import ReadOnly

from litellm.proxy.db.autorouter_session_rollup import (
    AUTOROUTER_BENCHMARKS_SQL,
    UPSERT_AUTOROUTER_SESSION_SQL,
    UPSERT_AUTOROUTER_USER_SESSION_SQL,
    AutoRouterTurnTransaction,
    flush_autorouter_turn_transactions,
    write_autorouter_turn,
)
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup

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
    baseline: "str | None" = None,
    estimated: bool = True,
    user_id: str = "",
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
        baseline,
        int(estimated),
        spend if estimated else 0.0,
        saved if estimated else 0.0,
        user_id,
    )


async def _benchmark_rows(
    db, start: datetime, end: datetime, key: str | None = None, user_id: str | None = None
) -> list[dict]:
    return await db.query_raw(
        AUTOROUTER_BENCHMARKS_SQL,
        start.isoformat(),
        end.isoformat(),
        key,
        user_id,
        start.date().isoformat(),
        (end - timedelta(days=1)).date().isoformat(),
    )


async def _days(db, key: str | None = None, user_id: str | None = None, router: str | None = None) -> list[dict]:
    rows = await _benchmark_rows(db, T0 - timedelta(days=1), T0 + timedelta(days=2), key, user_id)
    return [row for row in rows if row["turns"] and (router is None or row["router_name"] == router)]


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
    assert row["savings_estimated_turns"] == sum(writers)
    assert row["savings_estimated_actual_spend"] == pytest.approx(0.01 * sum(writers))
    assert row["savings_estimated_saved_spend"] == pytest.approx(0.02 * sum(writers))
    days: Final = await _days(db, key)
    assert len(days) == int(any(writers))
    for day in days:
        assert day["classifier_cost"] == row["classifier_cost"]
        assert day["classifier_cost_recorded_turns"] == day["turns"] == sum(writers)
        assert day["spend"] == pytest.approx(0.01 * sum(writers))
        assert day["saved_spend"] == pytest.approx(0.02 * sum(writers))
        assert day["savings_estimated_actual_spend"] == row["savings_estimated_actual_spend"]
        assert day["savings_estimated_saved_spend"] == row["savings_estimated_saved_spend"]


async def test_unknown_and_legacy_turns_preserve_actual_spend_without_entering_the_estimated_cohort(db: Prisma) -> None:
    key: Final = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, spend=0.25, saved=-0.05, baseline="opus")
    await _turn(
        db, key, "B", T0 + timedelta(seconds=1), spend=0.7, saved=0, baseline="sonnet", estimated=False
    )
    await _legacy_turn(db, key, T0 + timedelta(seconds=2))

    row: Final = await _row(db, key)
    assert row["saved_spend"] == pytest.approx(-0.03)
    assert row["savings_estimated_baseline_models"] == {"opus": 1}
    (day,) = await _days(db, key)
    assert (row["turns"], day["turns"]) == (3, 2)
    assert (row["spend"], day["spend"]) == (pytest.approx(0.96), pytest.approx(0.95))
    for actual in (row, day):
        assert actual["savings_estimated_turns"] == 1
        assert actual["savings_estimated_actual_spend"] == pytest.approx(0.25)
        assert actual["savings_estimated_saved_spend"] == pytest.approx(-0.05)


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

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, None)
    matching = [row for row in rows if row["router_name"] == router]
    assert len(matching) == 1
    grouped = matching[0]
    assert grouped["router_type"] == "complexity"
    assert grouped["sessions"] == 1
    assert grouped["session_turns"] == 2
    assert grouped["unordered_turns"] == 1
    assert grouped["session_seconds"] == pytest.approx(60.0)
    (day,) = await _days(db, router=router)
    assert (day["turns"], day["classifier_cost_recorded_turns"]) == (2, 2)
    assert day["spend"] == pytest.approx(0.5)
    assert day["saved_spend"] == pytest.approx(1.0)
    assert day["classifier_cost"] == pytest.approx(0.03)


async def test_the_benchmarks_aggregate_can_filter_to_one_key(db):
    router = f"r-{uuid.uuid4()}"
    first_key = f"k-{uuid.uuid4()}"
    second_key = f"k-{uuid.uuid4()}"
    await _turn(db, first_key, "A", T0, router=router, saved=0.5, classifier_cost=0.01)
    await _turn(db, second_key, "A", T0, router=router, saved=9.0, classifier_cost=0.09)

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), first_key, None)
    matching = [row for row in rows if row["router_name"] == router]
    assert len(matching) == 1
    assert matching[0]["sessions"] == 1
    (day,) = await _days(db, first_key, router=router)
    assert day["saved_spend"] == pytest.approx(0.5)
    assert day["classifier_cost"] == pytest.approx(0.01)
    assert day["classifier_cost_recorded_turns"] == 1

    unknown_key_rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), f"k-{uuid.uuid4()}", None)
    assert [row for row in unknown_key_rows if row["router_name"] == router] == []


class _BenchmarkRow(TypedDict):
    sessions: ReadOnly[int]
    session_turns: ReadOnly[int]
    turns: ReadOnly[int]
    same_model_turns: ReadOnly[int]
    first_visit_turns: ReadOnly[int]
    spend: ReadOnly[float]
    saved_spend: ReadOnly[float]
    tier_turns: ReadOnly[dict[str, int]]
    cache_hits: ReadOnly[int]
    savings_estimated_turns: ReadOnly[int]
    savings_estimated_actual_spend: ReadOnly[float]
    savings_estimated_saved_spend: ReadOnly[float]


async def _scoped_benchmarks(
    db: Prisma, router: str, user_id: str | None = None, key: str | None = None
) -> tuple[_BenchmarkRow, ...]:
    rows: Final = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), key, user_id)
    days: Final = await _days(db, key, user_id, router)
    return tuple(
        cast(_BenchmarkRow, {**row, **next(iter(days), {})}) for row in rows if row["router_name"] == router
    )


async def test_users_keep_written_identity_across_shared_keys_and_keyless_sessions(db: Prisma) -> None:
    router: Final = f"r-{uuid.uuid4()}"
    alice: Final = f"u-{uuid.uuid4()}"
    bob: Final = f"u-{uuid.uuid4()}"
    first_key: Final = f"k-{uuid.uuid4()}"
    second_key: Final = f"k-{uuid.uuid4()}"
    await _legacy_turn(db, first_key, T0, router=router)
    await _turn(db, first_key, "A", T0 + timedelta(seconds=10), router=router, user_id=alice, tier="simple")
    await _turn(
        db, first_key, "B", T0 + timedelta(seconds=20), router=router, user_id=bob, spend=0.03, saved=0.06, tier="complex"
    )
    await _turn(db, second_key, "C", T0, router=router, user_id=alice, spend=0.02, saved=0.04)
    await _turn(db, "", "A", T0, router=router, user_id=alice, ttl=300)
    await _turn(db, "", "A", T0 + timedelta(seconds=1), router=router, user_id=alice, hit=1)
    await _turn(db, "", "B", T0, router=router, user_id=bob, spend=0.04, saved=0.08)
    await _turn(db, second_key, "C", T0 - timedelta(days=40), router=router, user_id=alice, session_id="expired")

    alice_rows: Final = await _scoped_benchmarks(db, router, user_id=alice)
    bob_rows: Final = await _scoped_benchmarks(db, router, user_id=bob)
    global_rows: Final = await _scoped_benchmarks(db, router)
    key_rows: Final = await _scoped_benchmarks(db, router, key=first_key)
    intersection: Final = await _scoped_benchmarks(db, router, user_id=alice, key=first_key)
    assert len(alice_rows) == len(bob_rows) == len(global_rows) == len(key_rows) == len(intersection) == 1
    assert (alice_rows[0]["sessions"], alice_rows[0]["turns"], alice_rows[0]["same_model_turns"]) == (3, 4, 1)
    assert (alice_rows[0]["session_turns"], bob_rows[0]["session_turns"]) == (4, 2)
    assert (bob_rows[0]["sessions"], bob_rows[0]["turns"], bob_rows[0]["first_visit_turns"]) == (2, 2, 2)
    assert alice_rows[0]["spend"] == pytest.approx(0.05)
    assert bob_rows[0]["spend"] == pytest.approx(0.07)
    assert alice_rows[0]["tier_turns"] == {"simple": 1}
    assert bob_rows[0]["tier_turns"] == {"complex": 1}
    assert (alice_rows[0]["cache_hits"], bob_rows[0]["cache_hits"]) == (1, 0)
    assert (global_rows[0]["sessions"], global_rows[0]["session_turns"], global_rows[0]["turns"]) == (4, 7, 6)
    assert (alice_rows[0]["savings_estimated_turns"], bob_rows[0]["savings_estimated_turns"]) == (4, 2)
    assert global_rows[0]["savings_estimated_turns"] == 6
    for scoped in (alice_rows[0], bob_rows[0]):
        assert scoped["savings_estimated_actual_spend"] == pytest.approx(scoped["spend"])
        assert scoped["savings_estimated_saved_spend"] == pytest.approx(scoped["saved_spend"])
    assert global_rows[0]["spend"] == pytest.approx(alice_rows[0]["spend"] + bob_rows[0]["spend"])
    assert global_rows[0]["saved_spend"] == pytest.approx(alice_rows[0]["saved_spend"] + bob_rows[0]["saved_spend"])
    assert global_rows[0]["tier_turns"] == {"simple": 1, "complex": 1}
    assert (key_rows[0]["sessions"], key_rows[0]["session_turns"], key_rows[0]["turns"]) == (1, 3, 2)
    assert key_rows[0]["spend"] == pytest.approx(0.04)
    assert (intersection[0]["sessions"], intersection[0]["turns"]) == (1, 1)
    assert intersection[0]["spend"] == pytest.approx(0.01)
    assert await _scoped_benchmarks(db, router, user_id=bob, key=second_key) == ()
    assert await _scoped_benchmarks(db, router, user_id=f"u-{uuid.uuid4()}") == ()
    assert [
        row
        for row in await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, "")
        if row["router_name"] == router
    ] == []


async def test_a_failed_user_projection_rolls_back_the_keys_increment(db: Prisma) -> None:
    key: Final = f"k-{uuid.uuid4()}"
    user_id: Final = "".join(str(uuid.uuid4()) for _ in range(200))
    await _turn(db, key, "A", T0)
    before: Final = await _row(db, key)

    with pytest.raises(RawQueryError, match=r"index row (requires|size)"):
        await _turn(db, key, "B", T0 + timedelta(seconds=1), user_id=user_id)

    assert await _row(db, key) == before
    assert await db.query_raw('SELECT user_id FROM "LiteLLM_AutoRouterUserSession" WHERE user_id = $1', user_id) == []
    assert [day["turns"] for day in await _days(db, key)] == [1]

    first_user: Final = f"u-{uuid.uuid4()}"
    second_user: Final = f"u-{uuid.uuid4()}"
    turns: Final = tuple(
        AutoRouterTurnTransaction(
            api_key=key,
            user_id=user,
            session_id="s1",
            router_name="auto-1",
            router_type="complexity",
            model=model,
            turn_at=T0 + timedelta(seconds=second),
            total_tokens=100,
            spend=0.01,
            saved_spend=0.02,
            classifier_cost=0.0,
            covered=True,
            cache_hit=False,
            cache_ttl_seconds=None,
            cache_touched=False,
        )
        for user, model, second in (
            (first_user, "A", 1),
            (user_id, "B", 2),
            (first_user, "B", 3),
            (second_user, "C", 4),
            (first_user, "B", 5),
            (second_user, "C", 6),
            (user_id, "A", 7),
        )
    )
    await flush_autorouter_turn_transactions(SimpleNamespace(db=db), tuple(reversed(turns)), n_retry_times=0)

    key_row: Final = await _row(db, key)
    assert (key_row["turns"], key_row["last_model"], key_row["unordered_turns"]) == (2, "A", 0)
    assert key_row["spend"] == pytest.approx(0.02)
    user_rows: Final = await db.query_raw('SELECT * FROM "LiteLLM_AutoRouterUserSession" WHERE api_key = $1', key)
    by_user: Final = {row["user_id"]: row for row in user_rows}
    assert set(by_user) == {first_user, second_user}
    for user, count, model in ((first_user, 3, "B"), (second_user, 2, "C")):
        row: Final = by_user[user]
        assert (row["turns"], row["same_model_turns"], row["unordered_turns"], row["last_model"]) == (count, 1, 0, model)
        assert row["spend"] == pytest.approx(count * 0.01)
        assert row["saved_spend"] == pytest.approx(count * 0.02)
    days: Final = await db.query_raw(
        'SELECT user_id, turns, saved_spend FROM "LiteLLM_AutoRouterDailySpend" WHERE api_key = $1', key
    )
    assert {day["user_id"]: (day["turns"], day["saved_spend"]) for day in days} == {
        "": (1, pytest.approx(0.02)),
        first_user: (3, pytest.approx(0.06)),
        second_user: (2, pytest.approx(0.04)),
    }


async def test_user_session_cleanup_keeps_another_users_recent_keyless_session(db: Prisma) -> None:
    router: Final = f"r-{uuid.uuid4()}"
    expired_user: Final = f"u-{uuid.uuid4()}"
    recent_user: Final = f"u-{uuid.uuid4()}"
    await _turn(db, "", "A", T0 - timedelta(days=1), router=router, user_id=expired_user)
    await _turn(db, "", "A", T0 + timedelta(days=1), router=router, user_id=recent_user)
    cleaner: Final = SpendLogCleanup(general_settings={})

    await cleaner._delete_old_autorouter_user_session_rows(
        SimpleNamespace(db=db), T0.replace(tzinfo=timezone.utc), time.monotonic() + 60
    )

    assert await db.query_raw(
        'SELECT user_id, turns FROM "LiteLLM_AutoRouterUserSession" WHERE router_name = $1', router
    ) == [{"user_id": recent_user, "turns": 1}]


async def test_a_reconfigured_alias_reports_each_router_type_as_its_own_group(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, router_type="complexity")
    await _turn(
        db, key, "A", T0 + timedelta(seconds=10), session_id=f"s-{uuid.uuid4()}", router=router, router_type="quality"
    )

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, None)
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


async def test_baseline_models_count_the_turns_priced_against_each_baseline(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, baseline="opus")
    await _turn(db, key, "B", T0 + timedelta(seconds=10), baseline="opus")
    await _turn(db, key, "A", T0 + timedelta(seconds=20), baseline="sonnet")

    assert (await _row(db, key))["baseline_models"] == {"opus": 2, "sonnet": 1}


async def test_a_turn_priced_against_no_baseline_leaves_the_map_alone(db):
    key = f"k-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, baseline=None)
    assert (await _row(db, key))["baseline_models"] == {}

    await _turn(db, key, "A", T0 + timedelta(seconds=10), baseline="opus")
    await _turn(db, key, "A", T0 + timedelta(seconds=20), baseline=None)
    row = await _row(db, key)
    assert row["baseline_models"] == {"opus": 1}
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

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, None)
    grouped = next(row for row in rows if row["router_name"] == router)
    assert grouped["tier_turns"] == {"simple": 2, "complex": 1}
    assert grouped["session_turns"] == 4


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

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, None)
    by_type = {row["router_type"]: row["tier_turns"] for row in rows if row["router_name"] == router}
    assert by_type == {"complexity": {"medium": 1}, "quality": {"2": 1}}


async def test_a_window_with_no_tiered_turns_aggregates_to_an_empty_map(db):
    key = f"k-{uuid.uuid4()}"
    router = f"r-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, session_id=f"s-{uuid.uuid4()}", router=router, tier=None)

    rows = await _benchmark_rows(db, (T0 - timedelta(days=1)), (T0 + timedelta(days=1)), None, None)
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


async def test_a_cross_midnight_session_splits_its_money_by_request_day(db):
    key = f"k-{uuid.uuid4()}"
    router = f"auto-{uuid.uuid4()}"
    midnight = datetime(2026, 9, 2)
    await _turn(db, key, "A", midnight - timedelta(minutes=10), router=router, spend=1.0, saved=7.0, user_id="u1")
    await _turn(db, key, "A", midnight + timedelta(minutes=10), router=router, spend=1.0, saved=3.0, user_id="u1")
    await _turn(db, key, "B", midnight + timedelta(days=1), router=router, spend=1.0, saved=11.0, user_id="u1")

    assert (await _row(db, key, router=router))["saved_spend"] == 21.0
    days = await db.query_raw(
        'SELECT date, turns, saved_spend FROM "LiteLLM_AutoRouterDailySpend" WHERE api_key = $1 ORDER BY date', key
    )
    assert [(d["date"], d["turns"], d["saved_spend"]) for d in days] == [
        ("2026-09-01", 1, 7.0),
        ("2026-09-02", 1, 3.0),
        ("2026-09-03", 1, 11.0),
    ]
    for user_id in (None, "u1"):
        (selected,) = await _benchmark_rows(db, midnight, midnight + timedelta(days=1), key, user_id)
        assert (selected["sessions"], selected["session_turns"]) == (1, 3)
        assert (selected["turns"], selected["spend"], selected["saved_spend"]) == (1, 1.0, 3.0)


async def test_a_router_type_change_within_a_day_keeps_each_types_money_apart(db):
    key = f"k-{uuid.uuid4()}"
    router = f"auto-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, router=router, router_type="complexity", spend=1.0, saved=4.0)
    await _turn(db, key, "A", T0 + timedelta(hours=1), router=router, router_type="quality", spend=2.0, saved=0.0)

    days = {day["router_type"]: (day["turns"], day["spend"], day["saved_spend"]) for day in await _days(db, key)}
    assert days == {"complexity": (1, 1.0, 4.0), "quality": (1, 2.0, 0.0)}


async def test_a_router_type_change_mid_session_keeps_session_shape_with_the_sessions_type(db):
    key = f"k-{uuid.uuid4()}"
    router = f"auto-{uuid.uuid4()}"
    await _turn(db, key, "A", T0, router=router, router_type="complexity", spend=1.0, saved=4.0)
    await _turn(db, key, "B", T0 + timedelta(hours=1), router=router, router_type="quality", spend=2.0, saved=0.0)

    rows = {row["router_type"]: row for row in await _benchmark_rows(db, T0, T0 + timedelta(days=1), key)}
    assert set(rows) == {"complexity", "quality"}
    assert (rows["complexity"]["sessions"], rows["complexity"]["session_turns"], rows["complexity"]["turns"]) == (1, 2, 1)
    assert (rows["quality"]["sessions"], rows["quality"]["session_turns"], rows["quality"]["turns"]) == (0, 0, 1)
    assert rows["quality"]["spend"] == 2.0


@pytest.mark.parametrize("statement", [UPSERT_AUTOROUTER_SESSION_SQL, UPSERT_AUTOROUTER_USER_SESSION_SQL])
async def test_a_sessionless_turn_writes_its_router_day_row_and_no_session_row(db, statement: str):
    key = f"k-{uuid.uuid4()}"
    router = f"auto-{uuid.uuid4()}"
    for offset in range(2):
        await write_autorouter_turn(
            db,
            AutoRouterTurnTransaction(
                api_key=key,
                user_id="u-sessionless",
                session_id="",
                router_name=router,
                router_type="complexity",
                model="A",
                turn_at=T0 + timedelta(seconds=offset),
                total_tokens=10,
                spend=1.0,
                saved_spend=2.0,
                classifier_cost=0.1,
                covered=True,
                cache_hit=False,
                cache_ttl_seconds=None,
                cache_touched=True,
                savings_estimated_turns=1,
                savings_estimated_actual_spend=1.0,
                savings_estimated_saved_spend=2.0,
            ),
            statement,
        )

    (day,) = await _days(db, key, router=router)
    assert (day["turns"], day["spend"], day["saved_spend"], day["classifier_cost"]) == (2, 2.0, 4.0, 0.2)
    assert (day["sessions"], day["session_turns"]) == (0, 0)
    for table in ("LiteLLM_AutoRouterSession", "LiteLLM_AutoRouterUserSession"):
        assert await db.query_raw(f'SELECT 1 FROM "{table}" WHERE router_name = $1', router) == []


async def test_router_day_money_reconciles_with_the_overall_daily_total_including_sessionless_requests(db):
    from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key

    key = f"k-{uuid.uuid4()}"
    router = f"auto-{uuid.uuid4()}"
    requests = (("session-1", 0.25, 1.5), ("session-1", 0.5, 2.0), ("", 0.1, 0.25))
    for offset, (session_id, spend, saved) in enumerate(requests):
        await write_autorouter_turn(
            db,
            AutoRouterTurnTransaction(
                api_key=key,
                user_id="u1",
                session_id=session_id,
                router_name=router,
                router_type="complexity",
                model="A",
                turn_at=T0 + timedelta(seconds=offset),
                total_tokens=10,
                spend=spend,
                saved_spend=saved,
                classifier_cost=0.0,
                covered=True,
                cache_hit=False,
                cache_ttl_seconds=None,
                cache_touched=True,
                savings_estimated_turns=1,
                savings_estimated_actual_spend=spend,
                savings_estimated_saved_spend=saved,
            ),
        )
    table = DAILY_SPEND_TABLES["user"]
    statement, values = build_bulk_upsert(
        table,
        merge_by_conflict_key(
            table,
            tuple(
                {
                    "user_id": "u1",
                    "date": T0.date().isoformat(),
                    "api_key": key,
                    "model": "A",
                    "custom_llm_provider": "anthropic",
                    "model_group": router,
                    "spend": spend,
                    "api_requests": 1,
                    "successful_requests": 1,
                    "autorouter_savings_spend": saved,
                }
                for _, spend, saved in requests
            ),
        ),
    )
    await db.execute_raw(statement, *values)

    (overall,) = await db.query_raw(
        'SELECT SUM(autorouter_savings_spend)::float8 AS saved FROM "LiteLLM_DailyUserSpend" WHERE date = $1 AND api_key = $2',
        T0.date().isoformat(),
        key,
    )
    (row,) = await _days(db, key, router=router)
    assert overall["saved"] == row["saved_spend"] == pytest.approx(3.75)
    assert (row["turns"], row["spend"], row["sessions"], row["session_turns"]) == (3, pytest.approx(0.85), 1, 2)
