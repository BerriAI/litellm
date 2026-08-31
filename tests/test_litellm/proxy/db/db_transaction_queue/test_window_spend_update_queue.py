import json
from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendUpdateQueue,
    build_window_spend_transaction,
    to_naive_utc,
)

WINDOW_A = datetime(2026, 8, 1, tzinfo=timezone.utc)
WINDOW_B = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _txn(
    entity_id: str,
    window_start: datetime,
    spend: float,
    duration: str = "30d",
    entity_type: str = "key",
    request_id: str | None = None,
    started_at: datetime | None = None,
):
    return build_window_spend_transaction(
        entity_type=entity_type,
        entity_id=entity_id,
        window_duration=duration,
        window_start=window_start,
        spend=spend,
        request_id=request_id,
        started_at=started_at,
    )


def test_build_window_spend_transaction_stores_naive_utc_iso():
    """window_start rides the Redis buffer as a string and lands in a naive-UTC
    TIMESTAMP(3) column, so a non-UTC input must be converted, not truncated."""
    non_utc = datetime(2026, 8, 1, 20, 0, tzinfo=timezone(timedelta(hours=-4)))

    assert _txn("k1", non_utc, 1.0, request_id="req-1") == {
        "entity_type": "key",
        "entity_id": "k1",
        "window_duration": "30d",
        "window_start": "2026-08-02T00:00:00.000000",
        "spend": 1.0,
        "request_ids": ("req-1",),
        "started_at": None,
    }


def test_build_window_spend_transaction_stores_started_at_as_naive_utc_iso():
    """started_at is compared against LiteLLM_SpendLogs.startTime, which the
    spend log writer stores after converting the request start to UTC."""
    non_utc = datetime(2026, 8, 10, 8, 30, 15, 123456, tzinfo=timezone(timedelta(hours=-4)))

    assert _txn("k1", WINDOW_A, 1.0, started_at=non_utc)["started_at"] == "2026-08-10T12:30:15.123456"


@pytest.mark.asyncio
async def test_aggregation_keeps_the_earliest_started_at_of_the_batch():
    """The seed bounds its request-id exclusion at the batch's earliest start,
    so a later start must never win the merge."""
    queue = WindowSpendUpdateQueue()
    earliest = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-2", started_at=earliest + timedelta(seconds=5)))
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-1", started_at=earliest))
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-3"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 1
    assert aggregated[0]["started_at"] == "2026-08-10T12:00:00.000000"
    assert aggregated[0]["request_ids"] == ("req-1", "req-2", "req-3")


def test_to_naive_utc_leaves_naive_values_alone():
    naive = datetime(2026, 8, 1, 12, 0)
    assert to_naive_utc(naive) == naive


@pytest.mark.asyncio
async def test_aggregation_sums_increments_within_one_window():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.5))
    await queue.add_update(_txn("k1", WINDOW_A, 2.25))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 1
    assert aggregated[0]["spend"] == pytest.approx(3.75)


@pytest.mark.asyncio
async def test_aggregation_keeps_different_windows_of_same_entity_separate():
    """Merging across windows would fold spend from a window that already
    rolled into the new window's total, over-counting the new window."""
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0))
    await queue.add_update(_txn("k1", WINDOW_B, 2.0))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 2
    assert {payload["window_start"]: payload["spend"] for payload in aggregated} == {
        "2026-08-01T00:00:00.000000": 1.0,
        "2026-08-31T00:00:00.000000": 2.0,
    }


@pytest.mark.asyncio
async def test_aggregation_keeps_durations_entities_and_types_separate():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, duration="30d"))
    await queue.add_update(_txn("k1", WINDOW_A, 2.0, duration="7d"))
    await queue.add_update(_txn("k2", WINDOW_A, 4.0, duration="30d"))
    await queue.add_update(_txn("k1", WINDOW_A, 8.0, duration="30d", entity_type="team"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 4
    assert sorted(payload["spend"] for payload in aggregated) == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_aggregation_orders_by_primary_key_then_window_start():
    """The flush relies on this order: primary key first for cross-pod lock
    ordering, then window_start so an older window is applied before the roll
    that supersedes it."""
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("t1", WINDOW_A, 1.0, entity_type="team"))
    await queue.add_update(_txn("k2", WINDOW_B, 1.0))
    await queue.add_update(_txn("k2", WINDOW_A, 1.0))
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, duration="7d"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert [
        (payload["entity_type"], payload["entity_id"], payload["window_duration"], payload["window_start"])
        for payload in aggregated
    ] == [
        ("key", "k1", "7d", "2026-08-01T00:00:00.000000"),
        ("key", "k2", "30d", "2026-08-01T00:00:00.000000"),
        ("key", "k2", "30d", "2026-08-31T00:00:00.000000"),
        ("team", "t1", "30d", "2026-08-01T00:00:00.000000"),
    ]


@pytest.mark.asyncio
async def test_aggregation_does_not_collide_on_entity_ids_containing_a_separator():
    """entity_id is free-form (team ids are user supplied), so grouping must not
    depend on a flattened string key."""
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("a:30d:2026-08-01T00:00:00.000000:b", WINDOW_A, 1.0))
    await queue.add_update(_txn("b", WINDOW_A, 2.0))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 2


@pytest.mark.asyncio
async def test_flush_empties_the_queue():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0))

    assert await queue.flush_and_get_aggregated_window_spend_transactions() != ()
    assert await queue.flush_and_get_aggregated_window_spend_transactions() == ()


@pytest.mark.asyncio
async def test_aggregate_queue_updates_collapses_in_place():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0))
    await queue.add_update(_txn("k1", WINDOW_A, 2.0))
    await queue.add_update(_txn("k1", WINDOW_B, 4.0))

    await queue.aggregate_queue_updates()

    assert queue.update_queue.qsize() == 1
    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()
    assert sorted(payload["spend"] for payload in aggregated) == [3.0, 4.0]


@pytest.mark.asyncio
async def test_aggregation_does_not_mutate_the_queued_payloads():
    """The same payload can be re-aggregated after a failed Redis push, so
    aggregation must not accumulate into the caller's object."""
    queue = WindowSpendUpdateQueue()
    update = _txn("k1", WINDOW_A, 1.0)
    await queue.add_update(update)
    await queue.add_update(_txn("k1", WINDOW_A, 2.0))

    await queue.flush_and_get_aggregated_window_spend_transactions()

    assert update["spend"] == 1.0


def test_aggregation_survives_the_redis_json_round_trip():
    """The Redis buffer stores transactions as JSON, so the aggregated shape
    must reload into an equivalent aggregation."""
    aggregated = WindowSpendUpdateQueue.get_aggregated_window_spend_transactions(
        [(_txn("k1", WINDOW_A, 1.0),), (_txn("k1", WINDOW_B, 2.0),)]
    )

    reloaded = WindowSpendUpdateQueue.get_aggregated_window_spend_transactions([json.loads(json.dumps(aggregated))])

    assert reloaded == aggregated


@pytest.mark.asyncio
async def test_aggregation_unions_the_request_ids_of_merged_increments():
    """The seed excludes exactly the requests its batch already covers, so every
    merged increment's id has to survive aggregation."""
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-1"))
    await queue.add_update(_txn("k1", WINDOW_A, 2.0, request_id="req-2"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert len(aggregated) == 1
    assert aggregated[0]["request_ids"] == ("req-1", "req-2")


@pytest.mark.asyncio
async def test_request_ids_stay_with_their_own_window():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-a"))
    await queue.add_update(_txn("k1", WINDOW_B, 2.0, request_id="req-b"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert {payload["window_start"]: payload["request_ids"] for payload in aggregated} == {
        "2026-08-01T00:00:00.000000": ("req-a",),
        "2026-08-31T00:00:00.000000": ("req-b",),
    }


@pytest.mark.asyncio
async def test_request_ids_are_deduplicated_and_ordered():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-b"))
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-a"))
    await queue.add_update(_txn("k1", WINDOW_A, 1.0, request_id="req-a"))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert aggregated[0]["request_ids"] == ("req-a", "req-b")


@pytest.mark.asyncio
async def test_increment_without_a_request_id_carries_no_exclusion():
    queue = WindowSpendUpdateQueue()
    await queue.add_update(_txn("k1", WINDOW_A, 1.0))

    aggregated = await queue.flush_and_get_aggregated_window_spend_transactions()

    assert aggregated[0]["request_ids"] == ()


def test_request_ids_survive_the_redis_json_round_trip():
    aggregated = WindowSpendUpdateQueue.get_aggregated_window_spend_transactions(
        [(_txn("k1", WINDOW_A, 1.0, request_id="req-1"),)]
    )

    reloaded = WindowSpendUpdateQueue.get_aggregated_window_spend_transactions([json.loads(json.dumps(aggregated))])

    assert reloaded[0]["request_ids"] == ("req-1",)
    assert reloaded[0]["spend"] == 1.0
