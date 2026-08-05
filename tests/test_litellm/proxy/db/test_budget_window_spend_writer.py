import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.proxy.db.budget_window_spend_writer import (
    commit_window_spend_updates,
    roll_window_spend_row,
    spend_logs_total_excluding,
)
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    build_window_spend_transaction,
)

WINDOW_A = datetime(2026, 8, 1, tzinfo=timezone.utc)
WINDOW_B = datetime(2026, 8, 31, tzinfo=timezone.utc)

ENTITY_TYPE, ENTITY_ID, WINDOW_DURATION, WINDOW_START, INSERT_SPEND, INCREMENT, NOW = range(7)


class _FakeBatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute_raw(self, query: str, *args: Any) -> None:
        self.calls.append((query, args))


class _FakeDB:
    """Stands in for prisma_client.db; records every statement it is handed."""

    def __init__(self, existing_rows: list[dict[str, str]] | None = None) -> None:
        self.existing_rows = existing_rows or []
        self.query_raw_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_raw_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.batcher = _FakeBatcher()
        self.committed = False

    async def query_raw(self, query: str, *args: Any) -> list[dict[str, str]]:
        self.query_raw_calls.append((query, args))
        return self.existing_rows

    async def execute_raw(self, query: str, *args: Any) -> int:
        self.execute_raw_calls.append((query, args))
        return 1

    @asynccontextmanager
    async def _tx(self):
        yield self

    def tx(self, timeout: Any = None):
        return self._tx()

    @asynccontextmanager
    async def _batch(self):
        yield self.batcher
        self.committed = True

    def batch_(self):
        return self._batch()


class _FakePrismaClient:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db


class _RecordingAggregate:
    """Stands in for the LiteLLM_SpendLogs seed aggregate."""

    def __init__(self, value: float = 5.0) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        prisma_client: Any,
        entity_type: str,
        entity_id: str,
        window_start: datetime,
        exclude_request_ids: Any,
    ) -> float | None:
        self.calls.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "window_start": window_start,
                "exclude_request_ids": tuple(exclude_request_ids),
            }
        )
        return self.value


class _SpendLogsFake:
    """Sums the LiteLLM_SpendLogs rows it holds, honouring the request-id
    exclusion exactly as the real aggregate's NOT (request_id = ANY(...)) does."""

    def __init__(self, rows: tuple[tuple[str, float], ...]) -> None:
        self.rows = rows

    async def __call__(
        self,
        prisma_client: Any,
        entity_type: str,
        entity_id: str,
        window_start: datetime,
        exclude_request_ids: Any,
    ) -> float | None:
        excluded = frozenset(exclude_request_ids)
        return math.fsum(spend for request_id, spend in self.rows if request_id not in excluded)


def _existing(entity_type: str, entity_id: str, window_duration: str) -> dict[str, str]:
    return {"entity_type": entity_type, "entity_id": entity_id, "window_duration": window_duration}


@pytest.mark.asyncio
async def test_no_transactions_touches_no_database():
    db = _FakeDB()

    await commit_window_spend_updates(prisma_client=_FakePrismaClient(db), transactions=())

    assert db.query_raw_calls == []
    assert db.batcher.calls == []


@pytest.mark.asyncio
async def test_missing_row_is_seeded_from_spend_logs_once():
    """A row created mid-window would undercount everything spent before it
    existed, so a brand new primary key inserts the SpendLogs total plus this
    increment."""
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=5.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),),
        spend_logs_aggregate=aggregate,
    )

    assert len(aggregate.calls) == 1
    assert aggregate.calls[0]["entity_type"] == "key"
    assert aggregate.calls[0]["entity_id"] == "k1"
    assert aggregate.calls[0]["window_start"] == WINDOW_A

    (_, params), = db.batcher.calls
    assert params[ENTITY_TYPE] == "key"
    assert params[ENTITY_ID] == "k1"
    assert params[WINDOW_DURATION] == "30d"
    assert params[WINDOW_START] == datetime(2026, 8, 1)
    assert params[INSERT_SPEND] == pytest.approx(6.0)
    assert params[INCREMENT] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_existing_row_is_never_reseeded():
    """The seed is a full LiteLLM_SpendLogs scan; running it for a row that is
    already maintained would both cost a scan and double count."""
    db = _FakeDB(existing_rows=[_existing("key", "k1", "30d")])
    aggregate = _RecordingAggregate(value=5.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),),
        spend_logs_aggregate=aggregate,
    )

    assert aggregate.calls == []
    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(1.0)
    assert params[INCREMENT] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_seed_runs_only_for_the_primary_keys_that_are_missing():
    db = _FakeDB(existing_rows=[_existing("key", "k1", "30d")])
    aggregate = _RecordingAggregate(value=5.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),
            build_window_spend_transaction("team", "t1", "30d", WINDOW_A, 2.0),
        ),
        spend_logs_aggregate=aggregate,
    )

    assert [call["entity_id"] for call in aggregate.calls] == ["t1"]
    assert [call["entity_type"] for call in aggregate.calls] == ["team"]
    by_entity = {params[ENTITY_ID]: params for _, params in db.batcher.calls}
    assert by_entity["k1"][INSERT_SPEND] == pytest.approx(1.0)
    assert by_entity["t1"][INSERT_SPEND] == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_insert_spend_and_increment_differ_only_when_a_row_is_seeded():
    """The conflict arm adds the increment alone so two pods that both seed the
    same new window cannot add the SpendLogs base twice."""
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=9.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 0.25),),
        spend_logs_aggregate=aggregate,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(9.25)
    assert params[INCREMENT] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_upsert_sql_adds_for_a_current_window_and_replaces_for_a_newer_one():
    """The CASE is the whole contract: an increment at or behind the stored
    window_start accumulates, a newer one restarts the window."""
    db = _FakeDB(existing_rows=[_existing("key", "k1", "30d")])

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),),
    )

    (query, _), = db.batcher.calls
    normalized = " ".join(query.split())
    assert (
        'spend = CASE WHEN "LiteLLM_BudgetWindowSpend".window_start >= EXCLUDED.window_start '
        'THEN "LiteLLM_BudgetWindowSpend".spend + $6 ELSE EXCLUDED.spend END' in normalized
    )
    assert 'window_start = GREATEST("LiteLLM_BudgetWindowSpend".window_start, EXCLUDED.window_start)' in normalized
    assert "ON CONFLICT (entity_type, entity_id, window_duration) DO UPDATE SET" in normalized


@pytest.mark.asyncio
async def test_upsert_never_interpolates_values_into_the_sql():
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=0.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "'; DROP TABLE x; --", "30d", WINDOW_A, 1.0),),
        spend_logs_aggregate=aggregate,
    )

    (query, params), = db.batcher.calls
    assert "DROP TABLE" not in query
    assert params[ENTITY_ID] == "'; DROP TABLE x; --"


@pytest.mark.asyncio
async def test_upserts_are_ordered_by_primary_key_then_window_start():
    """Cross-pod lock ordering, plus an older window must be applied before the
    roll that supersedes it or the roll would be undone."""
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=0.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            build_window_spend_transaction("team", "t1", "30d", WINDOW_A, 1.0),
            build_window_spend_transaction("key", "k2", "30d", WINDOW_B, 1.0),
            build_window_spend_transaction("key", "k2", "30d", WINDOW_A, 1.0),
            build_window_spend_transaction("key", "k1", "7d", WINDOW_A, 1.0),
        ),
        spend_logs_aggregate=aggregate,
    )

    ordered = [(params[ENTITY_TYPE], params[ENTITY_ID], params[WINDOW_DURATION], params[WINDOW_START]) for _, params in db.batcher.calls]
    assert ordered == [
        ("key", "k1", "7d", datetime(2026, 8, 1)),
        ("key", "k2", "30d", datetime(2026, 8, 1)),
        ("key", "k2", "30d", datetime(2026, 8, 31)),
        ("team", "t1", "30d", datetime(2026, 8, 1)),
    ]


@pytest.mark.asyncio
async def test_existing_row_lookup_sends_every_primary_key_as_array_params():
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=0.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),
            build_window_spend_transaction("team", "t1", "7d", WINDOW_A, 1.0),
        ),
        spend_logs_aggregate=aggregate,
    )

    (query, params), = db.query_raw_calls
    assert "unnest($1::text[], $2::text[], $3::text[])" in query
    assert params == (("key", "team"), ("k1", "t1"), ("30d", "7d"))


@pytest.mark.asyncio
async def test_all_upserts_are_committed_in_one_transaction():
    db = _FakeDB(existing_rows=[_existing("key", "k1", "30d"), _existing("key", "k2", "30d")])

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),
            build_window_spend_transaction("key", "k2", "30d", WINDOW_A, 2.0),
        ),
    )

    assert len(db.batcher.calls) == 2
    assert db.committed is True


@pytest.mark.asyncio
async def test_unknown_entity_type_contributes_no_seed():
    """Only key and team windows have a LiteLLM_SpendLogs column to aggregate;
    anything else starts from its increment alone."""
    db = _FakeDB(existing_rows=[])

    async def no_such_column(prisma_client, entity_type, entity_id, window_start, exclude_request_ids):
        return None

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("user", "u1", "30d", WINDOW_A, 1.0),),
        spend_logs_aggregate=no_such_column,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_unavailable_spend_logs_aggregate_seeds_zero_rather_than_failing():
    db = _FakeDB(existing_rows=[])

    async def unavailable(prisma_client, entity_type, entity_id, window_start, exclude_request_ids):
        return None

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(build_window_spend_transaction("key", "k1", "30d", WINDOW_A, 1.0),),
        spend_logs_aggregate=unavailable,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_roll_window_spend_row_is_conditional_on_the_stored_window_being_older():
    """Unconditional zeroing would wipe increments a pod already applied under
    the new window."""
    db = _FakeDB()

    await roll_window_spend_row(
        prisma_client=_FakePrismaClient(db),
        entity_type="team",
        entity_id="t1",
        window_duration="30d",
        new_window_start=WINDOW_B,
    )

    (query, params), = db.execute_raw_calls
    normalized = " ".join(query.split())
    assert "SET window_start = ($4::timestamptz AT TIME ZONE 'UTC'), spend = 0" in normalized
    assert "WHERE entity_type = $1 AND entity_id = $2 AND window_duration = $3" in normalized
    assert "AND window_start < ($4::timestamptz AT TIME ZONE 'UTC')" in normalized
    assert params[:4] == ("team", "t1", "30d", datetime(2026, 8, 31))


@pytest.mark.asyncio
async def test_seed_receives_the_batch_request_ids_to_exclude():
    db = _FakeDB(existing_rows=[])
    aggregate = _RecordingAggregate(value=0.0)

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            {
                "entity_type": "key",
                "entity_id": "k1",
                "window_duration": "30d",
                "window_start": "2026-08-01T00:00:00.000000",
                "spend": 3.0,
                "request_ids": ("req-1", "req-2", "req-3"),
            },
        ),
        spend_logs_aggregate=aggregate,
    )

    assert aggregate.calls[0]["exclude_request_ids"] == ("req-1", "req-2", "req-3")


@pytest.mark.asyncio
async def test_new_row_is_not_double_counted_when_the_batch_logs_already_flushed():
    """The spend log writer drains on a ~2s poll while window increments flush
    on the ~10s batch tick, so a new row is normally seeded from a table that
    already holds this batch's rows. Counting them in both places is what made
    a fresh row land at exactly twice the true spend."""
    db = _FakeDB(existing_rows=[])
    already_flushed = _SpendLogsFake(
        rows=(("req-1", 0.000047), ("req-2", 0.000047), ("req-3", 0.000047)),
    )

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            {
                "entity_type": "key",
                "entity_id": "k1",
                "window_duration": "30d",
                "window_start": "2026-08-01T00:00:00.000000",
                "spend": 0.000141,
                "request_ids": ("req-1", "req-2", "req-3"),
            },
        ),
        spend_logs_aggregate=already_flushed,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(0.000141)


@pytest.mark.asyncio
async def test_new_row_still_covers_spend_that_predates_the_batch():
    """The exclusion must not throw away the pre-existing spend the seed is for."""
    db = _FakeDB(existing_rows=[])
    spend_logs = _SpendLogsFake(rows=(("older", 0.5), ("req-1", 0.000047)))

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            {
                "entity_type": "key",
                "entity_id": "k1",
                "window_duration": "30d",
                "window_start": "2026-08-01T00:00:00.000000",
                "spend": 0.000047,
                "request_ids": ("req-1",),
            },
        ),
        spend_logs_aggregate=spend_logs,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(0.500047)


@pytest.mark.asyncio
async def test_new_row_is_correct_when_the_batch_logs_have_not_flushed_yet():
    """The other side of the race: rows absent from the aggregate are still
    counted exactly once, by their increment."""
    db = _FakeDB(existing_rows=[])
    nothing_flushed = _SpendLogsFake(rows=())

    await commit_window_spend_updates(
        prisma_client=_FakePrismaClient(db),
        transactions=(
            {
                "entity_type": "key",
                "entity_id": "k1",
                "window_duration": "30d",
                "window_start": "2026-08-01T00:00:00.000000",
                "spend": 0.000141,
                "request_ids": ("req-1", "req-2", "req-3"),
            },
        ),
        spend_logs_aggregate=nothing_flushed,
    )

    (_, params), = db.batcher.calls
    assert params[INSERT_SPEND] == pytest.approx(0.000141)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity_type, expected_column",
    [("key", "api_key = $1"), ("team", "team_id = $1")],
)
async def test_seed_aggregate_sql_excludes_the_request_ids_by_parameter(entity_type, expected_column):
    db = _FakeDB(existing_rows=[{"total": 1.25}])

    total = await spend_logs_total_excluding(
        prisma_client=_FakePrismaClient(db),
        entity_type=entity_type,
        entity_id="e1",
        window_start=WINDOW_A,
        exclude_request_ids=("req-1", "req-2"),
    )

    assert total == pytest.approx(1.25)
    (query, params), = db.query_raw_calls
    normalized = " ".join(query.split())
    assert expected_column in normalized
    assert "NOT (request_id = ANY($3::text[]))" in normalized
    assert 'FROM "LiteLLM_SpendLogs"' in normalized
    assert params == ("e1", WINDOW_A, ("req-1", "req-2"))
    # The ids are bound, never spliced into the statement.
    assert "req-1" not in query


@pytest.mark.asyncio
async def test_seed_aggregate_returns_none_for_an_entity_type_with_no_spend_logs_column():
    db = _FakeDB(existing_rows=[])

    total = await spend_logs_total_excluding(
        prisma_client=_FakePrismaClient(db),
        entity_type="user",
        entity_id="u1",
        window_start=WINDOW_A,
        exclude_request_ids=(),
    )

    assert total is None
    assert db.query_raw_calls == []


@pytest.mark.asyncio
async def test_seed_aggregate_treats_an_entity_with_no_rows_as_zero():
    db = _FakeDB(existing_rows=[])

    total = await spend_logs_total_excluding(
        prisma_client=_FakePrismaClient(db),
        entity_type="key",
        entity_id="k-unknown",
        window_start=WINDOW_A,
        exclude_request_ids=(),
    )

    assert total == 0.0
