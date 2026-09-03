"""Window-spend reads in ``SpendCounterReseed``.

The maintained ``LiteLLM_BudgetWindowSpend`` row replaces a per-request
``LiteLLM_SpendLogs`` range scan, so these pin *when* the aggregate is still
allowed to run: only when the row is missing or belongs to an older window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed

WINDOW_START = datetime(2026, 8, 1, tzinfo=timezone.utc)


class _FakeWindowSpendTable:
    def __init__(self, row: SimpleNamespace | None, error: Exception | None = None) -> None:
        self._row = row
        self._error = error
        self.where_clauses: list[dict] = []

    async def find_unique(self, where: dict):
        self.where_clauses.append(where)
        if self._error is not None:
            raise self._error
        return self._row


class _FakeSpendLogsTable:
    def __init__(self, total: float) -> None:
        self._total = total
        self.call_count = 0

    async def group_by(self, by: list[str], where: dict, sum: dict):
        self.call_count += 1
        return [{by[0]: where.get(by[0]), "_sum": {"spend": self._total}}]


class _FakePrismaClient:
    def __init__(
        self,
        row: SimpleNamespace | None = None,
        spend_logs_total: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.db = SimpleNamespace(
            litellm_budgetwindowspend=_FakeWindowSpendTable(row=row, error=error),
            litellm_spendlogs=_FakeSpendLogsTable(total=spend_logs_total),
        )


def _row(window_start: datetime, spend: float) -> SimpleNamespace:
    return SimpleNamespace(window_start=window_start, spend=spend)


@pytest.mark.asyncio
async def test_window_from_table_reads_row_by_primary_key():
    """The lookup must use the table's own entity_type values ("key"), not the
    "Key"/"Team" labels the counter keys and spend-log aggregates use."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 4.5
    assert prisma.db.litellm_budgetwindowspend.where_clauses == [
        {
            "entity_type_entity_id_window_duration": {
                "entity_type": "key",
                "entity_id": "tok-1",
                "window_duration": "30d",
            }
        }
    ]


@pytest.mark.asyncio
async def test_window_from_table_maps_team_entity_type():
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 9.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Team",
        entity_id="team-1",
        window_duration="1d",
        expected_window_start=WINDOW_START,
    )

    assert result == 9.0
    inner = prisma.db.litellm_budgetwindowspend.where_clauses[0]["entity_type_entity_id_window_duration"]
    assert inner["entity_type"] == "team"


@pytest.mark.asyncio
async def test_window_from_table_trusts_row_newer_than_expected_window():
    """Regression: a pod holding a stale ``reset_at`` computes an expected start
    behind a window another pod already rolled. Trusting only an exact match
    would make it re-add the previous window's spend to the current one."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START + timedelta(days=1), 2.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 2.0


@pytest.mark.asyncio
async def test_window_from_table_rejects_row_from_previous_window():
    prisma = _FakePrismaClient(row=_row(WINDOW_START - timedelta(seconds=1), 99.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_table_treats_naive_row_timestamp_as_utc():
    """The column is ``timestamp(3)``, so a driver that hands back a naive value
    must still compare against the tz-aware expected start."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START.replace(tzinfo=None), 3.0))

    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma, entity_type",
    [
        (_FakePrismaClient(row=None), "Key"),
        (_FakePrismaClient(row=_row(WINDOW_START, 1.0)), "User"),
        (_FakePrismaClient(error=RuntimeError("connection reset")), "Key"),
        (None, "Key"),
    ],
)
async def test_window_from_table_returns_none_without_a_usable_row(prisma, entity_type):
    result = await SpendCounterReseed.window_from_table(
        prisma_client=prisma,
        entity_type=entity_type,
        entity_id="tok-1",
        window_duration="30d",
        expected_window_start=WINDOW_START,
    )

    assert result is None


@pytest.mark.asyncio
async def test_window_from_db_prefers_the_row_over_the_spend_logs_aggregate():
    """The aggregate range-scans an unindexed table; a current row must keep it
    from running at all."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=100.0)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert prisma.db.litellm_spendlogs.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [None, _row(WINDOW_START - timedelta(seconds=1), 99.0)],
    ids=["missing_row", "previous_window_row"],
)
async def test_window_from_db_falls_back_to_spend_logs(row):
    prisma = _FakePrismaClient(row=row, spend_logs_total=7.25)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 7.25
    assert prisma.db.litellm_spendlogs.call_count == 1


@pytest.mark.asyncio
async def test_window_from_db_without_a_duration_skips_the_row_lookup():
    """Callers that cannot name the window (no PK) keep the pre-table behavior."""
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=7.25)

    result = await SpendCounterReseed.window_from_db(
        prisma_client=prisma,
        entity_type="Key",
        entity_id="tok-1",
        window_duration=None,
        window_start=WINDOW_START,
    )

    assert result == 7.25
    assert prisma.db.litellm_budgetwindowspend.where_clauses == []


@pytest.mark.asyncio
async def test_coalesced_window_seeds_a_cold_counter_from_the_row():
    prisma = _FakePrismaClient(row=_row(WINDOW_START, 4.5), spend_logs_total=100.0)
    cache = DualCache()
    counter_key = "spend:key:tok-1:window:30d"

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_cache=cache,
        counter_key=counter_key,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert cache.in_memory_cache.get_cache(key=counter_key) == 4.5
    assert prisma.db.litellm_spendlogs.call_count == 0
