"""Window-spend reads in ``SpendCounterReseed``.

The maintained ``LiteLLM_BudgetWindowSpend`` row replaces a per-request
``LiteLLM_SpendLogs`` range scan, so these pin *when* the aggregate is still
allowed to run: only when the row is missing or belongs to an older window.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed

WINDOW_START = datetime(2026, 8, 1, tzinfo=timezone.utc)
DB_SPEND: Final = 2777.16
COUNTER_KEY: Final = "spend:org:test-org"


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


class _OrgRow:
    spend: Final = DB_SPEND


class _FakeOrgTable:
    def __init__(self, read_started: asyncio.Event, read_release: asyncio.Event):
        self.read_started: Final = read_started
        self.read_release: Final = read_release

    async def find_unique(self, where: dict[str, str]) -> _OrgRow:
        self.read_started.set()
        await self.read_release.wait()
        return _OrgRow()


class _FakeDB:
    def __init__(self, table: _FakeOrgTable):
        self.litellm_organizationtable: Final = table


class _FakePrisma:
    def __init__(self, table: _FakeOrgTable):
        self.db: Final = _FakeDB(table)


def _fake_prisma(read_started: asyncio.Event, read_release: asyncio.Event) -> _FakePrisma:
    return _FakePrisma(_FakeOrgTable(read_started=read_started, read_release=read_release))


@pytest.mark.asyncio
async def test_coalesced_no_redis_does_not_double_when_seeded_during_db_read():
    """
    Regression test for the cold-seed race (LIT-5516): with no Redis, if a
    concurrent task (e.g. reservation reconcile) seeds the in-memory counter
    while coalesced() is awaiting the DB read, the blind increment doubled
    the counter to 2x db_spend and falsely tripped budget enforcement.
    """
    cache: Final = DualCache(default_in_memory_ttl=60)
    db_read_started: Final = asyncio.Event()
    db_read_release: Final = asyncio.Event()

    async def seed_during_db_read():
        await db_read_started.wait()
        cache.in_memory_cache.set_cache(key=COUNTER_KEY, value=DB_SPEND)
        db_read_release.set()

    result, _ = await asyncio.gather(
        SpendCounterReseed.coalesced(
            prisma_client=_fake_prisma(db_read_started, db_read_release),
            spend_counter_cache=cache,
            counter_key=COUNTER_KEY,
        ),
        seed_during_db_read(),
    )

    assert result == DB_SPEND
    final_value: Final = cache.in_memory_cache.get_cache(key=COUNTER_KEY)
    assert float(final_value) == DB_SPEND


@pytest.mark.asyncio
async def test_coalesced_no_redis_seeds_cold_counter():
    cache: Final = DualCache(default_in_memory_ttl=60)
    db_read_started: Final = asyncio.Event()
    db_read_release: Final = asyncio.Event()
    db_read_release.set()

    result: Final = await SpendCounterReseed.coalesced(
        prisma_client=_fake_prisma(db_read_started, db_read_release),
        spend_counter_cache=cache,
        counter_key=COUNTER_KEY,
    )

    assert result == DB_SPEND
    assert float(cache.in_memory_cache.get_cache(key=COUNTER_KEY)) == DB_SPEND
