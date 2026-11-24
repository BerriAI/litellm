"""Window-spend reads in ``SpendCounterReseed``.

The maintained ``LiteLLM_BudgetWindowSpend`` row replaces a per-request
``LiteLLM_SpendLogs`` range scan, so these pin *when* the aggregate is still
allowed to run: only when the row is missing or belongs to an older window.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import PROXY_DB_LOOKUP_MAX_CONCURRENCY
from litellm.proxy.db.db_lookup_gate import LoopBoundSemaphore, db_lookup_stall_tracker
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore, user_counter, user_window_counter

WINDOW_START = datetime(2026, 8, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_user_window_reseeds_from_maintained_balance() -> None:
    prisma: Final = _FakePrismaClient(row=SimpleNamespace(window_start=WINDOW_START, spend=4.5))
    result: Final = await SpendCounterReseed.window_from_db(prisma, "User", "u", "1h", WINDOW_START)
    assert result == 4.5
    assert prisma.db.litellm_budgetwindowspend.where_clauses == [
        {"entity_type_entity_id_window_duration": {"entity_type": "user", "entity_id": "u", "window_duration": "1h"}}
    ]
    assert prisma.db.litellm_spendlogs.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "counter_key, expected",
    [(user_counter("u:window:1h"), 7.0), (user_window_counter("u", "1h"), None)],
    ids=["user_id_that_looks_like_a_window", "user_window"],
)
async def test_only_a_users_own_counter_reseeds_from_the_user_row(counter_key: str, expected: float | None) -> None:
    users: Final = _FakeFindUniqueTable(row=SimpleNamespace(spend=7.0))
    prisma: Final = SimpleNamespace(db=SimpleNamespace(litellm_usertable=users))
    assert await SpendCounterReseed.from_db(prisma, counter_key) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "family,table_name,id_field",
    (
        ("customer_org", "litellm_customerorganizationtable", "customer_org_id"),
        ("customer_team", "litellm_customerteamtable", "customer_team_id"),
        ("customer_member", "litellm_customermembertable", "customer_member_id"),
    ),
)
@pytest.mark.parametrize(
    "period,reset_at,expected",
    (
        ("lifetime", None, 4.5),
        (WINDOW_START.isoformat(), WINDOW_START, 4.5),
        (WINDOW_START.isoformat(), WINDOW_START.replace(tzinfo=None), 4.5),
        (WINDOW_START.isoformat(), WINDOW_START + timedelta(days=1), None),
        (WINDOW_START.isoformat(), WINDOW_START - timedelta(days=1), None),
        ("lifetime", WINDOW_START, None),
        (WINDOW_START.isoformat(), None, None),
        ("not-a-period", None, None),
        ("2026-99-99T00:00:00+00:00", WINDOW_START, None),
    ),
)
async def test_customer_counter_reseed_reads_only_the_matching_budget_period(
    family, table_name, id_field, period, reset_at, expected
):
    table: Final = _FakeFindUniqueTable(
        SimpleNamespace(spend=4.5, budget=SimpleNamespace(budget_duration="1mo", budget_reset_at=reset_at))
    )
    prisma: Final = SimpleNamespace(db=SimpleNamespace(**{table_name: table}))
    actual: Final = await SpendCounterReseed.from_db(prisma, f"spend:{family}:id:with:colons:{period}")
    assert actual == expected
    if period != "not-a-period":
        assert table.where_clauses == [{id_field: "id:with:colons"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (None, ConnectionError("database is unavailable")))
async def test_missing_or_unavailable_customer_spend_does_not_seed_a_counter(error):
    table: Final = _FakeFindUniqueTable(None, error)
    prisma: Final = SimpleNamespace(db=SimpleNamespace(litellm_customermembertable=table))
    assert await SpendCounterReseed.from_db(prisma, "spend:customer_member:member:lifetime") is None


class _FakeFindUniqueTable:
    def __init__(self, row: SimpleNamespace | None, error: Exception | None = None) -> None:
        self._row = row
        self._error = error
        self.where_clauses: list[dict] = []

    async def find_unique(self, where: dict, include: dict | None = None):
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


class _InFlightCountingTable:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.001)
        self.in_flight -= 1
        return SimpleNamespace(token=where["token"], spend=1.0)


class _FakePrismaClient:
    def __init__(
        self,
        row: SimpleNamespace | None = None,
        spend_logs_total: float = 0.0,
        error: Exception | None = None,
        end_user_row: SimpleNamespace | None = None,
        end_user_error: Exception | None = None,
        project_row: SimpleNamespace | None = None,
    ) -> None:
        self.db = SimpleNamespace(
            litellm_budgetwindowspend=_FakeFindUniqueTable(row=row, error=error),
            litellm_spendlogs=_FakeSpendLogsTable(total=spend_logs_total),
            litellm_endusertable=_FakeFindUniqueTable(row=end_user_row, error=end_user_error),
            litellm_verificationtoken=_InFlightCountingTable(),
            litellm_projecttable=_FakeFindUniqueTable(row=project_row),
        )


def _row(window_start: datetime, spend: float) -> SimpleNamespace:
    return SimpleNamespace(window_start=window_start, spend=spend)


class _PausedSpendTable:
    def __init__(self, spend: float) -> None:
        self.spend: Final = spend
        self.read_started: Final = asyncio.Event()
        self.resume_read: Final = asyncio.Event()

    async def find_unique(self, where: Mapping[str, object]) -> SimpleNamespace:
        self.read_started.set()
        await self.resume_read.wait()
        return _row(WINDOW_START, self.spend)


async def _reseed_with_paused_table(
    table: _PausedSpendTable, cache: SpendCounterStore, counter_key: str, window: bool
) -> float | None:
    prisma: Final = SimpleNamespace(db=SimpleNamespace(litellm_usertable=table, litellm_budgetwindowspend=table))
    if window:
        return await SpendCounterReseed.coalesced_window(
            prisma_client=prisma,
            spend_counter_store=cache,
            counter_key=counter_key,
            entity_type="Team",
            entity_id="team-1",
            window_duration="1d",
            window_start=WINDOW_START,
        )
    return await SpendCounterReseed.coalesced(
        prisma_client=prisma,
        spend_counter_store=cache,
        counter_key=counter_key,
    )


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
        (_FakePrismaClient(row=_row(WINDOW_START, 1.0)), "Unsupported"),
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
    cache = SpendCounterStore(connect(None).spend_counters())
    counter_key = "spend:key:tok-1:window:30d"

    result = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma,
        spend_counter_store=cache,
        counter_key=counter_key,
        entity_type="Key",
        entity_id="tok-1",
        window_duration="30d",
        window_start=WINDOW_START,
    )

    assert result == 4.5
    assert cache.get_local(counter_key) == 4.5
    assert prisma.db.litellm_spendlogs.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [False, True], ids=["primary", "window"])
@pytest.mark.parametrize("concurrent_spend", [989.01459411, 995.0, 900.0])
async def test_cold_reseed_does_not_add_database_spend_to_concurrent_cache(
    window: bool,
    concurrent_spend: float,
) -> None:
    cache: Final = SpendCounterStore(connect(None).spend_counters())
    counter_key: Final = "spend:team:team-1:window:1d" if window else "spend:user:user-1"
    db_spend: Final = 989.01459411
    table: Final = _PausedSpendTable(db_spend)
    reseed_task: Final = asyncio.create_task(_reseed_with_paused_table(table, cache, counter_key, window))

    await asyncio.wait_for(table.read_started.wait(), timeout=5)
    cache.set_local(counter_key, concurrent_spend)
    table.resume_read.set()
    result: Final = await asyncio.wait_for(reseed_task, timeout=5)

    expected: Final = max(db_spend, concurrent_spend)
    assert cache.get_local(counter_key) == expected
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [False, True], ids=["primary", "window"])
@pytest.mark.parametrize("batch", [False, True], ids=["single_increment", "batch_increment"])
@pytest.mark.parametrize("increment", [5.0, -5.0], ids=["charge", "refund"])
async def test_cold_reseed_preserves_concurrent_local_increment(
    monkeypatch: pytest.MonkeyPatch, window: bool, batch: bool, increment: float
) -> None:
    from litellm.proxy import proxy_server

    cache: Final = SpendCounterStore(connect(None).spend_counters())
    counter_key: Final = (
        f"spend:team:concurrent-{batch}-{increment}:window:1d"
        if window
        else f"spend:user:concurrent-{batch}-{increment}"
    )
    table: Final = _PausedSpendTable(100.0)
    monkeypatch.setattr(proxy_server, "spend_counter_store", cache)
    reseed_task: Final = asyncio.create_task(_reseed_with_paused_table(table, cache, counter_key, window))
    await asyncio.wait_for(table.read_started.wait(), timeout=5)

    increment_task: Final = asyncio.create_task(
        proxy_server._apply_spend_counter_increments(
            pending=(proxy_server.PendingSpendIncrement(counter_key=counter_key, increment=increment),)
        )
        if batch
        else proxy_server._increment_spend_counter_cache(counter_key=counter_key, increment=increment)
    )
    await asyncio.sleep(0)
    table.resume_read.set()
    await asyncio.wait_for(asyncio.gather(reseed_task, increment_task), timeout=5)

    assert cache.get_local(counter_key) == 100.0 + increment


@pytest.mark.asyncio
async def test_end_user_from_db_reads_the_end_user_row_by_user_id():
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=0.0))

    result: Final = await SpendCounterReseed.end_user_from_db(
        prisma_client=prisma, counter_key="spend:end_user:customer-42"
    )

    assert result == 0.0
    assert prisma.db.litellm_endusertable.where_clauses == [{"user_id": "customer-42"}]


@pytest.mark.asyncio
async def test_end_user_from_db_returns_the_recorded_spend():
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=12.5))

    assert (
        await SpendCounterReseed.end_user_from_db(prisma_client=prisma, counter_key="spend:end_user:customer-42")
        == 12.5
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("counter_key", ["spend:key:hashed", "spend:team:t1", "spend:tag:t1"])
async def test_end_user_from_db_ignores_other_counter_kinds_without_touching_the_db(counter_key):
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="x", spend=5.0))

    assert await SpendCounterReseed.end_user_from_db(prisma_client=prisma, counter_key=counter_key) is None
    assert prisma.db.litellm_endusertable.where_clauses == []


@pytest.mark.asyncio
async def test_end_user_from_db_returns_none_without_a_row_a_client_or_on_db_error():
    assert (
        await SpendCounterReseed.end_user_from_db(prisma_client=None, counter_key="spend:end_user:customer-42") is None
    )
    assert (
        await SpendCounterReseed.end_user_from_db(
            prisma_client=_FakePrismaClient(end_user_row=None), counter_key="spend:end_user:customer-42"
        )
        is None
    )
    assert (
        await SpendCounterReseed.end_user_from_db(
            prisma_client=_FakePrismaClient(end_user_error=RuntimeError("db down")),
            counter_key="spend:end_user:customer-42",
        )
        is None
    )


@pytest.mark.asyncio
async def test_from_db_bounds_in_flight_prisma_requests_across_counter_keys():
    """Per-counter singleflight only collapses duplicates of one key. A cold-cache burst
    over many distinct keys must still not flood the prisma engine HTTP pool (LIT-6435)."""
    prisma: Final = _FakePrismaClient()
    burst: Final = PROXY_DB_LOOKUP_MAX_CONCURRENCY * 5

    results: Final = await asyncio.gather(
        *(SpendCounterReseed.from_db(prisma_client=prisma, counter_key=f"spend:key:hashed-{i}") for i in range(burst))
    )

    assert results == [1.0] * burst
    assert prisma.db.litellm_verificationtoken.max_in_flight == PROXY_DB_LOOKUP_MAX_CONCURRENCY


@pytest.mark.asyncio
async def test_from_db_reseeds_project_counter_from_the_project_row():
    prisma: Final = _FakePrismaClient(project_row=SimpleNamespace(project_id="proj-1", spend=7.25))

    assert await SpendCounterReseed.from_db(prisma_client=prisma, counter_key="spend:project:proj-1") == 7.25
    assert prisma.db.litellm_projecttable.where_clauses == [{"project_id": "proj-1"}]


@pytest.mark.asyncio
async def test_from_db_returns_none_for_a_missing_project_row():
    prisma: Final = _FakePrismaClient(project_row=None)

    assert await SpendCounterReseed.from_db(prisma_client=prisma, counter_key="spend:project:proj-1") is None


@pytest.mark.asyncio
async def test_from_db_deadline_covers_the_wait_for_a_gate_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saturated gate must fail the lookup at the deadline instead of parking
    the request on a gate slot outside the bounded window."""
    gate: Final = LoopBoundSemaphore(1)
    monkeypatch.setattr("litellm.proxy.db.spend_counter_reseed.db_lookup_gate", gate)
    monkeypatch.setattr("litellm.proxy.db.db_lookup_gate.PROXY_DB_LOOKUP_DEADLINE_SECONDS", 0.05)
    find_unique: Final = AsyncMock()
    prisma: Final = SimpleNamespace(
        db=SimpleNamespace(litellm_verificationtoken=SimpleNamespace(find_unique=find_unique))
    )
    db_lookup_stall_tracker.clear()
    try:
        async with gate.current():
            result: Final = await asyncio.wait_for(
                SpendCounterReseed.from_db(prisma_client=prisma, counter_key="spend:key:abc"),
                timeout=1.0,
            )
        assert result is None
        assert db_lookup_stall_tracker.stalled_within(60.0)
        find_unique.assert_not_called()
    finally:
        db_lookup_stall_tracker.clear()


@pytest.mark.asyncio
async def test_from_db_still_never_reads_the_end_user_row():
    """A cold end-user counter keeps seeding from the cached end-user object the auth
    path already loaded; the row is read only as the budget floor."""
    prisma: Final = _FakePrismaClient(end_user_row=SimpleNamespace(user_id="customer-42", spend=5.0))

    assert await SpendCounterReseed.from_db(prisma_client=prisma, counter_key="spend:end_user:customer-42") is None
    assert prisma.db.litellm_endusertable.where_clauses == []
