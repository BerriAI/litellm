from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import WindowSpendTransaction
from litellm.proxy.hooks.model_budget_shared_windows import DatabaseSharedWindows, SharedWindow

SPEND_KEY: Final = "virtual_key_spend:key-hash:claude-opus*:1d"
WINDOW_START: Final = datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc)


class _Table:
    def __init__(self, row: object | None = None, error: Exception | None = None) -> None:
        self.row = row
        self.error = error
        self.reads = 0

    async def find_unique(self, where: object) -> object | None:
        self.reads = self.reads + 1
        if self.error is not None:
            raise self.error
        return self.row


class _Database:
    def __init__(self, table: _Table) -> None:
        self.litellm_budgetwindowspend = table
        self.opened_with: tuple[object, ...] = ()

    async def execute_raw(self, query: str, *params: object) -> int:
        self.opened_with = params
        self.litellm_budgetwindowspend.row = SimpleNamespace(window_start=params[3], spend=0.0)
        return 1


class _OutracedDatabase:
    def __init__(self, table: _Table, winner: object) -> None:
        self.litellm_budgetwindowspend = table
        self._winner = winner

    async def execute_raw(self, query: str, *params: object) -> int:
        self.litellm_budgetwindowspend.row = self._winner
        return 0


class _Sink:
    def __init__(self) -> None:
        self.transactions: tuple[WindowSpendTransaction, ...] = ()

    async def __call__(self, transaction: WindowSpendTransaction) -> None:
        self.transactions = (*self.transactions, transaction)


class _LaggingReplica:
    def __init__(self, writer: _Database) -> None:
        self.litellm_budgetwindowspend = _Table()
        self._writer = writer

    async def execute_raw(self, query: str, *params: object) -> int:
        return await self._writer.execute_raw(query, *params)


def _database_windows(table: _Table | None, sink: _Sink, lagging_replica: bool = False) -> DatabaseSharedWindows:
    writer: Final = None if table is None else _Database(table)
    prisma: Final = (
        None
        if writer is None
        else SimpleNamespace(db=_LaggingReplica(writer) if lagging_replica else writer, writer_db=writer)
    )
    return DatabaseSharedWindows(
        prisma_client_source=lambda: prisma,  # pyright: ignore[reportArgumentType]  # duck-typed prisma double
        sink=sink,
    )


def test_a_window_is_open_until_its_period_has_passed():
    window: Final = SharedWindow(started_at=WINDOW_START.timestamp(), spend=1.0)
    day: Final = 24 * 60 * 60

    assert window.is_open(budget_duration="1d", now=WINDOW_START.timestamp() + day - 1)
    assert not window.is_open(budget_duration="1d", now=WINDOW_START.timestamp() + day)


@pytest.mark.asyncio
async def test_read_returns_the_stored_window_and_rereads_only_after_the_cache_expires():
    table: Final = _Table(row=SimpleNamespace(window_start=WINDOW_START.replace(tzinfo=None), spend=2.5))
    windows: Final = _database_windows(table, _Sink())

    first: Final = await windows.read(spend_key=SPEND_KEY, budget_duration="1d")
    second: Final = await windows.read(spend_key=SPEND_KEY, budget_duration="1d")

    assert first == SharedWindow(started_at=WINDOW_START.timestamp(), spend=2.5)
    assert second == first
    assert table.reads == 1


@pytest.mark.asyncio
async def test_an_unreadable_table_reads_as_no_shared_window():
    table: Final = _Table(error=RuntimeError("database is down"))
    windows: Final = _database_windows(table, _Sink())

    assert await windows.read(spend_key=SPEND_KEY, budget_duration="1d") is None


@pytest.mark.asyncio
async def test_add_queues_the_charge_for_the_window_it_belongs_to():
    sink: Final = _Sink()
    windows: Final = _database_windows(_Table(), sink)

    await windows.add(spend_key=SPEND_KEY, budget_duration="1d", started_at=WINDOW_START.timestamp(), spend=0.25)

    assert len(sink.transactions) == 1
    queued: Final = sink.transactions[0]
    assert (queued["entity_id"], queued["window_duration"], queued["spend"]) == (SPEND_KEY, "1d", 0.25)
    assert datetime.fromisoformat(queued["window_start"]).replace(tzinfo=timezone.utc) == WINDOW_START


@pytest.mark.asyncio
async def test_nothing_is_queued_without_a_database():
    sink: Final = _Sink()
    windows: Final = _database_windows(None, sink)

    await windows.add(spend_key=SPEND_KEY, budget_duration="1d", started_at=WINDOW_START.timestamp(), spend=0.25)

    assert sink.transactions == ()
    assert await windows.read(spend_key=SPEND_KEY, budget_duration="1d") is None


@pytest.mark.asyncio
async def test_open_joins_the_window_another_replica_already_started():
    table: Final = _Table(row=SimpleNamespace(window_start=WINDOW_START.replace(tzinfo=None), spend=2.5))
    windows: Final = _database_windows(table, _Sink())

    started_at: Final = await windows.open(
        spend_key=SPEND_KEY, budget_duration="1d", now=WINDOW_START.timestamp() + 3600
    )

    assert started_at == WINDOW_START.timestamp()
    assert table.row.spend == 2.5


@pytest.mark.asyncio
async def test_open_reads_the_window_from_the_writer_while_the_read_replica_lags():
    table: Final = _Table(row=SimpleNamespace(window_start=WINDOW_START.replace(tzinfo=None), spend=2.5))
    windows: Final = _database_windows(table, _Sink(), lagging_replica=True)

    started_at: Final = await windows.open(
        spend_key=SPEND_KEY, budget_duration="1d", now=WINDOW_START.timestamp() + 3600
    )

    assert started_at == WINDOW_START.timestamp()


@pytest.mark.asyncio
async def test_open_joins_the_window_a_racing_replica_opened_first():
    winner: Final = SimpleNamespace(window_start=WINDOW_START.replace(tzinfo=None), spend=2.5)
    database: Final = _OutracedDatabase(_Table(), winner)
    windows: Final = DatabaseSharedWindows(
        prisma_client_source=lambda: SimpleNamespace(db=database, writer_db=database),  # pyright: ignore[reportArgumentType]  # duck-typed prisma double
        sink=_Sink(),
    )

    started_at: Final = await windows.open(spend_key=SPEND_KEY, budget_duration="1d", now=WINDOW_START.timestamp() + 60)

    assert started_at == WINDOW_START.timestamp()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_start", [None, WINDOW_START - timedelta(days=2)])
async def test_open_starts_a_window_at_now_when_none_is_open(stored_start: datetime | None):
    table: Final = _Table(
        row=None if stored_start is None else SimpleNamespace(window_start=stored_start.replace(tzinfo=None), spend=9.0)
    )
    windows: Final = _database_windows(table, _Sink())

    started_at: Final = await windows.open(spend_key=SPEND_KEY, budget_duration="1d", now=WINDOW_START.timestamp())

    assert started_at == WINDOW_START.timestamp()
    assert await windows.read(spend_key=SPEND_KEY, budget_duration="1d") == SharedWindow(
        started_at=WINDOW_START.timestamp(), spend=0.0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("table", [None, _Table(error=RuntimeError("database is down"))])
async def test_open_falls_back_to_this_replicas_own_window_without_a_database(table: _Table | None):
    windows: Final = _database_windows(table, _Sink())

    assert await windows.open(spend_key=SPEND_KEY, budget_duration="1d", now=WINDOW_START.timestamp()) == (
        WINDOW_START.timestamp()
    )
