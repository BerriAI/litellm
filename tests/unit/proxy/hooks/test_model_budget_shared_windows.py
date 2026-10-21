from datetime import datetime, timezone
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


class _Sink:
    def __init__(self) -> None:
        self.transactions: tuple[WindowSpendTransaction, ...] = ()

    async def __call__(self, transaction: WindowSpendTransaction) -> None:
        self.transactions = (*self.transactions, transaction)


def _database_windows(table: _Table | None, sink: _Sink) -> DatabaseSharedWindows:
    prisma: Final = None if table is None else SimpleNamespace(db=SimpleNamespace(litellm_budgetwindowspend=table))
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
