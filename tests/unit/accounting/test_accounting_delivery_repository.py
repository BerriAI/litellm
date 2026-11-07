from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
    DeliveryMetrics,
)


class _MockTransaction:
    def __init__(self, query_raw_result: Sequence[Mapping[str, object]]) -> None:
        self.query_raw = AsyncMock(return_value=query_raw_result)
        self.execute_raw = AsyncMock(return_value=0)


class _MockLedgerDatabase:
    def __init__(self, transaction: _MockTransaction) -> None:
        self._tx = transaction

    @asynccontextmanager
    async def tx(self, *, timeout: timedelta):
        yield self._tx


@pytest.mark.asyncio
async def test_metrics_executes_scoped_scalar_subqueries() -> None:
    expected_row: Final = {
        "pending_count": 5,
        "parked_count": 2,
        "oldest_pending_seconds": 120.5,
        "batch_count": 3,
        "table_bytes": 1048576,
    }
    tx = _MockTransaction([expected_row])
    db = _MockLedgerDatabase(tx)
    repository = AccountingDeliveryRepository(db)

    metrics = await repository.metrics()

    assert tx.query_raw.await_count == 1
    query: Final[str] = tx.query_raw.await_args[0][0]
    # Verify scalar subqueries scope to state = 'pending' and state = 'parked'
    assert "state = 'pending'" in query
    assert "state = 'parked'" in query
    assert "is_batch" in query
    assert "pg_total_relation_size" in query
    # Verify no un-scoped FROM table scan
    assert not query.endswith('FROM "LiteLLM_SpendLogDelivery"')

    assert isinstance(metrics, DeliveryMetrics)
    assert metrics.pending_count == 5
    assert metrics.parked_count == 2
    assert metrics.oldest_pending_seconds == 120.5
    assert metrics.batch_count == 3
    assert metrics.table_bytes == 1048576


@pytest.mark.asyncio
async def test_metrics_handles_zero_pending_and_parked() -> None:
    zero_row: Final = {
        "pending_count": 0,
        "parked_count": 0,
        "oldest_pending_seconds": 0.0,
        "batch_count": 0,
        "table_bytes": 4096,
    }
    tx = _MockTransaction([zero_row])
    db = _MockLedgerDatabase(tx)
    repository = AccountingDeliveryRepository(db)

    metrics = await repository.metrics()

    assert metrics.pending_count == 0
    assert metrics.parked_count == 0
    assert metrics.oldest_pending_seconds == 0.0
    assert metrics.batch_count == 0
    assert metrics.table_bytes == 4096


@pytest.mark.asyncio
async def test_cleanup_applied_validates_parameters() -> None:
    tx = _MockTransaction([])
    db = _MockLedgerDatabase(tx)
    repository = AccountingDeliveryRepository(db)

    with pytest.raises(ValueError, match="positive"):
        await repository.cleanup_applied(retention_days=0)

    with pytest.raises(ValueError, match="positive"):
        await repository.cleanup_applied(batch_size=0)
