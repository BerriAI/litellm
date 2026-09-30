"""
Tests for the CustomBatchLogger-based ClickHouse base logger.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.integrations.clickhouse import clickhouse_batch_logger as module
from litellm.integrations.clickhouse.clickhouse_batch_logger import ClickHouseBatchLogger
from litellm.integrations.custom_batch_logger import CustomBatchLogger


class _TestLogger(ClickHouseBatchLogger):
    table = "test_table"


def _logger(insert: AsyncMock) -> _TestLogger:
    client = MagicMock()
    client.insert_json_each_row = insert
    return _TestLogger(client=client)


def test_is_a_custom_batch_logger():
    assert issubclass(ClickHouseBatchLogger, CustomBatchLogger)


@pytest.mark.asyncio
async def test_flush_splits_into_batches_and_empties_queue():
    insert = AsyncMock()
    logger = _logger(insert)
    logger.batch_size = 2
    logger.log_queue.extend([{"i": i} for i in range(5)])

    await logger.flush_queue()

    assert [len(c.args[1]) for c in insert.await_args_list] == [2, 2, 1]
    assert all(c.args[0] == "test_table" for c in insert.await_args_list)
    assert logger.log_queue == []
    assert logger.rows_written == 5


@pytest.mark.asyncio
async def test_is_full_signals_backpressure():
    logger = _logger(AsyncMock())
    with patch.object(module, "CLICKHOUSE_MAX_BUFFERED_ROWS", 3):
        logger.log_queue.extend([{}, {}])
        assert logger.is_full() is False
        logger.log_queue.append({})
        assert logger.is_full() is True


@pytest.mark.asyncio
async def test_failed_insert_is_requeued_then_dropped():
    insert = AsyncMock(side_effect=RuntimeError("clickhouse down"))
    logger = _logger(insert)
    logger.log_queue.extend([{"request_id": "a"}, {"request_id": "b"}])

    with patch.object(module, "CLICKHOUSE_MAX_RETRIES", 2):
        await logger.flush_queue()
        assert len(logger.log_queue) == 2  # kept for retry
        await logger.flush_queue()

    assert insert.await_count == 2
    assert logger.rows_dropped == 2
    assert logger.rows_written == 0
    assert logger.log_queue == []
