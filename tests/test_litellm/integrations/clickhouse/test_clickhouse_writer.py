"""
Tests for the batched ClickHouse writer used by agent tracing.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.integrations.clickhouse import clickhouse_writer as writer_module
from litellm.integrations.clickhouse.clickhouse_writer import ClickHouseWriter


def _writer(insert: AsyncMock) -> ClickHouseWriter:
    client = MagicMock()
    client.insert_json_each_row = insert
    return ClickHouseWriter(client=client)


@pytest.mark.asyncio
async def test_flush_splits_into_batches():
    insert = AsyncMock()
    writer = _writer(insert)
    writer.enqueue("otel_traces", [{"i": i} for i in range(5)])

    with patch.object(writer_module, "CLICKHOUSE_BATCH_SIZE", 2):
        await writer.flush()

    assert [len(c.args[1]) for c in insert.await_args_list] == [2, 2, 1]
    assert writer.buffers["otel_traces"] == []
    assert writer.rows_written["otel_traces"] == 5


@pytest.mark.asyncio
async def test_is_full_signals_backpressure():
    writer = _writer(AsyncMock())
    with patch.object(writer_module, "CLICKHOUSE_MAX_BUFFERED_ROWS", 3):
        writer.enqueue("otel_traces", [{}, {}])
        assert writer.is_full("otel_traces") is False
        writer.enqueue("otel_traces", [{}])
        assert writer.is_full("otel_traces") is True


@pytest.mark.asyncio
async def test_failed_batch_is_retried_then_dropped():
    insert = AsyncMock(side_effect=RuntimeError("clickhouse down"))
    writer = _writer(insert)
    writer.enqueue("spend_logs", [{"request_id": "a"}, {"request_id": "b"}])

    with (
        patch.object(writer_module, "CLICKHOUSE_MAX_RETRIES", 2),
        patch.object(writer_module.asyncio, "sleep", AsyncMock()),
    ):
        await writer.flush()

    assert insert.await_count == 2
    assert writer.rows_dropped["spend_logs"] == 2
    assert writer.rows_written["spend_logs"] == 0
