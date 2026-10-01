import asyncio
import threading
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.clickhouse.clickhouse_batch_logger import ClickHouseBatchLogger


class SpendWriter(ClickHouseBatchLogger):
    table = "spend_logs"


@pytest.mark.asyncio
async def test_shutdown_delivers_all_accepted_rows_in_bounded_batches() -> None:
    insert: Final = AsyncMock()
    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert), batch_size=2)
    writer.enqueue(tuple({"request_id": str(i)} for i in range(5)))

    await writer.aclose()

    batches: Final = tuple(call.args[1] for call in insert.await_args_list)
    assert tuple(len(batch) for batch in batches) == (2, 2, 1)
    assert tuple(row["request_id"] for row in chain.from_iterable(batches)) == tuple(str(i) for i in range(5))
    assert writer.rows_written == 5
    assert writer.rows_dropped == writer.buffered_rows == writer.buffered_bytes == 0


@pytest.mark.asyncio
async def test_failed_batch_keeps_its_retry_identity_when_new_rows_arrive() -> None:
    insert: Final = AsyncMock(side_effect=[RuntimeError("offline"), None, None])
    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert), flush_interval=60)
    writer.enqueue(({"request_id": "first"},))
    await writer.flush_queue()
    writer.enqueue(({"request_id": "second"},))

    await writer.flush_queue()
    await writer.aclose()

    calls: Final = insert.await_args_list
    assert calls[0].args == calls[1].args
    assert calls[2].args[1] == ({"request_id": "second"},)
    assert writer.rows_written == 2 and writer.rows_dropped == 0


@pytest.mark.asyncio
async def test_retry_exhaustion_counts_lost_rows() -> None:
    writer: Final = SpendWriter(
        storage=MagicMock(insert_rows=AsyncMock(side_effect=RuntimeError("offline"))), max_retries=2
    )
    writer.enqueue(({"request_id": "a"}, {"request_id": "b"}))
    await writer.flush_queue()
    assert writer.buffered_rows == 2
    await writer.flush_queue()
    await writer.aclose()
    assert writer.rows_dropped == 2 and writer.rows_written == 0 and writer.buffered_rows == 0


@pytest.mark.asyncio
async def test_capacity_includes_inflight_rows_and_shutdown_reports_cancellation() -> None:
    entered: Final = threading.Event()

    async def insert_rows(*args: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert_rows), batch_size=1, max_rows=3)
    writer.enqueue(({"request_id": "inflight"},))
    assert await asyncio.to_thread(entered.wait, 2)
    writer.enqueue(tuple({"request_id": str(i)} for i in range(3)))
    assert writer.buffered_rows == 3
    assert writer.rows_dropped == 1
    assert writer.is_full()

    await writer.aclose(timeout=0.05)

    assert writer.rows_dropped == 4
    assert writer.rows_written == writer.buffered_rows == writer.buffered_bytes == 0


@pytest.mark.asyncio
async def test_byte_limit_rejects_oversized_rows_without_losing_small_rows() -> None:
    insert: Final = AsyncMock()
    writer: Final = SpendWriter(
        storage=MagicMock(insert_rows=insert), max_row_bytes=40, max_bytes=80, max_batch_bytes=60
    )
    writer.enqueue(({"value": "x" * 100}, {"value": "a" * 15}, {"value": "b" * 15}, {"value": "c" * 15}))
    assert writer.buffered_bytes <= 80
    assert writer.rows_dropped == 2

    await writer.aclose()

    assert writer.rows_written == 2
    assert tuple(call.args[1] for call in insert.await_args_list) == (({"value": "a" * 15}, {"value": "b" * 15}),)


def test_synchronous_submission_does_not_require_a_running_event_loop() -> None:
    insert: Final = AsyncMock()
    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert))
    writer.enqueue(({"request_id": "sync"},))
    asyncio.run(writer.aclose())
    assert insert.await_args.args[1] == ({"request_id": "sync"},)
    assert writer.rows_written == 1


@pytest.mark.asyncio
async def test_cancelled_flush_preserves_inflight_batch_for_retry() -> None:
    entered: Final = threading.Event()
    attempts: Final = AsyncMock()

    async def insert_rows(*args: object) -> None:
        await attempts(*args)
        if attempts.await_count == 1:
            entered.set()
            await asyncio.Event().wait()

    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert_rows), flush_interval=60)
    writer.enqueue(({"request_id": "event"},))
    flush: Final = asyncio.create_task(writer.flush_queue())
    assert await asyncio.to_thread(entered.wait, 2)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    await writer.flush_queue()
    await writer.aclose()
    assert attempts.await_args_list[0].args == attempts.await_args_list[1].args
    assert writer.rows_written == 1 and writer.rows_dropped == 0


@pytest.mark.asyncio
async def test_storage_startup_failure_buffers_rows_until_recovery() -> None:
    prepare: Final = AsyncMock(side_effect=[RuntimeError("starting"), None])
    insert: Final = AsyncMock()
    writer: Final = SpendWriter(storage=MagicMock(insert_rows=insert), prepare=prepare, flush_interval=60)
    writer.enqueue(({"request_id": "early"},))
    await writer.flush_queue()
    insert.assert_not_awaited()
    assert writer.buffered_rows == 1
    await writer.flush_queue()
    await writer.aclose()
    assert writer.rows_written == 1 and writer.rows_dropped == 0
