import asyncio
import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.constants import MAX_IN_MEMORY_QUEUE_FLUSH_COUNT
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue


class AggregatingTestQueue(BaseUpdateQueue):
    def __init__(self):
        super().__init__()
        self.aggregate_calls = 0
        self.continue_aggregation = asyncio.Event()

    async def aggregate_queue_updates(self):
        self.aggregate_calls += 1
        await self.continue_aggregation.wait()
        await self.flush_all_updates_from_in_memory_queue()


@pytest.mark.asyncio
async def test_queue_flush_limit():
    """
    Test to ensure we don't dequeue more than MAX_IN_MEMORY_QUEUE_FLUSH_COUNT items.
    """
    # Arrange
    queue = BaseUpdateQueue()
    # Override maxsize so the queue can hold all test items without blocking.
    # The default LITELLM_ASYNCIO_QUEUE_MAXSIZE (1000) equals MAX_IN_MEMORY_QUEUE_FLUSH_COUNT,
    # so adding more items than that would cause `await queue.put()` to block forever.
    items_to_add = MAX_IN_MEMORY_QUEUE_FLUSH_COUNT + 100
    queue.update_queue = asyncio.Queue(maxsize=items_to_add + 1)

    for i in range(items_to_add):
        await queue.add_update(f"test_update_{i}")

    # Act
    flushed_updates = await queue.flush_all_updates_from_in_memory_queue()

    # Assert
    assert (
        len(flushed_updates) == MAX_IN_MEMORY_QUEUE_FLUSH_COUNT
    ), f"Expected {MAX_IN_MEMORY_QUEUE_FLUSH_COUNT} items, but got {len(flushed_updates)}"

    # Verify remaining items are still in queue
    assert (
        queue.update_queue.qsize() == 100
    ), "Expected 100 items to remain in the queue"


def test_misconfigured_queue_thresholds_warns():
    """
    Test that a warning is logged when MAX_SIZE_IN_MEMORY_QUEUE >= LITELLM_ASYNCIO_QUEUE_MAXSIZE.

    This misconfiguration causes the spend aggregation check in SpendUpdateQueue.add_update()
    to never trigger because asyncio.Queue blocks before qsize() can reach the threshold.
    """
    import litellm.proxy.db.db_transaction_queue.base_update_queue as bq_module

    with (
        patch.object(bq_module, "MAX_SIZE_IN_MEMORY_QUEUE", 2000),
        patch.object(bq_module, "LITELLM_ASYNCIO_QUEUE_MAXSIZE", 1000),
        patch.object(bq_module.verbose_proxy_logger, "warning") as mock_warning,
    ):
        BaseUpdateQueue()
        assert any(
            "Misconfigured queue thresholds" in call_args[0][0]
            for call_args in mock_warning.call_args_list
        )


@pytest.mark.asyncio
async def test_schedule_queue_aggregation_runs_once_while_pending():
    queue = AggregatingTestQueue()
    queue.MAX_SIZE_IN_MEMORY_QUEUE = 2
    await queue.update_queue.put("first")
    await queue.update_queue.put("second")

    queue._schedule_queue_aggregation_if_needed()
    first_task = queue._aggregation_task
    queue._schedule_queue_aggregation_if_needed()

    assert queue._aggregation_task is first_task
    assert first_task is not None

    while queue.aggregate_calls == 0:
        await asyncio.sleep(0)

    queue.continue_aggregation.set()
    await queue._wait_for_pending_aggregation()

    assert queue.aggregate_calls == 1
    assert queue.update_queue.qsize() == 0


@pytest.mark.asyncio
async def test_schedule_queue_aggregation_skips_below_threshold():
    queue = AggregatingTestQueue()
    queue.MAX_SIZE_IN_MEMORY_QUEUE = 2
    await queue.update_queue.put("first")

    queue._schedule_queue_aggregation_if_needed()
    await queue._aggregate_queue_updates_if_needed()

    assert queue._aggregation_task is None
    assert queue.aggregate_calls == 0


@pytest.mark.asyncio
async def test_wait_for_pending_aggregation_ignores_completed_task():
    queue = AggregatingTestQueue()
    queue.continue_aggregation.set()
    queue._aggregation_task = asyncio.create_task(queue.aggregate_queue_updates())
    await queue._aggregation_task

    await queue._wait_for_pending_aggregation()

    assert queue.aggregate_calls == 1


@pytest.mark.asyncio
async def test_log_aggregation_task_exception_handles_errors():
    queue = AggregatingTestQueue()

    async def raise_error():
        raise RuntimeError("boom")

    task = asyncio.create_task(raise_error())
    await asyncio.sleep(0)

    with patch(
        "litellm.proxy.db.db_transaction_queue.base_update_queue.verbose_proxy_logger.error"
    ) as mock_error:
        queue._log_aggregation_task_exception(task)

    mock_error.assert_called_once()


@pytest.mark.asyncio
async def test_log_aggregation_task_exception_ignores_cancelled_tasks():
    queue = AggregatingTestQueue()

    async def wait_forever():
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever())
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    with patch(
        "litellm.proxy.db.db_transaction_queue.base_update_queue.verbose_proxy_logger.error"
    ) as mock_error:
        queue._log_aggregation_task_exception(task)

    mock_error.assert_not_called()
