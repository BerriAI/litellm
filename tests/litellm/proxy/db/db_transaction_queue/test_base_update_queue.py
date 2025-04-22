import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.constants import MAX_IN_MEMORY_QUEUE_FLUSH_COUNT
from litellm.proxy.db.db_transaction_queue.base_update_queue import BaseUpdateQueue


@pytest.mark.asyncio
async def test_queue_flush_limit():
    """
    Test to ensure we don't dequeue more than MAX_IN_MEMORY_QUEUE_FLUSH_COUNT items.
    """
    # Arrange
    queue = BaseUpdateQueue()
    # Add more items than the max flush count
    items_to_add = MAX_IN_MEMORY_QUEUE_FLUSH_COUNT + 100

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
