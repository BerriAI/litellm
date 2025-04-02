import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import Litellm_EntityType, SpendUpdateQueueItem
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


@pytest.fixture
def spend_queue():
    return SpendUpdateQueue()


@pytest.mark.asyncio
async def test_add_update(spend_queue):
    # Test adding a single update
    update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "entity_id": "user123",
        "response_cost": 0.5,
    }
    await spend_queue.add_update(update)

    # Verify update was added by checking queue size
    assert spend_queue.update_queue.qsize() == 1


@pytest.mark.asyncio
async def test_missing_response_cost(spend_queue):
    # Test with missing response_cost - should default to 0
    update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "entity_id": "user123",
    }

    await spend_queue.add_update(update)
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Should have created entry with 0 cost
    assert aggregated["user_list_transactions"]["user123"] == 0


@pytest.mark.asyncio
async def test_missing_entity_id(spend_queue):
    # Test with missing entity_id - should default to empty string
    update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "response_cost": 1.0,
    }

    await spend_queue.add_update(update)
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Should use empty string as key
    assert aggregated["user_list_transactions"][""] == 1.0


@pytest.mark.asyncio
async def test_none_values(spend_queue):
    # Test with None values
    update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "entity_id": None,  # type: ignore
        "response_cost": None,
    }

    await spend_queue.add_update(update)
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Should handle None values gracefully
    assert aggregated["user_list_transactions"][""] == 0


@pytest.mark.asyncio
async def test_multiple_updates_with_missing_fields(spend_queue):
    # Test multiple updates with various missing fields
    updates: list[SpendUpdateQueueItem] = [
        {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user123",
            "response_cost": 0.5,
        },
        {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user123",  # missing response_cost
        },
        {
            "entity_type": Litellm_EntityType.USER,  # missing entity_id
            "response_cost": 1.5,
        },
    ]

    for update in updates:
        await spend_queue.add_update(update)

    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Verify aggregation
    assert (
        aggregated["user_list_transactions"]["user123"] == 0.5
    )  # only the first update with valid cost
    assert (
        aggregated["user_list_transactions"][""] == 1.5
    )  # update with missing entity_id


@pytest.mark.asyncio
async def test_unknown_entity_type(spend_queue):
    # Test with unknown entity type
    update: SpendUpdateQueueItem = {
        "entity_type": "UNKNOWN_TYPE",  # type: ignore
        "entity_id": "123",
        "response_cost": 0.5,
    }

    await spend_queue.add_update(update)
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Should ignore unknown entity type
    assert all(len(transactions) == 0 for transactions in aggregated.values())


@pytest.mark.asyncio
async def test_missing_entity_type(spend_queue):
    # Test with missing entity type
    update: SpendUpdateQueueItem = {"entity_id": "123", "response_cost": 0.5}

    await spend_queue.add_update(update)
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )

    # Should ignore updates without entity type
    assert all(len(transactions) == 0 for transactions in aggregated.values())
