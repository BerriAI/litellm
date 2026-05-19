import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from litellm.constants import MAX_SIZE_IN_MEMORY_QUEUE
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


@pytest.mark.asyncio
async def test_queue_max_size_triggers_aggregation(monkeypatch, spend_queue):
    """Test that reaching MAX_SIZE_IN_MEMORY_QUEUE triggers aggregation"""
    # Override MAX_SIZE_IN_MEMORY_QUEUE for testing
    monkeypatch.setattr(spend_queue, "MAX_SIZE_IN_MEMORY_QUEUE", 6)

    # Add 6 updates for the same user (exceeding the max size)
    for i in range(6):
        update: SpendUpdateQueueItem = {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user123",
            "response_cost": 1.0,
        }
        await spend_queue.add_update(update)

    # Queue should have been aggregated, resulting in a single entry
    assert spend_queue.update_queue.qsize() == 1

    # Verify the aggregated cost is correct
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )
    assert aggregated["user_list_transactions"]["user123"] == 6.0


@pytest.mark.asyncio
async def test_aggregate_queue_updates_accuracy(spend_queue):
    """Test that queue aggregation correctly combines costs by entity type and ID"""
    # Add multiple updates for different entities
    updates = [
        {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user1",
            "response_cost": 1.5,
        },
        {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user1",
            "response_cost": 2.5,
        },
        {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user2",
            "response_cost": 3.0,
        },
        {
            "entity_type": Litellm_EntityType.TEAM,
            "entity_id": "team1",
            "response_cost": 5.0,
        },
    ]

    for update in updates:
        await spend_queue.update_queue.put(update)

    # Force aggregation
    await spend_queue.aggregate_queue_updates()

    # Queue size should now be 3 (user1, user2, team1)
    assert spend_queue.update_queue.qsize() == 3

    # Flush and verify aggregated values
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )
    print("aggregated values", aggregated)

    assert aggregated["user_list_transactions"]["user1"] == 4.0  # 1.5 + 2.5
    assert aggregated["user_list_transactions"]["user2"] == 3.0
    assert aggregated["team_list_transactions"]["team1"] == 5.0


def test_get_aggregated_spend_update_queue_item_does_not_mutate_original_updates(
    spend_queue,
):
    original_update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "entity_id": "user1",
        "response_cost": 10.0,
    }
    duplicate_key_update: SpendUpdateQueueItem = {
        "entity_type": Litellm_EntityType.USER,
        "entity_id": "user1",
        "response_cost": 20.0,
    }

    aggregated_updates = spend_queue._get_aggregated_spend_update_queue_item(
        [original_update, duplicate_key_update]
    )
    user1_aggregated_update = next(
        (
            update
            for update in aggregated_updates
            if update.get("entity_type") == Litellm_EntityType.USER
            and update.get("entity_id") == "user1"
        ),
        None,
    )

    assert original_update["response_cost"] == 10.0
    assert user1_aggregated_update is not None
    assert user1_aggregated_update["response_cost"] == 30.0
    assert user1_aggregated_update is not original_update


@pytest.mark.asyncio
async def test_queue_size_reduction_with_large_volume(monkeypatch, spend_queue):
    """Test that queue size is actually reduced when dealing with many items"""
    # Set a smaller MAX_SIZE for testing
    monkeypatch.setattr(spend_queue, "MAX_SIZE_IN_MEMORY_QUEUE", 10)

    # Add 30 updates (200 for user1, 10 for key1)
    for i in range(200):
        await spend_queue.add_update(
            {
                "entity_type": Litellm_EntityType.USER,
                "entity_id": "user1",
                "response_cost": 0.5,
            }
        )

    # At this point, aggregation should have happened at least once
    # Queue size should be much less than 20
    assert spend_queue.update_queue.qsize() <= 10

    for i in range(300):
        await spend_queue.add_update(
            {
                "entity_type": Litellm_EntityType.KEY,
                "entity_id": "key1",
                "response_cost": 1.0,
            }
        )

    # Queue should have at most 2 items after all this activity
    assert spend_queue.update_queue.qsize() <= 10

    # Verify total costs are correct
    aggregated = (
        await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    )
    assert aggregated["user_list_transactions"]["user1"] == 200 * 0.5
    assert aggregated["key_list_transactions"]["key1"] == 300 * 1.0
