import asyncio
import json
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import Litellm_EntityType, SpendUpdateQueueItem
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue



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
async def test_repeated_updates_share_one_queue_entry(spend_queue: SpendUpdateQueue) -> None:
    for _ in range(6):
        update: Final[SpendUpdateQueueItem] = {
            "entity_type": Litellm_EntityType.USER,
            "entity_id": "user123",
            "response_cost": 1.0,
        }
        await spend_queue.add_update(update)

    assert spend_queue.update_queue.qsize() == 1
    aggregated: Final = await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
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
async def test_queue_size_reduction_with_large_volume(spend_queue: SpendUpdateQueue) -> None:
    for _ in range(200):
        await spend_queue.add_update(
            {
                "entity_type": Litellm_EntityType.USER,
                "entity_id": "user1",
                "response_cost": 0.5,
            }
        )

    assert spend_queue.update_queue.qsize() == 1

    for _ in range(300):
        await spend_queue.add_update(
            {
                "entity_type": Litellm_EntityType.KEY,
                "entity_id": "key1",
                "response_cost": 1.0,
            }
        )

    assert spend_queue.update_queue.qsize() == 2
    aggregated: Final = await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert aggregated["user_list_transactions"]["user1"] == 200 * 0.5
    assert aggregated["key_list_transactions"]["key1"] == 300 * 1.0


@pytest.mark.asyncio
async def test_high_cardinality_updates_do_not_rescan_the_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: Final = SpendUpdateQueue()
    queue.MAX_SIZE_IN_MEMORY_QUEUE = 4
    aggregate: Final = AsyncMock(wraps=queue.aggregate_queue_updates)
    monkeypatch.setattr(queue, "aggregate_queue_updates", aggregate)
    updates: Final = tuple(
        SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id=f"key-{index}", response_cost=0.25)
        for index in range(6)
    )

    for _ in range(20):
        for update in updates:
            await queue.add_update(update)

    aggregate.assert_not_awaited()
    assert queue.update_queue.qsize() == len(updates)
    flushed: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert flushed["key_list_transactions"] == {f"key-{index}": 5.0 for index in range(6)}
    assert all(update["response_cost"] == 0.25 for update in updates)


@pytest.mark.asyncio
async def test_full_queue_coalesces_existing_keys_and_cancels_new_keys() -> None:
    queue: Final = SpendUpdateQueue()
    queue.update_queue = asyncio.Queue(maxsize=1)
    first: Final = SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="key", response_cost=1.0)
    await queue.add_update(first)
    await asyncio.wait_for(queue.add_update(first), timeout=1)
    blocked: Final = asyncio.create_task(
        queue.add_update(SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="cancelled", response_cost=9.0))
    )
    try:
        await asyncio.sleep(0)
        assert not blocked.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
    finally:
        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)

    flushed: Final = await queue.flush_all_updates_from_in_memory_queue()
    assert flushed == [SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="key", response_cost=2.0)]
    await queue.add_update(first)
    assert flushed[0]["response_cost"] == 2.0
    next_batch: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert next_batch["key_list_transactions"] == {"key": 1.0}
    assert first["response_cost"] == 1.0


@pytest.mark.asyncio
async def test_partial_flush_keeps_pending_costs_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.db.db_transaction_queue import base_update_queue

    monkeypatch.setattr(base_update_queue, "MAX_IN_MEMORY_QUEUE_FLUSH_COUNT", 1)
    queue: Final = SpendUpdateQueue()
    first: Final = SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="first", response_cost=1.0)
    second: Final = SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="second", response_cost=2.0)
    await queue.add_update(first)
    await queue.add_update(second)
    flushed: Final = await queue.flush_all_updates_from_in_memory_queue()
    await queue.add_update(first)
    await queue.add_update(second)

    assert flushed == [first]
    second_batch: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert second_batch["key_list_transactions"] == {"second": 4.0}
    third_batch: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert third_batch["key_list_transactions"] == {"first": 1.0}


@pytest.mark.asyncio
async def test_partial_flush_preserves_same_key_admitted_by_waiting_producers(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.db.db_transaction_queue import base_update_queue

    queue: Final = SpendUpdateQueue()
    queue.update_queue = asyncio.Queue(maxsize=2)
    for key in ("first", "second"):
        await queue.add_update(SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id=key, response_cost=1.0))
    duplicate: Final = SpendUpdateQueueItem(entity_type=Litellm_EntityType.KEY, entity_id="shared", response_cost=1.0)
    producers: Final = tuple(asyncio.create_task(queue.add_update(duplicate)) for _ in range(2))
    try:
        await asyncio.sleep(0)
        assert all(not producer.done() for producer in producers)
        await queue.flush_all_updates_from_in_memory_queue()
        await asyncio.wait_for(asyncio.gather(*producers), timeout=1)
    finally:
        for producer in producers:
            producer.cancel()
        await asyncio.gather(*producers, return_exceptions=True)

    monkeypatch.setattr(base_update_queue, "MAX_IN_MEMORY_QUEUE_FLUSH_COUNT", 1)
    first_batch: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert first_batch["key_list_transactions"] == {"shared": 1.0}
    await queue.add_update(duplicate)
    second_batch: Final = await queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert second_batch["key_list_transactions"] == {"shared": 2.0}
    assert queue.update_queue.empty()
