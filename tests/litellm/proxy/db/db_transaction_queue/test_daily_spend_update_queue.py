import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.constants import MAX_SIZE_IN_MEMORY_QUEUE
from litellm.proxy._types import (
    DailyUserSpendTransaction,
    Litellm_EntityType,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
    DailySpendUpdateQueue,
)
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue


@pytest.fixture
def daily_spend_update_queue():
    return DailySpendUpdateQueue()


@pytest.mark.asyncio
async def test_empty_queue_flush(daily_spend_update_queue):
    """Test flushing an empty queue returns an empty list"""
    result = await daily_spend_update_queue.flush_all_updates_from_in_memory_queue()
    assert result == []


@pytest.mark.asyncio
async def test_add_single_update(daily_spend_update_queue):
    """Test adding a single update to the queue"""
    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    # Add update to queue
    await daily_spend_update_queue.add_update({test_key: test_transaction})

    # Flush and check
    updates = await daily_spend_update_queue.flush_all_updates_from_in_memory_queue()
    assert len(updates) == 1
    assert test_key in updates[0]
    assert updates[0][test_key] == test_transaction


@pytest.mark.asyncio
async def test_add_multiple_updates(daily_spend_update_queue):
    """Test adding multiple updates to the queue"""
    test_key1 = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction1 = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    test_key2 = "user2_2023-01-01_key456_gpt-3.5-turbo_openai"
    test_transaction2 = {
        "spend": 5.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    # Add updates to queue
    await daily_spend_update_queue.add_update({test_key1: test_transaction1})
    await daily_spend_update_queue.add_update({test_key2: test_transaction2})

    # Flush and check
    updates = await daily_spend_update_queue.flush_all_updates_from_in_memory_queue()
    assert len(updates) == 2

    # Find each transaction in the list of updates
    found_transaction1 = False
    found_transaction2 = False

    for update in updates:
        if test_key1 in update:
            assert update[test_key1] == test_transaction1
            found_transaction1 = True
        if test_key2 in update:
            assert update[test_key2] == test_transaction2
            found_transaction2 = True

    assert found_transaction1
    assert found_transaction2


@pytest.mark.asyncio
async def test_aggregated_daily_spend_update_empty(daily_spend_update_queue):
    """Test aggregating updates from an empty queue"""
    result = (
        await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    )
    assert result == {}


@pytest.mark.asyncio
async def test_get_aggregated_daily_spend_update_transactions_single_key():
    """Test static method for aggregating a single key"""
    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    updates = [{test_key: test_transaction}]

    # Test aggregation
    result = DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
        updates
    )

    assert len(result) == 1
    assert test_key in result
    assert result[test_key] == test_transaction


@pytest.mark.asyncio
async def test_get_aggregated_daily_spend_update_transactions_multiple_keys():
    """Test static method for aggregating multiple different keys"""
    test_key1 = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction1 = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    test_key2 = "user2_2023-01-01_key456_gpt-3.5-turbo_openai"
    test_transaction2 = {
        "spend": 5.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    updates = [{test_key1: test_transaction1}, {test_key2: test_transaction2}]

    # Test aggregation
    result = DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
        updates
    )

    assert len(result) == 2
    assert test_key1 in result
    assert test_key2 in result
    assert result[test_key1] == test_transaction1
    assert result[test_key2] == test_transaction2


@pytest.mark.asyncio
async def test_get_aggregated_daily_spend_update_transactions_same_key():
    """Test static method for aggregating updates with the same key"""
    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction1 = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    test_transaction2 = {
        "spend": 5.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    expected_transaction = {
        "spend": 15.0,  # 10 + 5
        "prompt_tokens": 300,  # 100 + 200
        "completion_tokens": 80,  # 50 + 30
        "api_requests": 2,  # 1 + 1
        "successful_requests": 2,  # 1 + 1
        "failed_requests": 0,  # 0 + 0
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    updates = [{test_key: test_transaction1}, {test_key: test_transaction2}]

    # Test aggregation
    result = DailySpendUpdateQueue.get_aggregated_daily_spend_update_transactions(
        updates
    )

    assert len(result) == 1
    assert test_key in result
    assert result[test_key] == expected_transaction


@pytest.mark.asyncio
async def test_flush_and_get_aggregated_daily_spend_update_transactions(
    daily_spend_update_queue,
):
    """Test the full workflow of adding, flushing, and aggregating updates"""
    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction1 = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    test_transaction2 = {
        "spend": 5.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    expected_transaction = {
        "spend": 15.0,  # 10 + 5
        "prompt_tokens": 300,  # 100 + 200
        "completion_tokens": 80,  # 50 + 30
        "api_requests": 2,  # 1 + 1
        "successful_requests": 2,  # 1 + 1
        "failed_requests": 0,  # 0 + 0
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    # Add updates to queue
    await daily_spend_update_queue.add_update({test_key: test_transaction1})
    await daily_spend_update_queue.add_update({test_key: test_transaction2})

    # Flush and get aggregated transactions
    result = (
        await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    )

    assert len(result) == 1
    assert test_key in result
    assert result[test_key] == expected_transaction


@pytest.mark.asyncio
async def test_queue_max_size_triggers_aggregation(
    monkeypatch, daily_spend_update_queue
):
    """Test that reaching MAX_SIZE_IN_MEMORY_QUEUE triggers aggregation"""
    # Override MAX_SIZE_IN_MEMORY_QUEUE for testing
    litellm._turn_on_debug()
    monkeypatch.setattr(daily_spend_update_queue, "MAX_SIZE_IN_MEMORY_QUEUE", 6)

    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction = {
        "spend": 1.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    # Add 6 identical updates (exceeding the max size of 5)
    for i in range(6):
        await daily_spend_update_queue.add_update({test_key: test_transaction})

    # Queue should have aggregated to a single item
    assert daily_spend_update_queue.update_queue.qsize() == 1

    # Verify the aggregated values
    result = (
        await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    )
    assert result[test_key]["spend"] == 6.0
    assert result[test_key]["prompt_tokens"] == 600
    assert result[test_key]["completion_tokens"] == 300
    assert result[test_key]["api_requests"] == 6
    assert result[test_key]["successful_requests"] == 6
    assert result[test_key]["failed_requests"] == 0


@pytest.mark.asyncio
async def test_aggregate_queue_updates_accuracy(daily_spend_update_queue):
    """Test that queue aggregation correctly combines metrics by transaction key"""
    # Add multiple updates for different transaction keys
    test_key1 = "user1_2023-01-01_key123_gpt-4_openai"
    test_transaction1 = {
        "spend": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    test_key2 = "user1_2023-01-01_key123_gpt-4_openai"  # Same key
    test_transaction2 = {
        "spend": 5.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 0,
        "failed_requests": 1,
    }

    test_key3 = "user2_2023-01-01_key456_gpt-3.5-turbo_openai"  # Different key
    test_transaction3 = {
        "spend": 3.0,
        "prompt_tokens": 150,
        "completion_tokens": 25,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    # Add updates directly to the queue
    await daily_spend_update_queue.update_queue.put({test_key1: test_transaction1})
    await daily_spend_update_queue.update_queue.put({test_key2: test_transaction2})
    await daily_spend_update_queue.update_queue.put({test_key3: test_transaction3})

    # Force aggregation
    await daily_spend_update_queue.aggregate_queue_updates()
    updates = await daily_spend_update_queue.flush_all_updates_from_in_memory_queue()
    print("AGGREGATED UPDATES", json.dumps(updates, indent=4))
    daily_spend_update_transactions = updates[0]

    # Should have 2 keys after aggregation (test_key1/test_key2 combined, and test_key3)
    assert len(daily_spend_update_transactions) == 2

    # Check aggregated values for test_key1 (which is the same as test_key2)
    assert daily_spend_update_transactions[test_key1]["spend"] == 15.0
    assert daily_spend_update_transactions[test_key1]["prompt_tokens"] == 300
    assert daily_spend_update_transactions[test_key1]["completion_tokens"] == 80
    assert daily_spend_update_transactions[test_key1]["api_requests"] == 2
    assert daily_spend_update_transactions[test_key1]["successful_requests"] == 1
    assert daily_spend_update_transactions[test_key1]["failed_requests"] == 1

    # Check values for test_key3 remain the same
    assert daily_spend_update_transactions[test_key3]["spend"] == 3.0
    assert daily_spend_update_transactions[test_key3]["prompt_tokens"] == 150
    assert daily_spend_update_transactions[test_key3]["completion_tokens"] == 25
    assert daily_spend_update_transactions[test_key3]["api_requests"] == 1
    assert daily_spend_update_transactions[test_key3]["successful_requests"] == 1
    assert daily_spend_update_transactions[test_key3]["failed_requests"] == 0


@pytest.mark.asyncio
async def test_cache_token_fields_aggregation(daily_spend_update_queue):
    """Test that cache_read_input_tokens and cache_creation_input_tokens are handled and aggregated correctly."""
    test_key = "user1_2023-01-01_key123_gpt-4_openai"
    transaction1 = {
        "spend": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "cache_read_input_tokens": 7,
        "cache_creation_input_tokens": 3,
    }
    transaction2 = {
        "spend": 2.0,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 4,
    }
    # Add both updates
    await daily_spend_update_queue.add_update({test_key: transaction1})
    await daily_spend_update_queue.add_update({test_key: transaction2})
    # Aggregate
    await daily_spend_update_queue.aggregate_queue_updates()
    updates = await daily_spend_update_queue.flush_all_updates_from_in_memory_queue()
    assert len(updates) == 1
    agg = updates[0][test_key]
    assert agg["cache_read_input_tokens"] == 12  # 7 + 5
    assert agg["cache_creation_input_tokens"] == 7  # 3 + 4
    assert agg["spend"] == 3.0
    assert agg["prompt_tokens"] == 30
    assert agg["completion_tokens"] == 15
    assert agg["api_requests"] == 2
    assert agg["successful_requests"] == 2
    assert agg["failed_requests"] == 0


@pytest.mark.asyncio
async def test_queue_size_reduction_with_large_volume(
    monkeypatch, daily_spend_update_queue
):
    """Test that queue size is actually reduced when dealing with many items"""
    # Set a smaller MAX_SIZE for testing
    monkeypatch.setattr(daily_spend_update_queue, "MAX_SIZE_IN_MEMORY_QUEUE", 10)

    # Create transaction templates
    user1_key = "user1_2023-01-01_key123_gpt-4_openai"
    user1_transaction = {
        "spend": 0.5,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    user2_key = "user2_2023-01-01_key456_gpt-3.5-turbo_openai"
    user2_transaction = {
        "spend": 1.0,
        "prompt_tokens": 200,
        "completion_tokens": 30,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
    }

    # Add 30 updates (200 for user1, 10 for user2)
    for i in range(200):
        await daily_spend_update_queue.add_update({user1_key: user1_transaction})

    # At this point, aggregation should have happened at least once
    # Queue size should be much less than 10
    assert daily_spend_update_queue.update_queue.qsize() <= 10

    for i in range(100):
        await daily_spend_update_queue.add_update({user2_key: user2_transaction})

    # Queue should have at most 10 items after all this activity
    assert daily_spend_update_queue.update_queue.qsize() <= 10

    # Verify total costs are correct
    result = (
        await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    )
    print("RESULT", json.dumps(result, indent=4))

    assert result[user1_key]["spend"] == 200 * 0.5  # 10.0
    assert result[user1_key]["prompt_tokens"] == 200 * 100  # 2000
    assert result[user1_key]["completion_tokens"] == 200 * 50  # 1000
    assert result[user1_key]["api_requests"] == 200
    assert result[user1_key]["successful_requests"] == 200
    assert result[user1_key]["failed_requests"] == 0

    assert result[user2_key]["spend"] == 100 * 1.0  # 10.0
    assert result[user2_key]["prompt_tokens"] == 100 * 200  # 2000
    assert result[user2_key]["completion_tokens"] == 100 * 30  # 300
    assert result[user2_key]["api_requests"] == 100
    assert result[user2_key]["successful_requests"] == 100
    assert result[user2_key]["failed_requests"] == 0
