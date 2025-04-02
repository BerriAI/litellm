import asyncio
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    DailyUserSpendTransaction,
    Litellm_EntityType,
    SpendUpdateQueueItem,
)
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
    DailySpendUpdateQueue,
)
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


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
    }

    # Add updates to queue
    await daily_spend_update_queue.add_update({test_key: test_transaction1})
    await daily_spend_update_queue.add_update({test_key: test_transaction2})

    # Test full workflow
    result = (
        await daily_spend_update_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    )

    assert len(result) == 1
    assert test_key in result
    assert result[test_key] == expected_transaction
