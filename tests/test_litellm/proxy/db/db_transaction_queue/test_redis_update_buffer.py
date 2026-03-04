import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.types.caching import RedisPipelineRpushOperation


@pytest.fixture
def mock_redis_cache():
    """Create a mock RedisCache instance"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def redis_update_buffer(mock_redis_cache):
    """Create a RedisUpdateBuffer with a mock RedisCache"""
    return RedisUpdateBuffer(redis_cache=mock_redis_cache)


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_uses_pipeline(redis_update_buffer, mock_redis_cache):
    """
    Verify store_in_memory_spend_updates_in_redis calls async_rpush_pipeline once
    with the correct operations and skips empty queues.
    """
    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[3, 5, 2])

    # Create mock queues - only 3 of 7 have data
    spend_update_queue = AsyncMock()
    spend_update_queue.flush_and_get_aggregated_db_spend_update_transactions = AsyncMock(
        return_value={"key_list_transactions": {"key1": 1.0}}
    )

    daily_spend_queue = AsyncMock()
    daily_spend_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={"user_key1": {"spend": 1.0}}
    )

    daily_team_queue = AsyncMock()
    daily_team_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={"team_key1": {"spend": 2.0}}
    )

    # Empty queues
    daily_org_queue = AsyncMock()
    daily_org_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={}
    )

    daily_end_user_queue = AsyncMock()
    daily_end_user_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value=None
    )

    daily_agent_queue = AsyncMock()
    daily_agent_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={}
    )

    daily_tag_queue = AsyncMock()
    daily_tag_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={}
    )

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=spend_update_queue,
        daily_spend_update_queue=daily_spend_queue,
        daily_team_spend_update_queue=daily_team_queue,
        daily_org_spend_update_queue=daily_org_queue,
        daily_end_user_spend_update_queue=daily_end_user_queue,
        daily_agent_spend_update_queue=daily_agent_queue,
        daily_tag_spend_update_queue=daily_tag_queue,
    )

    # Should be called exactly once (pipeline)
    mock_redis_cache.async_rpush_pipeline.assert_called_once()

    # Verify only 3 operations were included (empty ones skipped)
    call_args = mock_redis_cache.async_rpush_pipeline.call_args
    rpush_list = call_args.kwargs["rpush_list"]
    assert len(rpush_list) == 3


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_all_empty_returns_early(
    redis_update_buffer, mock_redis_cache
):
    """
    When all queues are empty, pipeline should never be called.
    """
    mock_redis_cache.async_rpush_pipeline = AsyncMock()

    # All queues return empty
    empty_queue = AsyncMock()
    empty_queue.flush_and_get_aggregated_db_spend_update_transactions = AsyncMock(
        return_value={}
    )
    empty_daily_queue = AsyncMock()
    empty_daily_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(
        return_value={}
    )

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=empty_queue,
        daily_spend_update_queue=empty_daily_queue,
        daily_team_spend_update_queue=empty_daily_queue,
        daily_org_spend_update_queue=empty_daily_queue,
        daily_end_user_spend_update_queue=empty_daily_queue,
        daily_agent_spend_update_queue=empty_daily_queue,
        daily_tag_spend_update_queue=empty_daily_queue,
    )

    mock_redis_cache.async_rpush_pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_get_all_transactions_from_redis_buffer_pipeline(
    redis_update_buffer, mock_redis_cache
):
    """
    Verify get_all_transactions_from_redis_buffer_pipeline correctly parses
    and aggregates results from async_lpop_pipeline.
    """
    # Simulate pipeline results: slot 0 = spend updates, slots 1-6 = daily categories
    db_spend_json = json.dumps(
        {
            "key_list_transactions": {"key1": 1.0, "key2": 2.0},
            "user_list_transactions": {"user1": 0.5},
            "end_user_list_transactions": {},
            "team_list_transactions": {},
            "team_member_list_transactions": {},
            "org_list_transactions": {},
            "tag_list_transactions": {},
        }
    )
    daily_user_json = json.dumps({"user_key1": {"spend": 1.0, "api_requests": 1}})
    daily_team_json = json.dumps({"team_key1": {"spend": 2.0, "api_requests": 2}})

    mock_redis_cache.async_lpop_pipeline = AsyncMock(
        return_value=[
            [db_spend_json],        # slot 0: db spend updates
            [daily_user_json],      # slot 1: daily user
            [daily_team_json],      # slot 2: daily team
            None,                    # slot 3: daily org (empty)
            None,                    # slot 4: daily end-user (empty)
            None,                    # slot 5: daily agent (empty)
            None,                    # slot 6: daily tag (empty)
        ]
    )

    result = await redis_update_buffer.get_all_transactions_from_redis_buffer_pipeline()

    assert len(result) == 7
    db_spend, daily_user, daily_team, daily_org, daily_end_user, daily_agent, daily_tag = result

    # Verify db spend was parsed correctly
    assert db_spend is not None
    assert db_spend["key_list_transactions"]["key1"] == 1.0
    assert db_spend["key_list_transactions"]["key2"] == 2.0
    assert db_spend["user_list_transactions"]["user1"] == 0.5

    # Verify daily user was parsed
    assert daily_user is not None
    assert daily_user["user_key1"]["spend"] == 1.0

    # Verify daily team was parsed
    assert daily_team is not None
    assert daily_team["team_key1"]["spend"] == 2.0

    # Verify empty slots
    assert daily_org is None
    assert daily_end_user is None
    assert daily_agent is None
    assert daily_tag is None

    # Verify pipeline was called once with correct keys
    mock_redis_cache.async_lpop_pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_get_all_transactions_from_redis_buffer_pipeline_no_redis():
    """When redis_cache is None, should return all Nones"""
    buffer = RedisUpdateBuffer(redis_cache=None)
    result = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert result == (None, None, None, None, None, None, None)
