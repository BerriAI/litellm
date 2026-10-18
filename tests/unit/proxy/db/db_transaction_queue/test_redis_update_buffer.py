import asyncio
import json
import socket
from datetime import datetime, timezone
from itertools import count
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fakeredis import FakeRedis as SyncFakeRedis
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from redis.asyncio.client import Pipeline
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import DailyTagSpendTransaction
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import DailySpendUpdateQueue
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.proxy_server import ProxyStartupEvent


def _daily_tag_spend_update_transactions() -> dict[str, DailyTagSpendTransaction]:
    return {
        "research_2025-01-15_api-key-123_gpt-4o_openai_/chat/completions": DailyTagSpendTransaction(
            tag="research",
            request_id="request-123",
            date="2025-01-15",
            api_key="api-key-123",
            model="gpt-4o",
            model_group=None,
            mcp_namespaced_tool_name=None,
            custom_llm_provider="openai",
            endpoint="/chat/completions",
            prompt_tokens=12,
            completion_tokens=7,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            compression_saved_tokens=0,
            compression_savings_spend=0.0,
            prompt_caching_savings_spend=0.0,
            gateway_injected_caching_savings_spend=0.0,
            spend=0.0042,
            api_requests=1,
            successful_requests=1,
            failed_requests=0,
        )
    }


@pytest.mark.asyncio
async def test_daily_tag_spend_restored_to_in_memory_queue_on_rpush_failure(
    redis_update_buffer: RedisUpdateBuffer, mock_redis_cache: AsyncMock
):
    mock_redis_cache.async_rpush = AsyncMock(side_effect=ConnectionError("redis went away"))
    queue: Final = DailySpendUpdateQueue()
    original_transactions: Final = _daily_tag_spend_update_transactions()
    await queue.add_update(original_transactions)

    await redis_update_buffer.store_in_memory_daily_tag_spend_updates_in_redis(queue)

    assert await queue.flush_and_get_aggregated_daily_spend_update_transactions() == original_transactions


@pytest.mark.asyncio
async def test_daily_tag_spend_restored_after_rpush_failure_is_pushed_once_on_next_tick(
    redis_update_buffer: RedisUpdateBuffer, mock_redis_cache: AsyncMock
):
    mock_redis_cache.async_rpush = AsyncMock(side_effect=ConnectionError("redis went away"))
    queue: Final = DailySpendUpdateQueue()
    original_transactions: Final = _daily_tag_spend_update_transactions()
    await queue.add_update(original_transactions)

    await redis_update_buffer.store_in_memory_daily_tag_spend_updates_in_redis(queue)
    mock_redis_cache.async_rpush = AsyncMock(return_value=1)

    await redis_update_buffer.store_in_memory_daily_tag_spend_updates_in_redis(queue)

    mock_redis_cache.async_rpush.assert_called_once()
    redis_key: Final = mock_redis_cache.async_rpush.call_args.kwargs["key"]
    values: Final = mock_redis_cache.async_rpush.call_args.kwargs["values"]
    assert redis_key == REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY
    assert json.loads(values[0]) == json.loads(safe_dumps(original_transactions))
    assert await queue.flush_and_get_aggregated_daily_spend_update_transactions() == {}


@pytest.fixture
def mock_redis_cache():
    """Create a mock RedisCache instance"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def redis_update_buffer(mock_redis_cache):
    """Create a RedisUpdateBuffer with a mock RedisCache"""
    return RedisUpdateBuffer(redis_cache=mock_redis_cache)


def _closed_port() -> int:
    with socket.socket() as unbound:
        unbound.bind(("127.0.0.1", 0))
        return unbound.getsockname()[1]


def _cache_with_fakeredis() -> tuple[RedisCache, FakeRedis]:
    server: Final = FakeServer()
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError("no running event loop")),
        patch(  # test-quality-ok: RedisCache.__init__ builds its sync client eagerly, with no injection point
            "litellm._redis.get_redis_client", return_value=SyncFakeRedis(server=server)
        ),
    ):
        redis_cache: Final = RedisCache(host="127.0.0.1", port=_closed_port(), namespace="litellm")
    redis_cache.redis_version = "7.0.0"
    redis_client: Final = FakeRedis(server=server)
    litellm.in_memory_llm_clients_cache.set_cache(key=redis_cache._get_async_client_cache_key(), value=redis_client)
    return redis_cache, redis_client


def _hang_after_set_lands(
    monkeypatch: pytest.MonkeyPatch, redis_client: FakeRedis, key_fragment: str, landed: asyncio.Event
) -> None:
    """Let the acknowledgement SET reach Redis, then block until the drain task is cancelled"""
    original_set: Final = redis_client.set

    async def set_then_hang(name: str, value: str, ex: int | None = None, xx: bool = False) -> object:
        result: Final = await original_set(name, value, ex=ex, xx=xx)
        if key_fragment not in name:
            return result
        landed.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr(redis_client, "set", set_then_hang)


def _lose_set_replies(monkeypatch: pytest.MonkeyPatch, redis_client: FakeRedis, key_fragment: str) -> None:
    original_set: Final = redis_client.set

    async def set_then_timeout(name: str, value: str, ex: int | None = None, xx: bool = False) -> object:
        result: Final = await original_set(name, value, ex=ex, xx=xx)
        if key_fragment in name:
            raise RedisTimeoutError("SET reply lost")
        return result

    monkeypatch.setattr(redis_client, "set", set_then_timeout)


def _timeout_first_pipeline_execute(monkeypatch: pytest.MonkeyPatch, redis_client: FakeRedis) -> None:
    original_pipeline: Final = redis_client.pipeline
    pipeline_calls: Final = count()

    def pipeline_with_timeout(transaction: bool = True) -> Pipeline:
        pipeline: Final = original_pipeline(transaction=transaction)
        if next(pipeline_calls) == 0:
            original_execute: Final = pipeline.execute

            async def execute_then_timeout(raise_on_error: bool = True) -> list[object]:
                await original_execute(raise_on_error=raise_on_error)
                raise RedisTimeoutError("simulated response timeout")

            monkeypatch.setattr(pipeline, "execute", execute_then_timeout)
        return pipeline

    monkeypatch.setattr(redis_client, "pipeline", pipeline_with_timeout)


def _hold_first_pipeline_until_release(
    monkeypatch: pytest.MonkeyPatch,
    redis_client: FakeRedis,
    move_executed: asyncio.Event,
    release_move: asyncio.Event,
) -> None:
    original_pipeline: Final = redis_client.pipeline
    pipeline_calls: Final = count()

    def pipeline_with_timeout(transaction: bool = True) -> Pipeline:
        pipeline: Final = original_pipeline(transaction=transaction)
        if next(pipeline_calls) == 0:
            original_execute: Final = pipeline.execute

            async def execute_after_release() -> list[object]:
                results: Final = await original_execute()
                move_executed.set()
                await release_move.wait()
                return results

            monkeypatch.setattr(pipeline, "execute", execute_after_release)
        return pipeline

    monkeypatch.setattr(redis_client, "pipeline", pipeline_with_timeout)


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_uses_pipeline(redis_update_buffer, mock_redis_cache):
    """
    Verify store_in_memory_spend_updates_in_redis calls async_rpush_pipeline once
    with the correct operations and skips empty queues.
    """
    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[3, 5, 2])

    # Create mock queues - only 3 of 6 have data
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
    daily_org_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value={})

    daily_end_user_queue = AsyncMock()
    daily_end_user_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value=None)

    daily_agent_queue = AsyncMock()
    daily_agent_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value={})

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=spend_update_queue,
        daily_spend_update_queue=daily_spend_queue,
        daily_team_spend_update_queue=daily_team_queue,
        daily_org_spend_update_queue=daily_org_queue,
        daily_end_user_spend_update_queue=daily_end_user_queue,
        daily_agent_spend_update_queue=daily_agent_queue,
    )

    # Should be called exactly once (pipeline)
    mock_redis_cache.async_rpush_pipeline.assert_called_once()

    # Verify only 3 operations were included (empty ones skipped)
    call_args = mock_redis_cache.async_rpush_pipeline.call_args
    rpush_list = call_args.kwargs["rpush_list"]
    assert len(rpush_list) == 3


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_restores_on_rpush_failure(redis_update_buffer, mock_redis_cache):
    """
    If async_rpush_pipeline raises, the already-drained transactions must be
    put back into the in-memory queues so the next scheduler tick retries.
    Without this, any transient Redis hiccup silently loses spend data.
    """
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
        DailySpendUpdateQueue,
    )
    from litellm.proxy.db.db_transaction_queue.spend_update_queue import (
        SpendUpdateQueue,
    )

    mock_redis_cache.async_rpush_pipeline = AsyncMock(side_effect=ConnectionError("redis went away"))

    spend_queue = SpendUpdateQueue()
    daily_user_queue = DailySpendUpdateQueue()
    daily_team_queue = DailySpendUpdateQueue()
    daily_org_queue = DailySpendUpdateQueue()
    daily_end_user_queue = DailySpendUpdateQueue()
    daily_agent_queue = DailySpendUpdateQueue()

    # Seed real queues with data so flush_and_get_aggregated returns it
    await spend_queue.add_update(
        {
            "entity_type": Litellm_EntityType.KEY,
            "entity_id": "key-abc",
            "response_cost": 1.5,
        }
    )
    await spend_queue.add_update(
        {
            "entity_type": Litellm_EntityType.TEAM,
            "entity_id": "team-xyz",
            "response_cost": 2.5,
        }
    )
    await daily_user_queue.add_update(
        {
            "user1_day_model": {
                "spend": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        }
    )

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=spend_queue,
        daily_spend_update_queue=daily_user_queue,
        daily_team_spend_update_queue=daily_team_queue,
        daily_org_spend_update_queue=daily_org_queue,
        daily_end_user_spend_update_queue=daily_end_user_queue,
        daily_agent_spend_update_queue=daily_agent_queue,
    )

    # After restore, the main spend queue should hold one item per
    # (entity_type, entity_id) pair with the aggregated cost
    restored_spend = await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert restored_spend["key_list_transactions"] == {"key-abc": 1.5}
    assert restored_spend["team_list_transactions"] == {"team-xyz": 2.5}

    # Daily user queue should hold the same aggregated dict
    restored_daily = await daily_user_queue.flush_and_get_aggregated_daily_spend_update_transactions()
    assert restored_daily == {
        "user1_day_model": {
            "spend": 1.0,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
    }


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_all_empty_returns_early(redis_update_buffer, mock_redis_cache):
    """
    When all queues are empty, pipeline should never be called.
    """
    mock_redis_cache.async_rpush_pipeline = AsyncMock()

    # All queues return empty
    empty_queue = AsyncMock()
    empty_queue.flush_and_get_aggregated_db_spend_update_transactions = AsyncMock(return_value={})
    empty_daily_queue = AsyncMock()
    empty_daily_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value={})

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=empty_queue,
        daily_spend_update_queue=empty_daily_queue,
        daily_team_spend_update_queue=empty_daily_queue,
        daily_org_spend_update_queue=empty_daily_queue,
        daily_end_user_spend_update_queue=empty_daily_queue,
        daily_agent_spend_update_queue=empty_daily_queue,
    )

    mock_redis_cache.async_rpush_pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_get_all_transactions_from_redis_buffer_pipeline(redis_update_buffer, mock_redis_cache):
    """
    Verify get_all_transactions_from_redis_buffer_pipeline correctly parses
    and aggregates results from each buffer drain.
    """
    # Simulate pipeline results: slot 0 = spend updates, slots 1-5 = daily categories,
    # slot 6 = budget window spend
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
    window_spend_json = json.dumps(
        [
            {
                "entity_type": "key",
                "entity_id": "hashed-token",
                "window_duration": "30d",
                "window_start": "2026-08-01T00:00:00.000000",
                "spend": 3.0,
                "started_at": None,
            }
        ]
    )

    mock_redis_cache.async_drain_list_to_processing = AsyncMock(
        side_effect=(
            [db_spend_json],
            [daily_user_json],
            [daily_team_json],
            [],
            [],
            [],
            [window_spend_json, window_spend_json],
        )
    )
    mock_redis_cache.async_ack_processing_list = AsyncMock()

    result = await redis_update_buffer.get_all_transactions_from_redis_buffer_pipeline()

    assert len(result) == 7
    (
        db_spend,
        daily_user,
        daily_team,
        daily_org,
        daily_end_user,
        daily_agent,
        window_spend,
    ) = result

    # Budget window spend from two pods is summed per window, not overwritten.
    assert window_spend is not None
    assert len(window_spend) == 1
    assert window_spend[0]["spend"] == 6.0
    assert window_spend[0]["entity_id"] == "hashed-token"

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

    from litellm.constants import (
        REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
        REDIS_UPDATE_BUFFER_KEY,
        REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY,
    )

    drained_keys: Final = tuple(
        call.kwargs["key"] for call in mock_redis_cache.async_drain_list_to_processing.await_args_list
    )
    assert drained_keys == (
        REDIS_UPDATE_BUFFER_KEY,
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
        REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
        REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY,
    )
    assert mock_redis_cache.async_ack_processing_list.await_count == 4


@pytest.mark.asyncio
async def test_org_member_spend_is_summed_across_pods_and_restored_on_rpush_failure(
    redis_update_buffer: RedisUpdateBuffer, mock_redis_cache: AsyncMock
):
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import (
        DailySpendUpdateQueue,
    )
    from litellm.proxy.db.db_transaction_queue.spend_update_queue import (
        SpendUpdateQueue,
    )

    member_key: Final = "organization_id::org-1::user_id::user-1"
    pod_json: Final = json.dumps({"org_member_list_transactions": {member_key: 0.25}})
    mock_redis_cache.async_drain_list_to_processing = AsyncMock(
        side_effect=([pod_json, pod_json], [], [], [], [], [], [])
    )

    (db_spend, *_rest) = await redis_update_buffer.get_all_transactions_from_redis_buffer_pipeline()

    assert db_spend is not None
    assert db_spend["org_member_list_transactions"] == {member_key: 0.5}

    mock_redis_cache.async_rpush_pipeline = AsyncMock(side_effect=ConnectionError("redis went away"))
    spend_queue: Final = SpendUpdateQueue()
    await spend_queue.add_update(
        {
            "entity_type": Litellm_EntityType.ORGANIZATION_MEMBER,
            "entity_id": member_key,
            "response_cost": 1.5,
        }
    )
    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=spend_queue,
        daily_spend_update_queue=DailySpendUpdateQueue(),
        daily_team_spend_update_queue=DailySpendUpdateQueue(),
        daily_org_spend_update_queue=DailySpendUpdateQueue(),
        daily_end_user_spend_update_queue=DailySpendUpdateQueue(),
        daily_agent_spend_update_queue=DailySpendUpdateQueue(),
    )

    restored_spend: Final = await spend_queue.flush_and_get_aggregated_db_spend_update_transactions()
    assert restored_spend["org_member_list_transactions"] == {member_key: 1.5}


@pytest.mark.asyncio
async def test_get_all_transactions_from_redis_buffer_pipeline_no_redis():
    """When redis_cache is None, should return all Nones"""
    buffer = RedisUpdateBuffer(redis_cache=None)
    result = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert result == (None, None, None, None, None, None, None)


@pytest.mark.asyncio
async def test_restore_transactions_to_redis_pushes_only_provided(redis_update_buffer, mock_redis_cache):
    """
    restore_transactions_to_redis re-pushes only the transaction sets it was
    given, to their matching buffer keys, so uncommitted spend can be retried.
    """
    from litellm.constants import (
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_UPDATE_BUFFER_KEY,
    )

    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[1, 1])

    db_spend = {"key_list_transactions": {"key1": 1.0}}
    daily_user = {"user_key1": {"spend": 1.0}}

    await redis_update_buffer.restore_transactions_to_redis(
        db_spend_update_transactions=db_spend,
        daily_spend_update_transactions=daily_user,
    )

    mock_redis_cache.async_rpush_pipeline.assert_called_once()
    rpush_list = mock_redis_cache.async_rpush_pipeline.call_args.kwargs["rpush_list"]
    pushed_keys = {op["key"] for op in rpush_list}
    assert pushed_keys == {
        REDIS_UPDATE_BUFFER_KEY,
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
    }
    # Payloads round-trip through the same JSON encoding used on the store path
    payloads = {op["key"]: json.loads(op["values"][0]) for op in rpush_list}
    assert payloads[REDIS_UPDATE_BUFFER_KEY] == db_spend
    assert payloads[REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY] == daily_user


@pytest.mark.asyncio
async def test_restored_window_spend_transactions_drain_back_unchanged(redis_update_buffer, mock_redis_cache):
    """A window commit that fails after a drain must be re-pushed
    in the store path's encoding, so the next drain returns the same increments."""
    from litellm.constants import REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY
    from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
        build_window_spend_transaction,
    )

    window_transactions = (
        build_window_spend_transaction(
            entity_type="key",
            entity_id="hashed-token",
            window_duration="30d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=3.0,
            started_at=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
        ),
    )
    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[1])

    await redis_update_buffer.restore_transactions_to_redis(window_spend_update_transactions=window_transactions)

    rpush_list = mock_redis_cache.async_rpush_pipeline.call_args.kwargs["rpush_list"]
    assert [op["key"] for op in rpush_list] == [REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY]

    mock_redis_cache.async_drain_list_to_processing = AsyncMock(
        side_effect=([], [], [], [], [], [], list(rpush_list[0]["values"]))
    )
    drained = await redis_update_buffer.get_all_transactions_from_redis_buffer_pipeline()

    assert drained[6] == window_transactions


@pytest.mark.asyncio
async def test_restore_transactions_to_redis_noop_when_empty(redis_update_buffer, mock_redis_cache):
    """Nothing to restore -> no Redis call."""
    mock_redis_cache.async_rpush_pipeline = AsyncMock()
    await redis_update_buffer.restore_transactions_to_redis()
    mock_redis_cache.async_rpush_pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_restore_transactions_to_redis_swallows_redis_error(redis_update_buffer, mock_redis_cache):
    """A Redis failure during restore must not propagate to the caller's finally block."""
    from redis.exceptions import RedisError

    mock_redis_cache.async_rpush_pipeline = AsyncMock(side_effect=RedisError("redis down"))

    await redis_update_buffer.restore_transactions_to_redis(
        db_spend_update_transactions={"key_list_transactions": {"key1": 1.0}},
    )

    mock_redis_cache.async_rpush_pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_drain_recovers_spend_after_move_reply_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import Litellm_EntityType
    from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import DailySpendUpdateQueue
    from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue

    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    spend_queue: Final = SpendUpdateQueue()
    await spend_queue.add_update({"entity_type": Litellm_EntityType.KEY, "entity_id": "key-abc", "response_cost": 0.25})
    await spend_queue.add_update({"entity_type": Litellm_EntityType.KEY, "entity_id": "key-abc", "response_cost": 0.5})
    await spend_queue.add_update(
        {"entity_type": Litellm_EntityType.TEAM, "entity_id": "team-xyz", "response_cost": 1.25}
    )
    await buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=spend_queue,
        daily_spend_update_queue=DailySpendUpdateQueue(),
        daily_team_spend_update_queue=DailySpendUpdateQueue(),
        daily_org_spend_update_queue=DailySpendUpdateQueue(),
        daily_end_user_spend_update_queue=DailySpendUpdateQueue(),
        daily_agent_spend_update_queue=DailySpendUpdateQueue(),
    )

    _timeout_first_pipeline_execute(monkeypatch, redis_client)
    with pytest.raises(RedisTimeoutError, match="simulated response timeout"):
        await buffer.get_all_transactions_from_redis_buffer_pipeline()

    recovered: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert recovered[0] is not None
    assert recovered[0]["key_list_transactions"] == {"key-abc": 0.75}
    assert recovered[0]["team_list_transactions"] == {"team-xyz": 1.25}
    assert recovered[1:] == (None, None, None, None, None, None)
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None, None, None, None, None, None, None)


@pytest.mark.asyncio
async def test_pipeline_drain_keeps_all_buffers_pending_after_partial_move_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.constants import (
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_UPDATE_BUFFER_KEY,
    )

    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    db_spend: Final = json.dumps({"key_list_transactions": {"key-partial": 1.25}})
    daily_spend: Final = json.dumps({"user-partial": {"spend": 2.5, "api_requests": 1}})
    await redis_cache.async_rpush(key=REDIS_UPDATE_BUFFER_KEY, values=(db_spend,))
    await redis_cache.async_rpush(key=REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY, values=(daily_spend,))

    _timeout_first_pipeline_execute(monkeypatch, redis_client)
    with pytest.raises(RedisTimeoutError, match="simulated response timeout"):
        await buffer.get_all_transactions_from_redis_buffer_pipeline()

    recovered: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert recovered[0] is not None
    assert recovered[0]["key_list_transactions"] == {"key-partial": 1.25}
    assert recovered[1] == {"user-partial": {"spend": 2.5, "api_requests": 1}}
    assert recovered[2:] == (None, None, None, None, None)
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None, None, None, None, None, None, None)


@pytest.mark.asyncio
async def test_pipeline_drain_returns_only_keys_with_successful_acknowledgements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.constants import (
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_UPDATE_BUFFER_KEY,
    )

    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    db_spend: Final = json.dumps({"key_list_transactions": {"key-ack": 1.5}})
    daily_spend: Final = json.dumps({"user-ack": {"spend": 2.75, "api_requests": 1}})
    await redis_cache.async_rpush(key=REDIS_UPDATE_BUFFER_KEY, values=(db_spend,))
    await redis_cache.async_rpush(key=REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY, values=(daily_spend,))

    original_set: Final = redis_client.set
    _lose_set_replies(monkeypatch, redis_client, f"{REDIS_UPDATE_BUFFER_KEY}:processing")
    first_result: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert first_result[0] is None
    assert first_result[1] == {"user-ack": {"spend": 2.75, "api_requests": 1}}
    assert first_result[2:] == (None, None, None, None, None)

    monkeypatch.setattr(redis_client, "set", original_set)
    second_result: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert second_result[0] is not None
    assert second_result[0]["key_list_transactions"] == {"key-ack": 1.5}
    assert second_result[1:] == (None, None, None, None, None, None)
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None, None, None, None, None, None, None)


@pytest.mark.asyncio
async def test_spend_logs_are_recovered_once_after_move_reply_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
    rows: Final = ({"request_id": "req-a"}, {"request_id": "req-b"})
    assert await buffer.store_spend_logs_in_redis(rows) is True

    _timeout_first_pipeline_execute(monkeypatch, redis_client)
    with pytest.raises(RedisTimeoutError, match="simulated response timeout"):
        await buffer.get_spend_logs_from_redis_buffer(limit=10)

    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == rows
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_forked_workers_do_not_take_over_each_others_pending_spend_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    process_id: Final = MagicMock(return_value=100)
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache, process_id=process_id)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
    parent_rows: Final = ({"request_id": "req-parent"},)
    assert await buffer.store_spend_logs_in_redis(parent_rows) is True

    _timeout_first_pipeline_execute(monkeypatch, redis_client)
    with pytest.raises(RedisTimeoutError, match="simulated response timeout"):
        await buffer.get_spend_logs_from_redis_buffer(limit=10)

    process_id.return_value = 200
    child_rows: Final = ({"request_id": "req-child"},)
    assert await buffer.store_spend_logs_in_redis(child_rows) is True
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == child_rows

    process_id.return_value = 100
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == parent_rows
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_overlapping_spend_log_drains_serialize_and_recover_rows_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
    rows: Final = ({"request_id": "req-a"}, {"request_id": "req-b"})
    assert await buffer.store_spend_logs_in_redis(rows) is True

    move_executed: Final = asyncio.Event()
    release_move: Final = asyncio.Event()
    _hold_first_pipeline_until_release(
        monkeypatch,
        redis_client,
        move_executed,
        release_move,
    )

    first_call: Final = asyncio.create_task(buffer.get_spend_logs_from_redis_buffer(limit=10))
    await move_executed.wait()

    second_started: Final = asyncio.Event()

    async def start_second_drain() -> tuple[dict[str, object], ...]:
        second_started.set()
        return await buffer.get_spend_logs_from_redis_buffer(limit=10)

    second_call: Final = asyncio.create_task(start_second_drain())
    await second_started.wait()

    release_move.set()
    first_result: Final = await first_call
    second_result: Final = await second_call
    request_ids: Final = tuple(str(row["request_id"]) for row in first_result + second_result)
    assert sorted(request_ids) == ["req-a", "req-b"]
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_ack_failure_leaves_items_for_the_next_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    payload: Final = json.dumps(
        {
            "key_list_transactions": {"key-ack": 1.5},
            "user_list_transactions": {},
            "end_user_list_transactions": {},
            "team_list_transactions": {},
            "team_member_list_transactions": {},
            "org_list_transactions": {},
            "tag_list_transactions": {},
        }
    )
    await redis_cache.async_rpush(key="litellm_spend_update_buffer", values=(payload,))

    original_set: Final = redis_client.set
    failed_set: Final = AsyncMock(side_effect=RedisTimeoutError("SET failed"))
    monkeypatch.setattr(redis_client, "set", failed_set)
    with pytest.raises(RedisTimeoutError, match="SET failed"):
        await buffer.get_all_update_transactions_from_redis_buffer()

    monkeypatch.setattr(redis_client, "set", original_set)
    recovered: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert recovered is not None
    assert recovered["key_list_transactions"] == {"key-ack": 1.5}
    assert await buffer.get_all_update_transactions_from_redis_buffer() is None


@pytest.mark.asyncio
async def test_lost_ack_reply_still_hands_the_drained_rows_to_the_next_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    payload: Final = json.dumps({"key_list_transactions": {"key-lost-ack": 2.5}})
    await redis_cache.async_rpush(key="litellm_spend_update_buffer", values=(payload,))

    original_set: Final = redis_client.set
    _lose_set_replies(monkeypatch, redis_client, "litellm_spend_update_buffer")
    with pytest.raises(RedisTimeoutError, match="SET reply lost"):
        await buffer.get_all_update_transactions_from_redis_buffer()

    monkeypatch.setattr(redis_client, "set", original_set)
    recovered: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert recovered is not None
    assert recovered["key_list_transactions"] == {"key-lost-ack": 2.5}
    assert await buffer.get_all_update_transactions_from_redis_buffer() is None


@pytest.mark.asyncio
async def test_rows_held_after_a_lost_ack_reply_survive_a_pipeline_drain_that_fails_on_a_sibling_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.constants import (
        REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        REDIS_UPDATE_BUFFER_KEY,
    )

    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    db_spend: Final = json.dumps({"key_list_transactions": {"key-held": 1.5}})
    await redis_cache.async_rpush(key=REDIS_UPDATE_BUFFER_KEY, values=(db_spend,))

    original_set: Final = redis_client.set
    _lose_set_replies(monkeypatch, redis_client, f"{REDIS_UPDATE_BUFFER_KEY}:processing")
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None,) * 7
    monkeypatch.setattr(redis_client, "set", original_set)

    original_type: Final = redis_client.type

    async def type_fails_for_sibling(name: str) -> object:
        if f"{REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY}:processing" in name:
            raise RedisTimeoutError("sibling drain failed")
        return await original_type(name)

    monkeypatch.setattr(redis_client, "type", type_fails_for_sibling)
    with pytest.raises(RedisTimeoutError, match="sibling drain failed"):
        await buffer.get_all_transactions_from_redis_buffer_pipeline()
    monkeypatch.setattr(redis_client, "type", original_type)

    recovered: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert recovered[0] is not None
    assert recovered[0]["key_list_transactions"] == {"key-held": 1.5}
    assert recovered[1:] == (None,) * 6
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None,) * 7


def test_validate_redis_transaction_buffer_raises_without_redis():
    """
    When use_redis_transaction_buffer=true but no Redis cache is configured,
    the proxy should refuse to start with a clear error message.
    """
    with pytest.raises(ValueError, match="use_redis_transaction_buffer"):
        ProxyStartupEvent._validate_redis_transaction_buffer_config(
            general_settings={"use_redis_transaction_buffer": True},
            redis_usage_cache=None,
        )


def test_validate_redis_transaction_buffer_passes_with_redis():
    """
    When use_redis_transaction_buffer=true and Redis cache is configured,
    validation should pass without error.
    """
    # Should not raise
    ProxyStartupEvent._validate_redis_transaction_buffer_config(
        general_settings={"use_redis_transaction_buffer": True},
        redis_usage_cache=MagicMock(),
    )


def test_validate_redis_transaction_buffer_passes_when_disabled():
    """
    When use_redis_transaction_buffer is not set or false,
    validation should pass regardless of Redis configuration.
    """
    # Should not raise even without Redis
    ProxyStartupEvent._validate_redis_transaction_buffer_config(
        general_settings={},
        redis_usage_cache=None,
    )


def test_get_transaction_buffer_redis_cache_builds_from_env(monkeypatch):
    """
    When use_redis_transaction_buffer=true, a standalone RedisCache is built from
    REDIS_* environment variables so the buffer works without a Redis cache backend.
    """
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")

    with patch("litellm.proxy.proxy_server.RedisCache") as mock_redis_cache:
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )

    mock_redis_cache.assert_called_once()
    assert mock_redis_cache.call_args.kwargs["host"] == "localhost"
    assert result is mock_redis_cache.return_value


def test_get_transaction_buffer_redis_cache_none_when_disabled():
    """When use_redis_transaction_buffer is not enabled, no standalone cache is built."""
    result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
        general_settings={},
    )
    assert result is None


def test_get_transaction_buffer_redis_cache_none_without_redis_env():
    """
    When use_redis_transaction_buffer=true but no REDIS_* env vars are set,
    no standalone cache is built (startup validation then raises the config error).
    """
    with patch("litellm._redis._redis_kwargs_from_environment", return_value={}):
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )
    assert result is None


def test_get_transaction_buffer_redis_cache_none_without_host_or_url():
    """
    A REDIS_* var that is not a connection target (e.g. REDIS_SOCKET_TIMEOUT) must not
    trigger a build. Without a host or url, get_redis_client raises, so return None and
    let startup validation surface the config error instead of crashing.
    """
    with patch(
        "litellm._redis._redis_kwargs_from_environment",
        return_value={"socket_timeout": 5.0},
    ):
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": True},
        )
    assert result is None


def test_get_transaction_buffer_redis_cache_parses_string_flag(monkeypatch):
    """
    use_redis_transaction_buffer accepts a string value (e.g. from env/YAML); "true"
    is parsed to a bool before the standalone cache is built.
    """
    monkeypatch.setenv("REDIS_HOST", "localhost")

    with patch("litellm.proxy.proxy_server.RedisCache") as mock_redis_cache:
        result = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings={"use_redis_transaction_buffer": "true"},
        )

    mock_redis_cache.assert_called_once()
    assert result is mock_redis_cache.return_value


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_pushes_budget_window_spend(redis_update_buffer, mock_redis_cache):
    """The budget window queue has to ride the same rpush as the daily queues,
    otherwise multi-pod deployments never persist per-window spend."""
    from datetime import datetime, timezone

    from litellm.constants import REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY
    from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
        WindowSpendUpdateQueue,
        build_window_spend_transaction,
    )

    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[1])

    empty_queue = AsyncMock()
    empty_queue.flush_and_get_aggregated_db_spend_update_transactions = AsyncMock(return_value={})
    empty_daily_queue = AsyncMock()
    empty_daily_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value={})

    window_queue = WindowSpendUpdateQueue()
    await window_queue.add_update(
        build_window_spend_transaction(
            entity_type="key",
            entity_id="hashed-token",
            window_duration="30d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=1.25,
            started_at=datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
    )

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=empty_queue,
        daily_spend_update_queue=empty_daily_queue,
        daily_team_spend_update_queue=empty_daily_queue,
        daily_org_spend_update_queue=empty_daily_queue,
        daily_end_user_spend_update_queue=empty_daily_queue,
        daily_agent_spend_update_queue=empty_daily_queue,
        window_spend_update_queue=window_queue,
    )

    rpush_list = mock_redis_cache.async_rpush_pipeline.call_args.kwargs["rpush_list"]
    assert len(rpush_list) == 1
    assert rpush_list[0]["key"] == REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY
    pushed = json.loads(rpush_list[0]["values"][0])
    assert pushed == [
        {
            "entity_type": "key",
            "entity_id": "hashed-token",
            "window_duration": "30d",
            "window_start": "2026-08-01T00:00:00.000000",
            "spend": 1.25,
            "started_at": "2026-08-10T12:00:00.000000",
            "request_ids": [],
        }
    ]


@pytest.mark.asyncio
async def test_budget_window_payloads_keep_request_ids_for_older_workers(redis_update_buffer, mock_redis_cache):
    """A leader from before the field was dropped indexes request_ids while
    merging what it drained, and the drain removes the payload, so one without
    the key would cost a rolling deploy those increments."""
    from datetime import datetime, timezone

    from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
        WindowSpendUpdateQueue,
        build_window_spend_transaction,
    )

    mock_redis_cache.async_rpush_pipeline = AsyncMock(return_value=[1])
    window_queue = WindowSpendUpdateQueue()
    await window_queue.add_update(
        build_window_spend_transaction(
            entity_type="key",
            entity_id="hashed-token",
            window_duration="30d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=1.25,
        )
    )

    await redis_update_buffer.restore_transactions_to_redis(
        window_spend_update_transactions=await window_queue.flush_and_get_aggregated_window_spend_transactions(),
    )

    rpush_list = mock_redis_cache.async_rpush_pipeline.call_args.kwargs["rpush_list"]
    restored = json.loads(rpush_list[0]["values"][0])
    assert [payload["request_ids"] for payload in restored] == [[]]


@pytest.mark.asyncio
async def test_store_in_memory_spend_updates_restores_budget_window_spend_on_rpush_failure(
    redis_update_buffer, mock_redis_cache
):
    """The window queue is drained before the rpush, so a Redis hiccup would
    silently drop per-window spend without the restore."""
    from datetime import datetime, timezone

    from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
        WindowSpendUpdateQueue,
        build_window_spend_transaction,
    )

    mock_redis_cache.async_rpush_pipeline = AsyncMock(side_effect=ConnectionError("redis went away"))

    empty_queue = AsyncMock()
    empty_queue.flush_and_get_aggregated_db_spend_update_transactions = AsyncMock(return_value={})
    empty_daily_queue = AsyncMock()
    empty_daily_queue.flush_and_get_aggregated_daily_spend_update_transactions = AsyncMock(return_value={})

    window_queue = WindowSpendUpdateQueue()
    await window_queue.add_update(
        build_window_spend_transaction(
            entity_type="team",
            entity_id="team-1",
            window_duration="7d",
            window_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
            spend=4.0,
        )
    )

    await redis_update_buffer.store_in_memory_spend_updates_in_redis(
        spend_update_queue=empty_queue,
        daily_spend_update_queue=empty_daily_queue,
        daily_team_spend_update_queue=empty_daily_queue,
        daily_org_spend_update_queue=empty_daily_queue,
        daily_end_user_spend_update_queue=empty_daily_queue,
        daily_agent_spend_update_queue=empty_daily_queue,
        window_spend_update_queue=window_queue,
    )

    restored = await window_queue.flush_and_get_aggregated_window_spend_transactions()
    assert [payload["spend"] for payload in restored] == [4.0]
    assert [payload["entity_id"] for payload in restored] == ["team-1"]


class _ListRedis:
    def __init__(self) -> None:
        self.rows: list[str] = []
        self.processing: list[str] = []

    async def async_rpush_and_trim(self, key: str, values: list[str], max_len: int) -> int:
        self.rows.extend(values)
        pushed_len = len(self.rows)
        del self.rows[:-max_len]
        return pushed_len

    async def async_drain_list_to_processing(
        self, key: str, worker_id: str, count: int, ttl_seconds: int, recovery_seconds: int
    ) -> list[str]:
        if self.processing:
            return self.processing
        self.processing = self.rows[:count]
        self.rows = self.rows[count:]
        return self.processing

    async def async_ack_processing_list(self, key: str, worker_id: str, ttl_seconds: int) -> bool:
        self.processing = []
        return True


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_drops_oldest_rows_past_the_cap():
    redis = _ListRedis()
    buffer = RedisUpdateBuffer(redis_cache=redis)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "old"}, {"request_id": "mid"}], max_rows=2) is True
    assert await buffer.store_spend_logs_in_redis([{"request_id": "new"}], max_rows=2) is True

    parked = await buffer.get_spend_logs_from_redis_buffer(limit=10)
    assert [row["request_id"] for row in parked] == ["mid", "new"]
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_reports_failure_without_redis():
    buffer = RedisUpdateBuffer(redis_cache=None)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "a"}]) is False
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()


@pytest.mark.asyncio
async def test_store_spend_logs_in_redis_is_off_unless_transaction_buffering_is_enabled():
    redis = _ListRedis()
    buffer = RedisUpdateBuffer(redis_cache=redis)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=False)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "a"}]) is False
    assert redis.rows == []


@pytest.mark.asyncio
async def test_rows_whose_acknowledgement_was_cancelled_after_landing_reach_the_next_pipeline_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    payload: Final = json.dumps({"key_list_transactions": {"key-cancelled-ack": 4.5}})
    await redis_cache.async_rpush(key="litellm_spend_update_buffer", values=(payload,))
    original_set: Final = redis_client.set
    landed: Final = asyncio.Event()
    _hang_after_set_lands(monkeypatch, redis_client, "litellm_spend_update_buffer", landed)

    shutdown_drain: Final = asyncio.create_task(buffer.get_all_update_transactions_from_redis_buffer())
    await landed.wait()
    shutdown_drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_drain

    monkeypatch.setattr(redis_client, "set", original_set)
    recovered: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert recovered is not None
    assert recovered["key_list_transactions"] == {"key-cancelled-ack": 4.5}
    assert await buffer.get_all_update_transactions_from_redis_buffer() is None


@pytest.mark.asyncio
async def test_spend_logs_whose_acknowledgement_was_cancelled_after_landing_reach_the_next_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_cache, redis_client = _cache_with_fakeredis()
    buffer: Final = RedisUpdateBuffer(redis_cache=redis_cache)
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
    rows: Final = ({"request_id": "req-cancelled"},)
    assert await buffer.store_spend_logs_in_redis(rows) is True
    original_set: Final = redis_client.set
    landed: Final = asyncio.Event()
    _hang_after_set_lands(monkeypatch, redis_client, "litellm_spend_logs_buffer", landed)

    shutdown_drain: Final = asyncio.create_task(buffer.get_spend_logs_from_redis_buffer(limit=10))
    await landed.wait()
    shutdown_drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_drain

    monkeypatch.setattr(redis_client, "set", original_set)
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == rows
    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ()
