import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.proxy_server import ProxyStartupEvent


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
    and aggregates results from async_lpop_pipeline.
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

    mock_redis_cache.async_lpop_pipeline = AsyncMock(
        return_value=[
            [db_spend_json],  # slot 0: db spend updates
            [daily_user_json],  # slot 1: daily user
            [daily_team_json],  # slot 2: daily team
            None,  # slot 3: daily org (empty)
            None,  # slot 4: daily end-user (empty)
            None,  # slot 5: daily agent (empty)
            [window_spend_json, window_spend_json],  # slot 6: budget window spend
        ]
    )

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

    # Verify pipeline was called once with correct keys
    mock_redis_cache.async_lpop_pipeline.assert_called_once()
    from litellm.constants import REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY

    popped_keys = [op["key"] for op in mock_redis_cache.async_lpop_pipeline.call_args.kwargs["lpop_list"]]
    assert popped_keys[6] == REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY


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
    """A window commit that fails after the destructive lpop must be re-pushed
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

    mock_redis_cache.async_lpop_pipeline = AsyncMock(
        return_value=[None, None, None, None, None, None, list(rpush_list[0]["values"])]
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
    merging what it popped, and the pop is destructive, so a payload without
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
