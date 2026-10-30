import json
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
import pytest_asyncio
from redis.exceptions import RedisError
from fakeredis.aioredis import FakeConnection

from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY, REDIS_UPDATE_BUFFER_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import DailyTagSpendTransaction, DBSpendUpdateTransactions, Litellm_EntityType
from litellm.proxy.db.db_transaction_queue.daily_spend_update_queue import DailySpendUpdateQueue
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.spend_update_queue import SpendUpdateQueue
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import (
    WindowSpendUpdateQueue,
    build_window_spend_transaction,
)
from litellm.proxy.proxy_server import ProxyStartupEvent
from tests._support.fake_redis import fake_redis_cache


class _LostPushReply(FakeConnection):
    async def read_response(self, **kwargs: object) -> object:
        reply: Final[object] = await super().read_response(**kwargs)
        if isinstance(reply, int):
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("push applied but reply lost")
        return reply


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


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest_asyncio.fixture(loop_scope="function")
async def cache(server: fakeredis.FakeServer) -> RedisCache:
    return fake_redis_cache(server, namespace="buffer")


@pytest.fixture
def buffer(cache: RedisCache) -> RedisUpdateBuffer:
    return RedisUpdateBuffer(redis_cache=cache)


def _entities() -> DBSpendUpdateTransactions:
    return DBSpendUpdateTransactions(
        user_list_transactions={"使用者": 1.5},
        end_user_list_transactions={"end": 2.0},
        key_list_transactions={"key": 3.0},
        team_list_transactions={"team": 4.0},
        team_member_list_transactions={"team::user": 5.0},
        org_list_transactions={"org": 6.0},
        org_member_list_transactions={"org::user": 7.0},
        project_list_transactions={"project": 8.0},
        tag_list_transactions={"tag": 9.0},
        agent_list_transactions={"agent": 10.0},
        model_access_group_list_transactions={"group": 11.0},
    )


async def _fill_spend(spend: SpendUpdateQueue) -> None:
    kinds: Final = (
        Litellm_EntityType.USER,
        Litellm_EntityType.END_USER,
        Litellm_EntityType.KEY,
        Litellm_EntityType.TEAM,
        Litellm_EntityType.TEAM_MEMBER,
        Litellm_EntityType.ORGANIZATION,
        Litellm_EntityType.ORGANIZATION_MEMBER,
        Litellm_EntityType.PROJECT,
        Litellm_EntityType.TAG,
        Litellm_EntityType.AGENT,
        Litellm_EntityType.MODEL_ACCESS_GROUP,
    )
    for kind, values in zip(kinds, _entities().values()):
        for entity, amount in values.items():
            await spend.add_update({"entity_type": kind, "entity_id": entity, "response_cost": amount})


async def _flush(buffer: RedisUpdateBuffer, spend: SpendUpdateQueue, window: WindowSpendUpdateQueue) -> None:
    await buffer.store_in_memory_spend_updates_in_redis(spend, *(DailySpendUpdateQueue() for _ in range(5)), window)


@pytest.mark.asyncio
async def test_entity_wire_works_in_both_directions_with_namespace_and_no_ttl(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    payload: Final = _entities()
    await buffer.restore_transactions_to_redis(db_spend_update_transactions=payload)
    wire: Final = await client.lpop("buffer:litellm_spend_update_buffer")
    assert wire == safe_dumps(payload).encode()
    assert json.loads(wire) == payload
    await client.rpush("buffer:litellm_spend_update_buffer", safe_dumps(payload), safe_dumps(payload))
    assert await client.pttl("buffer:litellm_spend_update_buffer") == -1
    combined: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert combined == {
        field: {entity: 2 * spend for entity, spend in values.items()} for field, values in payload.items()
    }
    assert await client.llen("buffer:litellm_spend_update_buffer") == 0
    assert await client.exists(REDIS_UPDATE_BUFFER_KEY) == 0


@pytest.mark.asyncio
async def test_legacy_partial_null_and_nonfinite_entity_fields(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    await client.rpush(
        "buffer:litellm_spend_update_buffer",
        '{"user_list_transactions": {"old": NaN}, "org_list_transactions": null}',
        '{"user_list_transactions": {"old": 2.5}, "unknown": {"ignored": 1}}',
    )
    result: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert result == {**{field: {} for field in _entities()}, "user_list_transactions": {"old": 2.5}}


@pytest.mark.asyncio
async def test_entity_pop_is_bounded_and_malformed_entries_stay_consumed(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    await client.rpush(
        "buffer:litellm_spend_update_buffer", *('{"user_list_transactions": {"a": 1}}' for _ in range(101))
    )
    first: Final = await buffer.get_all_update_transactions_from_redis_buffer()
    assert first is not None and first["user_list_transactions"] == {"a": 100}
    assert await client.llen("buffer:litellm_spend_update_buffer") == 1
    await client.rpush("buffer:litellm_spend_update_buffer", "invalid")
    with pytest.raises(ValueError, match="invalid spend update buffer"):
        await buffer.get_all_update_transactions_from_redis_buffer()
    assert await client.llen("buffer:litellm_spend_update_buffer") == 0


@pytest.mark.asyncio
async def test_shared_pipeline_flushes_entities_and_windows_without_tag(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    spend: Final = SpendUpdateQueue()
    await _fill_spend(spend)
    window: Final = WindowSpendUpdateQueue()
    transaction: Final = build_window_spend_transaction(
        entity_type="key",
        entity_id="key",
        window_duration="1d",
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        spend=1.25,
        started_at=None,
    )
    await window.add_update(transaction)
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    await client.rpush(
        "buffer:" + REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY, safe_dumps(_daily_tag_spend_update_transactions())
    )
    await _flush(buffer, spend, window)
    result: Final = await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert result[0] == _entities()
    assert result[1:6] == (None, None, None, None, None)
    assert result[6] == (transaction,)
    assert await client.llen("buffer:" + REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY) == 1
    assert await spend.flush_and_get_aggregated_db_spend_update_transactions() == {field: {} for field in _entities()}


@pytest.mark.asyncio
async def test_partial_push_failure_restores_all_sources_even_when_entity_push_applied(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    await client.set("buffer:litellm_window_spend_update_buffer", "wrongtype")
    spend: Final = SpendUpdateQueue()
    await _fill_spend(spend)
    window: Final = WindowSpendUpdateQueue()
    transaction: Final = build_window_spend_transaction(
        entity_type="key",
        entity_id="key",
        window_duration="1d",
        window_start=datetime(2026, 1, 1),
        spend=1.25,
        started_at=None,
    )
    await window.add_update(transaction)
    await _flush(buffer, spend, window)
    assert await spend.flush_and_get_aggregated_db_spend_update_transactions() == _entities()
    assert await window.flush_and_get_aggregated_window_spend_transactions() == (transaction,)
    assert await client.llen("buffer:litellm_spend_update_buffer") == 1
    assert await client.get("buffer:litellm_window_spend_update_buffer") == b"wrongtype"


@pytest.mark.asyncio
async def test_partial_pop_failure_does_not_recreate_consumed_entity_entries(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    await buffer.restore_transactions_to_redis(db_spend_update_transactions=_entities())
    await client.set("buffer:litellm_daily_spend_update_buffer", "wrongtype")
    with pytest.raises(RedisError):
        await buffer.get_all_transactions_from_redis_buffer_pipeline()
    assert await client.llen("buffer:litellm_spend_update_buffer") == 0


@pytest.mark.asyncio
async def test_outage_restores_source_and_breaker_prevents_a_second_drain(
    buffer: RedisUpdateBuffer, server: fakeredis.FakeServer
) -> None:
    spend: Final = SpendUpdateQueue()
    await _fill_spend(spend)
    server.connected = False
    await _flush(buffer, spend, WindowSpendUpdateQueue())
    assert await spend.flush_and_get_aggregated_db_spend_update_transactions() == _entities()
    with pytest.raises(RedisError):
        await buffer.get_all_update_transactions_from_redis_buffer()
    await buffer.restore_transactions_to_redis(db_spend_update_transactions=_entities())


@pytest.mark.asyncio
async def test_lost_push_ack_restores_sources_without_removing_the_applied_entry(
    server: fakeredis.FakeServer,
) -> None:
    cache: Final = fake_redis_cache(server, namespace="buffer")
    cache.init_async_client().connection_pool.connection_class = _LostPushReply
    buffer: Final = RedisUpdateBuffer(redis_cache=cache)
    spend: Final = SpendUpdateQueue()
    await _fill_spend(spend)
    await _flush(buffer, spend, WindowSpendUpdateQueue())
    assert await spend.flush_and_get_aggregated_db_spend_update_transactions() == _entities()
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await client.lrange("buffer:litellm_spend_update_buffer", 0, -1) == [safe_dumps(_entities()).encode()]


@pytest.mark.asyncio
async def test_buffer_without_redis_does_not_drain_local_sources() -> None:
    buffer: Final = RedisUpdateBuffer(redis_cache=None)
    spend: Final = SpendUpdateQueue()
    await _fill_spend(spend)
    await _flush(buffer, spend, WindowSpendUpdateQueue())
    assert await spend.flush_and_get_aggregated_db_spend_update_transactions() == _entities()
    assert await buffer.get_all_transactions_from_redis_buffer_pipeline() == (None,) * 7
    assert await buffer.get_all_update_transactions_from_redis_buffer() is None
    await buffer.restore_transactions_to_redis(db_spend_update_transactions=_entities())


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


PARKED_DAILY_KEY: Final = "user-1_2025-01-15_api-key-123_gpt-4o_openai_/chat/completions"


def _parked_daily_batch(spend: float) -> str:
    return json.dumps(
        {
            PARKED_DAILY_KEY: {
                "spend": spend,
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "api_requests": 1,
                "successful_requests": 1,
                "failed_requests": 0,
            }
        },
        allow_nan=True,
    )


class _ListRedis:
    def __init__(self) -> None:
        self.rows: list[str] = []

    async def async_rpush(self, key: str, values: list[str]) -> int:
        self.rows.extend(values)
        return len(self.rows)

    async def async_rpush_and_trim(self, key: str, values: list[str], max_len: int) -> int:
        self.rows.extend(values)
        pushed_len = len(self.rows)
        del self.rows[:-max_len]
        return pushed_len

    async def async_lpop(self, key: str, count: int | None = None, **kwargs: object) -> list[str] | None:
        if not self.rows:
            return None
        popped = self.rows[:count]
        del self.rows[:count]
        return popped


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
@pytest.mark.parametrize("spend", [float("nan"), float("inf")])
async def test_a_non_finite_daily_spend_buffered_in_redis_drains_as_zero(spend: float):
    buffer: Final = RedisUpdateBuffer(redis_cache=_ListRedis())
    queue: Final = DailySpendUpdateQueue()
    await queue.add_update(json.loads(_parked_daily_batch(spend)))
    await queue.add_update(json.loads(_parked_daily_batch(0.0042)))

    await buffer.store_in_memory_daily_tag_spend_updates_in_redis(queue)
    drained: Final = await buffer.get_all_daily_tag_spend_update_transactions_from_redis_buffer()

    assert drained is not None
    assert drained[PARKED_DAILY_KEY]["spend"] == 0.0
    assert drained[PARKED_DAILY_KEY]["api_requests"] == 2


@pytest.mark.asyncio
async def test_a_spend_log_row_parked_with_a_nan_spend_is_read_back_with_zero_spend():
    buffer: Final = RedisUpdateBuffer(redis_cache=_ListRedis())
    buffer.should_commit_spend_updates_to_redis = MagicMock(return_value=True)

    assert await buffer.store_spend_logs_in_redis([{"request_id": "nan-row", "spend": float("nan")}]) is True

    assert await buffer.get_spend_logs_from_redis_buffer(limit=10) == ({"request_id": "nan-row", "spend": 0.0},)


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
async def test_tool_discoveries_go_back_on_the_queue_when_the_redis_push_fails():
    from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import ToolDiscoveryQueue

    redis_cache = MagicMock()
    redis_cache.async_rpush_pipeline = AsyncMock(side_effect=ConnectionError("redis down"))
    queue = ToolDiscoveryQueue()
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})
    queue.add_update({"tool_name": "get_weather", "origin": "user_defined"})

    await RedisUpdateBuffer(redis_cache=redis_cache).store_tool_discoveries_in_redis(queue)

    assert queue.flush() == [{"tool_name": "get_weather", "origin": "user_defined", "call_count": 2}]
