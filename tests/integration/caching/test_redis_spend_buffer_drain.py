import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import pytest
import redis.asyncio as async_redis
from integration._support.redis_process import owned_redis
from redis.asyncio.client import Pipeline
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_BUFFER_PROCESSING_LIST_TTL_SECONDS, REDIS_SPEND_LOGS_BUFFER_KEY
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer

_DrainScenario = Callable[[RedisCache, async_redis.Redis], Awaitable[None]]


def _run_against_real_redis(tmp_path: Path, redis_version: str | float, scenario: _DrainScenario) -> None:
    async def drive() -> None:
        with owned_redis(tmp_path) as redis_server:
            redis_cache: Final = RedisCache(host=redis_server.host, port=redis_server.port, namespace="drain-test")
            redis_cache.redis_version = redis_version
            redis_client: Final = async_redis.Redis(host=redis_server.host, port=redis_server.port)
            try:
                await scenario(redis_cache, redis_client)
            finally:
                await redis_client.aclose()
                await redis_cache.disconnect()

    asyncio.run(drive())


@pytest.mark.parametrize("redis_version", ["6.0", 6.1])
def test_legacy_redis_drain_keeps_rows_pending_in_order_until_acknowledged(
    redis_version: str | float, tmp_path: Path
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3", "row-4", "row-5")

        first_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        processing_key: Final = "drain-test:buffer:processing:worker-1"
        assert first_batch == ["row-1", "row-2"]
        assert await redis_client.llen(processing_key) == 2
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-3", b"row-4", b"row-5"]

        pending_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        assert pending_batch == ["row-1", "row-2"]
        assert await redis_client.lrange(processing_key, 0, -1) == [b"row-1", b"row-2"]

        await redis_cache.async_ack_processing_list(key="buffer", worker_id="worker-1", ttl_seconds=60)
        assert await redis_cache.async_processing_list_state(key="buffer", worker_id="worker-1") == "acknowledged"
        next_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        assert next_batch == ["row-3", "row-4"]
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-5"]

    _run_against_real_redis(tmp_path, redis_version, scenario)


def test_lua_drain_as_half_open_probe_closes_the_circuit_breaker(tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        breaker: Final = redis_cache._circuit_breaker
        await redis_client.rpush("drain-test:buffer", "row-1")
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        assert breaker.state == breaker.OPEN
        breaker.recovery_timeout = 0
        await asyncio.sleep(0.01)

        drained: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=1, ttl_seconds=60
        )

        assert drained == ["row-1"]
        assert breaker.state == breaker.CLOSED
        await redis_cache.async_set_cache("after-probe", "ok")
        assert await redis_client.get("drain-test:after-probe") is not None

    _run_against_real_redis(tmp_path, "6.0.16", scenario)


@pytest.mark.parametrize("redis_version", ["Unknown", "latest", ""])
def test_unknown_redis_version_drains_through_lua_without_lmove(redis_version: str, tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.execute_command("ACL", "SETUSER", "default", "-lmove")
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3")

        drained: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )

        assert drained == ["row-1", "row-2"]
        assert await redis_client.lrange("drain-test:buffer:processing:worker-1", 0, -1) == [b"row-1", b"row-2"]
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-3"]

    _run_against_real_redis(tmp_path, redis_version, scenario)


@pytest.mark.parametrize("redis_version", ["6.0", "7.2"])
def test_abandoned_processing_list_returns_to_the_buffer_for_another_worker(redis_version: str, tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        if redis_version == "6.0":
            await redis_client.execute_command("ACL", "SETUSER", "default", "-lmove")
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3")
        abandoned: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="old-leader", count=2, ttl_seconds=60
        )
        assert abandoned == ["row-1", "row-2"]

        fresh_list_untouched: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="new-leader", count=3, ttl_seconds=60, recovery_seconds=60
        )
        assert fresh_list_untouched == ["row-3"]
        await redis_cache.async_ack_processing_list(key="buffer", worker_id="new-leader", ttl_seconds=60)

        recovered: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="new-leader", count=3, ttl_seconds=60, recovery_seconds=0
        )
        assert recovered == ["row-1", "row-2"]
        assert await redis_cache.async_processing_list_state(key="buffer", worker_id="old-leader") == "missing"

    _run_against_real_redis(tmp_path, redis_version, scenario)


def test_a_retried_legacy_drain_script_keeps_the_rows_its_first_run_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3")
        cache_client: Final = redis_cache.init_async_client()
        original_eval: Final = cache_client.eval

        async def eval_retried_after_lost_reply(*args: object) -> object:
            await original_eval(*args)
            return await original_eval(*args)

        monkeypatch.setattr(cache_client, "eval", eval_retried_after_lost_reply)

        drained: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )

        assert drained == ["row-1", "row-2"]
        assert await redis_client.lrange("drain-test:buffer:processing:worker-1", 0, -1) == [b"row-1", b"row-2"]
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-3"]

    _run_against_real_redis(tmp_path, "6.0", scenario)


def _cut_first_pipeline(
    monkeypatch: pytest.MonkeyPatch, redis_cache: RedisCache, sent_commands: int | None, error: Exception
) -> None:
    """Run the first pipeline's first sent_commands commands (all when None) on Redis, then fail like a lost link"""
    cache_client: Final = redis_cache.init_async_client()
    original_pipeline: Final = cache_client.pipeline
    pipeline_calls: Final = iter(range(2))

    def pipeline_cut_after_send(transaction: bool = True) -> Pipeline:
        pipeline: Final = original_pipeline(transaction=transaction)
        if next(pipeline_calls, None) != 0:
            return pipeline
        original_execute: Final = pipeline.execute

        async def execute_then_fail(raise_on_error: bool = True) -> list[object]:
            commands: Final = tuple(pipeline.command_stack)
            pipeline.command_stack = list(commands if sent_commands is None else commands[:sent_commands])
            await original_execute(raise_on_error=raise_on_error)
            raise error

        monkeypatch.setattr(pipeline, "execute", execute_then_fail)
        return pipeline

    monkeypatch.setattr(cache_client, "pipeline", pipeline_cut_after_send)


def test_rows_moved_before_a_dropped_connection_are_recoverable_by_another_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2")
        _cut_first_pipeline(
            monkeypatch, redis_cache, sent_commands=3, error=RedisConnectionError("connection dropped mid-pipeline")
        )
        with pytest.raises(RedisConnectionError, match="connection dropped"):
            await redis_cache.async_drain_list_to_processing(
                key="buffer", worker_id="worker-1", count=10, ttl_seconds=60
            )

        recovered: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-2", count=10, ttl_seconds=60, recovery_seconds=0
        )

        assert recovered == ["row-1", "row-2"]

    _run_against_real_redis(tmp_path, "7.2", scenario)


@pytest.mark.parametrize("redis_version", ["6.0", "7.2"])
def test_an_owner_cannot_acknowledge_a_list_another_worker_already_claimed(redis_version: str, tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1")
        assert await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="owner", count=10, ttl_seconds=60
        ) == ["row-1"]
        assert await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="claimer", count=10, ttl_seconds=60, recovery_seconds=0
        ) == ["row-1"]

        assert await redis_cache.async_ack_processing_list(key="buffer", worker_id="owner", ttl_seconds=60) is False
        assert await redis_cache.async_ack_processing_list(key="buffer", worker_id="claimer", ttl_seconds=60) is True
        assert (
            await redis_cache.async_drain_list_to_processing(key="buffer", worker_id="owner", count=10, ttl_seconds=60)
            == []
        )

    _run_against_real_redis(tmp_path, redis_version, scenario)


def test_recovery_leaves_an_acknowledgement_marker_and_its_registration_in_place(tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1")
        assert await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="owner", count=10, ttl_seconds=60
        ) == ["row-1"]
        assert await redis_cache.async_ack_processing_list(key="buffer", worker_id="owner", ttl_seconds=60) is True

        assert (
            await redis_cache.async_drain_list_to_processing(
                key="buffer", worker_id="other", count=10, ttl_seconds=60, recovery_seconds=0
            )
            == []
        )

        assert await redis_cache.async_processing_list_state(key="buffer", worker_id="owner") == "acknowledged"
        assert b"drain-test:buffer:processing:owner" in await redis_client.smembers(
            "drain-test:buffer:processing-lists"
        )

    _run_against_real_redis(tmp_path, "7.2", scenario)


def test_new_leader_recovers_rows_an_old_leader_drained_but_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        old_leader: Final = RedisUpdateBuffer(redis_cache=redis_cache)
        old_leader.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
        new_leader: Final = RedisUpdateBuffer(redis_cache=redis_cache, processing_list_recovery_seconds=0)
        new_leader.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
        rows: Final = ({"request_id": "req-1"}, {"request_id": "req-2"})
        await redis_client.rpush(f"drain-test:{REDIS_SPEND_LOGS_BUFFER_KEY}", *(json.dumps(row) for row in rows))
        _cut_first_pipeline(
            monkeypatch, redis_cache, sent_commands=None, error=RedisTimeoutError("simulated response timeout")
        )
        with pytest.raises(RedisTimeoutError, match="simulated response timeout"):
            await old_leader.get_spend_logs_from_redis_buffer(limit=10)

        assert await new_leader.get_spend_logs_from_redis_buffer(limit=10) == rows
        assert await old_leader.get_spend_logs_from_redis_buffer(limit=10) == ()
        assert await new_leader.get_spend_logs_from_redis_buffer(limit=10) == ()

    _run_against_real_redis(tmp_path, "7.2", scenario)


def test_rows_another_pod_claims_mid_flush_are_consumed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        old_leader: Final = RedisUpdateBuffer(redis_cache=redis_cache)
        old_leader.should_commit_spend_updates_to_redis = MagicMock(return_value=True)
        rows: Final = ({"request_id": "req-1"}, {"request_id": "req-2"})
        await redis_client.rpush(f"drain-test:{REDIS_SPEND_LOGS_BUFFER_KEY}", *(json.dumps(row) for row in rows))
        cache_client: Final = redis_cache.init_async_client()
        original_set: Final = cache_client.set
        other_pod_claims: Final = asyncio.Queue()

        async def other_pod_claims_before_ack(name: str, value: str, ex: int | None = None, xx: bool = False) -> object:
            if other_pod_claims.empty():
                other_pod_claims.put_nowait(
                    await redis_cache.async_drain_list_to_processing(
                        key=REDIS_SPEND_LOGS_BUFFER_KEY,
                        worker_id="other-pod",
                        count=10,
                        ttl_seconds=REDIS_BUFFER_PROCESSING_LIST_TTL_SECONDS,
                        recovery_seconds=0,
                    )
                )
            return await original_set(name, value, ex=ex, xx=xx)

        monkeypatch.setattr(cache_client, "set", other_pod_claims_before_ack)
        old_leader_rows: Final = await old_leader.get_spend_logs_from_redis_buffer(limit=10)

        claimed: Final = tuple(json.loads(row) for row in other_pod_claims.get_nowait())
        assert (old_leader_rows, claimed) == ((), rows)
        assert await old_leader.get_spend_logs_from_redis_buffer(limit=10) == ()

    _run_against_real_redis(tmp_path, "7.2", scenario)


@pytest.mark.parametrize("redis_version", ["6.0", "7.2"])
def test_rows_pending_past_their_registration_stay_recoverable_by_a_new_leader(
    redis_version: str, tmp_path: Path
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:spend-buffer", "row-1", "row-2")
        assert await redis_cache.async_drain_list_to_processing(
            key="spend-buffer", worker_id="owner", count=10, ttl_seconds=REDIS_BUFFER_PROCESSING_LIST_TTL_SECONDS
        ) == ["row-1", "row-2"]
        await redis_client.delete("drain-test:spend-buffer:processing-lists")
        assert await redis_cache.async_drain_list_to_processing(
            key="spend-buffer", worker_id="owner", count=10, ttl_seconds=REDIS_BUFFER_PROCESSING_LIST_TTL_SECONDS
        ) == ["row-1", "row-2"]

        assert await redis_cache.async_drain_list_to_processing(
            key="spend-buffer",
            worker_id="new-leader",
            count=10,
            ttl_seconds=REDIS_BUFFER_PROCESSING_LIST_TTL_SECONDS,
            recovery_seconds=0,
        ) == ["row-1", "row-2"]

    _run_against_real_redis(tmp_path, redis_version, scenario)
