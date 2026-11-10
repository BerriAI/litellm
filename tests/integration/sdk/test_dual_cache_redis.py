import asyncio
import os
import uuid
from typing import Final
from unittest.mock import patch

import pytest

from litellm.caching.affinity_cache import claim_affinity_pin, read_affinity_pin
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache


def _redis_cache() -> RedisCache:
    return RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


@pytest.mark.asyncio
async def test_a_value_only_in_redis_is_read_once_from_redis_then_from_memory() -> None:
    redis_cache: Final = _redis_cache()
    dual_cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)
    sync_key: Final = f"redis-only-sync-{uuid.uuid4()}"
    async_key: Final = f"redis-only-async-{uuid.uuid4()}"
    redis_cache.set_cache(sync_key, {"v": "sync"})
    await redis_cache.async_set_cache(async_key, {"v": "async"})

    assert dual_cache.get_cache(sync_key) == {"v": "sync"}
    assert await dual_cache.async_get_cache(async_key) == {"v": "async"}

    with (
        patch.object(redis_cache, "get_cache") as sync_redis_read,
        patch.object(redis_cache, "async_get_cache") as async_redis_read,
    ):
        assert dual_cache.get_cache(sync_key) == {"v": "sync"}
        assert await dual_cache.async_get_cache(async_key) == {"v": "async"}
        sync_redis_read.assert_not_called()
        async_redis_read.assert_not_called()


@pytest.mark.asyncio
async def test_a_deleted_key_is_gone_from_both_memory_and_redis() -> None:
    redis_cache: Final = _redis_cache()
    dual_cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)
    sync_key: Final = f"deleted-sync-{uuid.uuid4()}"
    async_key: Final = f"deleted-async-{uuid.uuid4()}"
    dual_cache.set_cache(sync_key, {"v": "sync"})
    await dual_cache.async_set_cache(async_key, {"v": "async"})

    dual_cache.delete_cache(sync_key)
    await dual_cache.async_delete_cache(async_key)

    assert dual_cache.get_cache(sync_key) is None
    assert await dual_cache.async_get_cache(async_key) is None
    assert redis_cache.get_cache(sync_key) is None
    assert await redis_cache.async_get_cache(async_key) is None


@pytest.mark.asyncio
async def test_a_batch_read_without_an_in_memory_cache_reads_redis() -> None:
    redis_cache: Final = _redis_cache()
    dual_cache: Final = DualCache(in_memory_cache=None, redis_cache=redis_cache)
    key: Final = f"no-memory-{uuid.uuid4()}"
    await redis_cache.async_set_cache(key, {"v": "from-redis"})

    assert await dual_cache.async_batch_get_cache([key]) == [{"v": "from-redis"}]


@pytest.mark.asyncio
async def test_sync_and_async_batch_reads_share_one_redis_without_sync_reads_going_async() -> None:
    redis_cache: Final = _redis_cache()
    dual_cache: Final = DualCache(redis_cache=redis_cache)
    run_id: Final = uuid.uuid4().hex
    sync_keys: Final = [f"sync_{run_id}_{index}" for index in range(5)]
    async_keys: Final = [f"async_{run_id}_{index}" for index in range(5)]
    in_loop_keys: Final = [f"in_loop_{run_id}_{index}" for index in range(3)]
    survivor_key: Final = f"survivor_{run_id}"
    expected: Final = {key: {"key": key} for key in [*sync_keys, *async_keys, *in_loop_keys, survivor_key]}
    await asyncio.gather(*(redis_cache.async_set_cache(key, value, ttl=60) for key, value in expected.items()))

    concurrent_results: Final = await asyncio.gather(
        *(asyncio.to_thread(dual_cache.batch_get_cache, keys=[key]) for key in sync_keys),
        *(dual_cache.async_batch_get_cache(keys=[key]) for key in async_keys),
    )
    assert list(concurrent_results) == [[expected[key]] for key in [*sync_keys, *async_keys]]

    with patch.object(
        redis_cache,
        "async_batch_get_cache",
        side_effect=AssertionError("sync batch reads must not call async Redis"),
    ):
        in_loop_results: Final = [dual_cache.batch_get_cache(keys=[key]) for key in in_loop_keys]

    assert in_loop_results == [[expected[key]] for key in in_loop_keys]
    assert await dual_cache.async_batch_get_cache(keys=[survivor_key]) == [expected[survivor_key]]


@pytest.mark.asyncio
async def test_concurrent_session_repin_claims_keep_the_shared_winner_after_recovery() -> None:
    redis: Final = _redis_cache()
    workers: Final = (DualCache(redis_cache=redis), DualCache(redis_cache=redis))
    key: Final = f"session-repin-{uuid.uuid4().hex}"
    choices: Final = ({"model_id": "replacement"}, {"model_id": "other"})
    try:
        await redis.async_set_cache(key, {"model_id": "original"}, ttl=60)
        for worker in workers:
            worker.in_memory_cache.set_cache(key, {"model_id": "original"}, ttl=60)
        winners: Final = await asyncio.gather(
            *(
                claim_affinity_pin(worker, key, choice, 60, eligible_values=choices, preserve_legacy_model_ids=True)
                for worker, choice in zip(workers, choices, strict=True)
            )
        )
        assert winners[0] == winners[1] and winners[0] in choices
        late_original: Final = await claim_affinity_pin(
            workers[0],
            key,
            {"model_id": "original"},
            60,
            eligible_values=({"model_id": "original"}, *choices),
            preserve_legacy_model_ids=True,
        )
        assert late_original == winners[0]
        assert await redis.async_get_cache(key) == winners[0]
    finally:
        await redis.async_delete_cache(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["original", '"original"', '{"model_id":"original"}'])
async def test_session_repin_lua_preserves_healthy_legacy_pins_and_replaces_unavailable_ones(stored: str) -> None:
    redis: Final = _redis_cache()
    cache: Final = DualCache(redis_cache=redis)
    key: Final = f"session-repin-legacy-{uuid.uuid4().hex}"
    replacement: Final = {"model_id": "replacement"}
    try:
        redis.redis_client.set(redis.check_and_fix_namespace(key), stored, ex=60)
        preserved: Final = await claim_affinity_pin(
            cache,
            key,
            replacement,
            60,
            eligible_values=({"model_id": "original"}, replacement),
            preserve_legacy_model_ids=True,
        )
        assert preserved in ("original", {"model_id": "original"})
        moved: Final = await claim_affinity_pin(
            cache,
            key,
            replacement,
            60,
            eligible_values=(replacement,),
            preserve_legacy_model_ids=True,
        )
        assert moved == replacement
        assert await redis.async_get_cache(key) == replacement
    finally:
        await redis.async_delete_cache(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ['{"model":"current","tier":"REASONING"}', '"current"', "current"])
async def test_tier_pin_reads_shared_state_refreshes_ttl_and_forgets_deleted_pins(stored: str) -> None:
    redis: Final = _redis_cache()
    redis.namespace = f"tier-read-{uuid.uuid4().hex}"
    workers: Final = (DualCache(redis_cache=redis), DualCache(redis_cache=redis))
    key: Final = "session"
    namespaced: Final = redis.check_and_fix_namespace(key)
    try:
        for worker in workers:
            worker.in_memory_cache.set_cache(key, {"model": "old", "tier": "SIMPLE"}, ttl=60)
        redis.redis_client.set(namespaced, stored, ex=10)
        first: Final = await read_affinity_pin(workers[0], key, 60)
        second: Final = await read_affinity_pin(workers[1], key, 60)
        assert first == second and first in ("current", {"model": "current", "tier": "REASONING"})
        assert all(worker.in_memory_cache.get_cache(key) == first for worker in workers)
        assert 10 < redis.redis_client.ttl(namespaced) <= 60
        replacement: Final = {"model": "newer", "tier": "SIMPLE"}
        await redis.async_set_cache(key, replacement, ttl=10)
        assert await read_affinity_pin(workers[0], key, 60) == replacement
        assert await read_affinity_pin(workers[1], key, 60) == replacement
        await redis.async_delete_cache(key)
        for worker in workers:
            assert await read_affinity_pin(worker, key, 60) is None
            assert worker.in_memory_cache.get_cache(key) is None
        assert redis.redis_client.exists(namespaced) == 0
    finally:
        await redis.async_delete_cache(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["joined", "separate", "script_error"])
async def test_tier_pin_joins_pending_reads_without_reusing_request_state(mode: str) -> None:
    import json
    from collections.abc import Iterable

    import litellm
    from redis.asyncio import BlockingConnectionPool, Connection, Redis

    from litellm.caching.redis_batch import request_redis_batch_scope

    sends: Final[list[bytes]] = []

    class RecordingConnection(Connection):
        async def send_packed_command(
            self, command: bytes | str | Iterable[bytes], check_health: bool = True
        ) -> None:
            packed: Final = command.encode() if isinstance(command, str) else (
                command if isinstance(command, bytes) else b"".join(command)
            )
            sends.append(packed)
            await super().send_packed_command(packed, check_health=check_health)

    namespace: Final = f"tier-batch-{uuid.uuid4().hex}"
    redis: Final = RedisCache(
        host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]),
        namespace=namespace, client_name=namespace,
    )
    pool: Final = BlockingConnectionPool(
        host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]),
        connection_class=RecordingConnection,
    )
    client: Final = Redis(connection_pool=pool)
    client_key: Final = redis._get_async_client_cache_key()
    litellm.in_memory_llm_clients_cache.set_cache(key=client_key, value=client)
    cache: Final = DualCache(redis_cache=redis)
    key: Final = redis.check_and_fix_namespace("pin")
    other_key: Final = redis.check_and_fix_namespace("other")
    stale: Final = {"model": "old", "tier": "SIMPLE"}
    current: Final = {"model": "current", "tier": "REASONING"}
    newer: Final = {"model": "newer", "tier": "COMPLEX"}
    try:
        redis.redis_client.set(key, json.dumps(current), ex=10)
        redis.redis_client.set(other_key, json.dumps("other-value"), ex=60)
        assert await read_affinity_pin(cache, "pin", 60) == current
        cache.in_memory_cache.set_cache("pin", stale, ttl=60)
        assert cache.in_memory_cache.get_cache("pin") == stale
        if mode == "script_error":
            redis.redis_client.delete(key)
            redis.redis_client.rpush(key, "wrong-type")
        start: Final = len(sends)
        with request_redis_batch_scope() as request:
            batch: Final = request.batch(redis)
            pending: Final = batch.mget(("other",))
            if mode == "separate":
                assert await pending == {"other": "other-value"}
            assert await read_affinity_pin(cache, "pin", 60) == (stale if mode == "script_error" else current)
            assert await pending == {"other": "other-value"}
            wire: Final = tuple(data for data in sends[start:] if b"MGET" in data or b"EVALSHA" in data)
            assert len(wire) == (2 if mode == "separate" else 1)
            assert sum(b"MGET" in data for data in wire) == sum(b"EVALSHA" in data for data in wire) == 1
            if mode == "script_error":
                assert redis.redis_client.type(key) == b"list"
                redis.redis_client.delete(key)
            else:
                assert 10 < redis.redis_client.ttl(key) <= 60
            redis.redis_client.set(key, json.dumps(newer), ex=10)
            later_pending: Final = batch.mget(("other",))
            assert await read_affinity_pin(cache, "pin", 60) == newer
            assert await later_pending == {"other": "other-value"}
            assert cache.in_memory_cache.get_cache("pin") == newer
            assert 10 < redis.redis_client.ttl(key) <= 60
            assert redis.redis_client.get(key) == json.dumps(newer).encode()
            redis.redis_client.delete(key)
            assert await read_affinity_pin(cache, "pin", 60) is None
            assert cache.in_memory_cache.get_cache("pin") is None
            assert redis.redis_client.exists(key) == 0
    finally:
        redis.redis_client.delete(key, other_key)
        for cached_key in tuple(litellm.in_memory_llm_clients_cache.cache_dict):
            if client_key in cached_key:
                litellm.in_memory_llm_clients_cache.delete_cache(cached_key)
        await client.aclose()
        await pool.aclose()
        await redis.disconnect()
        redis.redis_client.close()
