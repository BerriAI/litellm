import asyncio
import os
import uuid
from typing import Final
from unittest.mock import patch

import pytest

from litellm.caching.affinity_cache import claim_affinity_pin
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
