import asyncio
import os
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Final
from unittest.mock import patch

import pytest
import pytest_asyncio
from integration._support.async_server import LoopbackServer, close_writer
from integration._support.ports import allocate_port
from pydantic import ValidationError

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.scheduler import FlowItem, Scheduler, SchedulerCacheKeys


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


@pytest_asyncio.fixture(loop_scope="function")
async def shared_scheduler_cache() -> AsyncIterator[RedisCache]:
    cache: Final = RedisCache(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        namespace="scheduler-atomic-" + uuid.uuid4().hex,
    )
    try:
        yield cache
    finally:
        client: Final = cache.init_async_client()
        await client.delete(cache.check_and_fix_namespace(f"{SchedulerCacheKeys.queue.value}:shared"))
        await client.aclose()
        cache.redis_client.close()


async def test_shared_scheduler_preserves_concurrent_admissions(shared_scheduler_cache: RedisCache) -> None:
    first: Final = Scheduler(redis_cache=shared_scheduler_cache)
    second: Final = Scheduler(redis_cache=shared_scheduler_cache)
    await asyncio.gather(first.get_queue("shared"), second.get_queue("shared"))
    await asyncio.gather(
        first.add_request(FlowItem(priority=1, request_id="first", model_name="shared")),
        second.add_request(FlowItem(priority=2, request_id="second", model_name="shared")),
    )
    assert await shared_scheduler_cache.async_get_cache("scheduler:queue:shared") == [[1, "first"], [2, "second"]]
    assert await first.get_queue("shared") == [(1, "first"), (2, "second")]
    assert await second.get_queue("shared") == [(1, "first"), (2, "second")]


async def test_shared_scheduler_does_not_resurrect_peer_removal(shared_scheduler_cache: RedisCache) -> None:
    await shared_scheduler_cache.async_set_cache("scheduler:queue:shared", [[1, "removed"], [2, "waiting"]])
    first: Final = Scheduler(redis_cache=shared_scheduler_cache)
    second: Final = Scheduler(redis_cache=shared_scheduler_cache)
    await asyncio.gather(first.get_queue("shared"), second.get_queue("shared"))
    await first.remove_request("removed", "shared")
    await second.add_request(FlowItem(priority=3, request_id="new", model_name="shared"))
    assert await shared_scheduler_cache.async_get_cache("scheduler:queue:shared") == [[2, "waiting"], [3, "new"]]


async def test_shared_scheduler_observes_peer_pop(shared_scheduler_cache: RedisCache) -> None:
    await shared_scheduler_cache.async_set_cache("scheduler:queue:shared", [[1, "first"], [2, "second"]])
    first: Final = Scheduler(redis_cache=shared_scheduler_cache)
    second: Final = Scheduler(redis_cache=shared_scheduler_cache)
    await asyncio.gather(first.get_queue("shared"), second.get_queue("shared"))
    assert await first.poll(FlowItem(priority=1, request_id="first", model_name="shared"), [])
    assert await second.poll(FlowItem(priority=2, request_id="second", model_name="shared"), [])
    assert await shared_scheduler_cache.async_get_cache("scheduler:queue:shared") == []


async def test_shared_scheduler_keeps_other_requests_during_concurrent_cleanup(
    shared_scheduler_cache: RedisCache,
) -> None:
    schedulers: Final = tuple(Scheduler(redis_cache=shared_scheduler_cache) for _ in range(16))
    await asyncio.gather(
        *(
            scheduler.add_request(FlowItem(priority=i, request_id=str(i), model_name="shared"))
            for i, scheduler in enumerate(schedulers)
        )
    )
    await asyncio.gather(
        *(scheduler.remove_request(str(i), "shared") for i, scheduler in enumerate(schedulers) if i % 2 == 0)
    )
    assert await schedulers[0].get_queue("shared") == [(i, str(i)) for i in range(16) if i % 2]


@pytest.fixture
def interrupted_scheduler_cache(shared_scheduler_cache: RedisCache) -> Iterator[tuple[RedisCache, threading.Event]]:
    interrupted: Final = threading.Event()

    async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])
        )

        async def forward(source: asyncio.StreamReader, destination: asyncio.StreamWriter) -> None:
            try:
                while chunk := await source.read(65536):
                    if interrupted.is_set():
                        return
                    destination.write(chunk)
                    await destination.drain()
            except ConnectionError:
                return
            finally:
                await close_writer(destination)

        await asyncio.gather(forward(reader, upstream_writer), forward(upstream_reader, writer))

    port: Final = allocate_port()
    server: Final = LoopbackServer(port, relay)
    server.start()
    cache: Final = RedisCache(host="127.0.0.1", port=port, namespace=shared_scheduler_cache.namespace)
    try:
        yield cache, interrupted
    finally:
        cache.redis_client.close()
        server.stop()


async def test_shared_scheduler_keeps_local_fallback_out_of_recovered_shared_queue(
    shared_scheduler_cache: RedisCache,
    interrupted_scheduler_cache: tuple[RedisCache, threading.Event],
) -> None:
    cache, interrupted = interrupted_scheduler_cache
    scheduler: Final = Scheduler(redis_cache=cache)
    try:
        await scheduler.add_request(FlowItem(priority=1, request_id="removed", model_name="shared"))
        interrupted.set()
        await scheduler.add_request(FlowItem(priority=2, request_id="local", model_name="shared"))
        assert await scheduler.get_queue("shared") == [(1, "removed"), (2, "local")]
        await Scheduler(redis_cache=shared_scheduler_cache).remove_request("removed", "shared")
        interrupted.clear()
        await scheduler.add_request(FlowItem(priority=3, request_id="new", model_name="shared"))
        assert await scheduler.get_queue("shared") == [(3, "new")]
        assert await shared_scheduler_cache.async_get_cache("scheduler:queue:shared") == [[3, "new"]]
    finally:
        interrupted.clear()
        await cache.init_async_client().aclose()


async def test_shared_scheduler_preserves_invalid_queue_for_diagnosis(shared_scheduler_cache: RedisCache) -> None:
    await shared_scheduler_cache.async_set_cache("scheduler:queue:shared", {"invalid": "queue"})
    scheduler: Final = Scheduler(redis_cache=shared_scheduler_cache)
    with pytest.raises(ValidationError):
        await scheduler.add_request(FlowItem(priority=1, request_id="first", model_name="shared"))
    assert await shared_scheduler_cache.async_get_cache("scheduler:queue:shared") == {"invalid": "queue"}


@pytest.mark.parametrize("priority", [0, -1, 2**80])
async def test_shared_scheduler_retains_priority_and_request_id(
    shared_scheduler_cache: RedisCache, priority: int
) -> None:
    scheduler: Final = Scheduler(redis_cache=shared_scheduler_cache)
    scheduler.cache.update_cache_ttl(default_in_memory_ttl=17, default_redis_ttl=None)
    await scheduler.add_request(FlowItem(priority=priority, request_id="请求-é", model_name="shared"))
    assert await scheduler.get_queue("shared") == [(priority, "请求-é")]
    assert (
        0
        < shared_scheduler_cache.redis_client.ttl(
            shared_scheduler_cache.check_and_fix_namespace("scheduler:queue:shared")
        )
        <= 17
    )
