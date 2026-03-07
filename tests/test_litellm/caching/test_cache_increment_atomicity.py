import asyncio
import threading
import time

import pytest

from litellm.caching.disk_cache import DiskCache
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache


def _run_concurrent_threaded_increments(
    increment_fn, num_threads: int, increments_per_thread: int
):
    barrier = threading.Barrier(num_threads)

    def _worker():
        barrier.wait()
        for _ in range(increments_per_thread):
            increment_fn()

    threads = [threading.Thread(target=_worker) for _ in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def test_in_memory_increment_cache_is_atomic_under_thread_concurrency(monkeypatch):
    cache = InMemoryCache()
    cache.set_cache("counter", 0)

    original_get_cache = cache.get_cache

    def delayed_get_cache(*args, **kwargs):
        result = original_get_cache(*args, **kwargs)
        time.sleep(0.0001)
        return result

    # Delay reads to force interleaving; increment must still be correct.
    monkeypatch.setattr(cache, "get_cache", delayed_get_cache)

    num_threads = 8
    increments_per_thread = 200
    _run_concurrent_threaded_increments(
        increment_fn=lambda: cache.increment_cache("counter", 1),
        num_threads=num_threads,
        increments_per_thread=increments_per_thread,
    )

    assert cache.get_cache("counter") == num_threads * increments_per_thread


@pytest.mark.asyncio
async def test_in_memory_async_increment_cache_is_atomic_under_async_concurrency(
    monkeypatch,
):
    cache = InMemoryCache()
    await cache.async_set_cache("counter", 0)

    original_async_get_cache = cache.async_get_cache
    original_async_set_cache = cache.async_set_cache

    async def delayed_async_get_cache(*args, **kwargs):
        result = await original_async_get_cache(*args, **kwargs)
        await asyncio.sleep(0)
        return result

    async def delayed_async_set_cache(*args, **kwargs):
        await asyncio.sleep(0)
        return await original_async_set_cache(*args, **kwargs)

    # If async_increment ever regresses to async read-modify-write, these hooks
    # force interleaving and expose lost updates.
    monkeypatch.setattr(cache, "async_get_cache", delayed_async_get_cache)
    monkeypatch.setattr(cache, "async_set_cache", delayed_async_set_cache)

    num_tasks = 8
    increments_per_task = 200

    async def _worker():
        for _ in range(increments_per_task):
            await cache.async_increment("counter", 1)
            await asyncio.sleep(0)

    await asyncio.gather(*[_worker() for _ in range(num_tasks)])

    assert await cache.async_get_cache("counter") == num_tasks * increments_per_task


def test_disk_increment_cache_is_atomic_under_thread_concurrency(tmp_path, monkeypatch):
    pytest.importorskip("diskcache")
    cache = DiskCache(disk_cache_dir=str(tmp_path / "disk-cache"))
    cache.set_cache("counter", 0)

    original_disk_get = cache.disk_cache.get

    def delayed_disk_get(*args, **kwargs):
        result = original_disk_get(*args, **kwargs)
        time.sleep(0.0001)
        return result

    # Delay the actual storage read path used by increment_cache.
    monkeypatch.setattr(cache.disk_cache, "get", delayed_disk_get)

    num_threads = 6
    increments_per_thread = 150
    _run_concurrent_threaded_increments(
        increment_fn=lambda: cache.increment_cache("counter", 1),
        num_threads=num_threads,
        increments_per_thread=increments_per_thread,
    )

    assert cache.get_cache("counter") == num_threads * increments_per_thread


@pytest.mark.asyncio
async def test_disk_async_increment_cache_is_atomic_under_async_concurrency(
    tmp_path, monkeypatch
):
    pytest.importorskip("diskcache")
    cache = DiskCache(disk_cache_dir=str(tmp_path / "disk-cache-async"))
    await cache.async_set_cache("counter", 0)

    original_async_get_cache = cache.async_get_cache
    original_async_set_cache = cache.async_set_cache

    async def delayed_async_get_cache(*args, **kwargs):
        result = await original_async_get_cache(*args, **kwargs)
        await asyncio.sleep(0)
        return result

    async def delayed_async_set_cache(*args, **kwargs):
        await asyncio.sleep(0)
        return await original_async_set_cache(*args, **kwargs)

    # If async_increment ever regresses to async read-modify-write, these hooks
    # force interleaving and expose lost updates.
    monkeypatch.setattr(cache, "async_get_cache", delayed_async_get_cache)
    monkeypatch.setattr(cache, "async_set_cache", delayed_async_set_cache)

    num_tasks = 6
    increments_per_task = 150

    async def _worker():
        for _ in range(increments_per_task):
            await cache.async_increment("counter", 1)
            await asyncio.sleep(0)

    await asyncio.gather(*[_worker() for _ in range(num_tasks)])

    assert await cache.async_get_cache("counter") == num_tasks * increments_per_task


def test_dual_cache_increment_is_atomic_when_using_in_memory_only(monkeypatch):
    in_memory_cache = InMemoryCache()
    cache = DualCache(in_memory_cache=in_memory_cache, redis_cache=None)
    cache.set_cache("counter", 0)

    original_get_cache = in_memory_cache.get_cache

    def delayed_get_cache(*args, **kwargs):
        result = original_get_cache(*args, **kwargs)
        time.sleep(0.0001)
        return result

    monkeypatch.setattr(in_memory_cache, "get_cache", delayed_get_cache)

    num_threads = 8
    increments_per_thread = 200
    _run_concurrent_threaded_increments(
        increment_fn=lambda: cache.increment_cache("counter", 1),
        num_threads=num_threads,
        increments_per_thread=increments_per_thread,
    )

    assert cache.get_cache("counter") == num_threads * increments_per_thread


@pytest.mark.asyncio
async def test_dual_cache_async_increment_is_atomic_when_using_in_memory_only():
    cache = DualCache(in_memory_cache=InMemoryCache(), redis_cache=None)
    await cache.async_set_cache("counter", 0)

    num_tasks = 8
    increments_per_task = 200

    async def _worker():
        for _ in range(increments_per_task):
            await cache.async_increment_cache("counter", 1)
            await asyncio.sleep(0)

    await asyncio.gather(*[_worker() for _ in range(num_tasks)])

    assert await cache.async_get_cache("counter") == num_tasks * increments_per_task
