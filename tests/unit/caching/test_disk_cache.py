import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from pydantic import JsonValue

pytest.importorskip("diskcache")

from litellm.caching.disk_cache import DiskCache


class _SlowInt(int):
    def __add__(self, value: int) -> "_SlowInt":
        time.sleep(0.05)
        return _SlowInt(int(self) + value)


@pytest.fixture
def cache(tmp_path):
    return DiskCache(disk_cache_dir=str(tmp_path))


@pytest.mark.parametrize("value", (0, False, "", [], {}, 1, {"answer": "cached"}))
async def test_cache_preserves_falsy_values(cache: DiskCache, value: JsonValue) -> None:
    await cache.async_set_cache("result", value)

    result: Final = cache.get_cache("result")
    assert result == value
    assert type(result) is type(value)
    assert await cache.async_get_cache("result") == value
    assert cache.batch_get_cache(["result", "missing"]) == [value, None]
    assert await cache.async_batch_get_cache(["result", "missing"]) == [value, None]


def test_increment_cache_starts_from_zero_when_key_missing(cache):
    assert cache.increment_cache("counter", 3) == 3
    assert cache.get_cache("counter") == 3


def test_increment_cache_adds_to_existing_int(cache):
    cache.set_cache("counter", 7)
    assert cache.increment_cache("counter", 5) == 12
    assert cache.get_cache("counter") == 12


def test_increment_cache_preserves_zero_result(cache: DiskCache) -> None:
    assert cache.increment_cache("counter", 1) == 1
    assert cache.increment_cache("counter", -1) == 0
    assert cache.get_cache("counter") == 0


def test_increment_cache_treats_non_int_cached_value_as_zero(cache):
    cache.set_cache("counter", "not-a-number")
    assert cache.increment_cache("counter", 4) == 4
    assert cache.get_cache("counter") == 4


def test_increment_cache_is_atomic_under_thread_concurrency(cache):
    seed = 1000
    cache.set_cache("counter", _SlowInt(seed))
    thread_count = 8
    barrier = threading.Barrier(thread_count)

    def increment(_: int) -> int:
        barrier.wait()
        return cache.increment_cache("counter", 1)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        tuple(executor.map(increment, range(thread_count)))

    assert cache.get_cache("counter") == seed + thread_count


async def test_async_increment_starts_from_zero_when_key_missing(cache):
    assert await cache.async_increment("counter", 2) == 2


async def test_async_increment_adds_to_existing_int(cache):
    await cache.async_set_cache("counter", 10)
    assert await cache.async_increment("counter", 5) == 15


async def test_async_increment_treats_non_int_cached_value_as_zero(cache):
    await cache.async_set_cache("counter", "corrupt")
    assert await cache.async_increment("counter", 9) == 9
