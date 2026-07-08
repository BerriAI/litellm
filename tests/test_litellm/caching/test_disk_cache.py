import pytest

pytest.importorskip("diskcache")

from litellm.caching.disk_cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache(disk_cache_dir=str(tmp_path))


def test_increment_cache_starts_from_zero_when_key_missing(cache):
    assert cache.increment_cache("counter", 3) == 3
    assert cache.get_cache("counter") == 3


def test_increment_cache_adds_to_existing_int(cache):
    cache.set_cache("counter", 7)
    assert cache.increment_cache("counter", 5) == 12
    assert cache.get_cache("counter") == 12


def test_increment_cache_treats_non_int_cached_value_as_zero(cache):
    cache.set_cache("counter", "not-a-number")
    assert cache.increment_cache("counter", 4) == 4
    assert cache.get_cache("counter") == 4


async def test_async_increment_starts_from_zero_when_key_missing(cache):
    assert await cache.async_increment("counter", 2) == 2


async def test_async_increment_adds_to_existing_int(cache):
    await cache.async_set_cache("counter", 10)
    assert await cache.async_increment("counter", 5) == 15


async def test_async_increment_treats_non_int_cached_value_as_zero(cache):
    await cache.async_set_cache("counter", "corrupt")
    assert await cache.async_increment("counter", 9) == 9
