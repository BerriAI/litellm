import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from litellm.caching.redis_cache import RedisCache

TTL: Final = 600


@pytest.fixture
def counter() -> Iterator[tuple[RedisCache, str, str]]:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    key: Final = f"increment-with-floor-{uuid.uuid4()}"
    yield cache, key, cache.check_and_fix_namespace(key=key)
    cache.delete_cache(key)


def test_a_counter_adds_every_increment_and_reads_back_what_it_holds(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, _ = counter

    assert cache.increment_with_floor(key, 3, TTL) == 3
    assert cache.increment_with_floor(key, 2, TTL) == 5
    assert cache.batch_get_counts([key]) == (5,)


def test_a_decrement_past_zero_leaves_the_counter_at_zero(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, _ = counter

    assert cache.increment_with_floor(key, 1, TTL) == 1
    assert cache.increment_with_floor(key, -5, TTL) == 0
    assert cache.batch_get_counts([key]) == (0,)


def test_traffic_never_pushes_a_counters_expiry_back_out(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    assert cache.redis_client.ttl(namespaced_key) > TTL - 60

    cache.redis_client.expire(namespaced_key, 30)
    cache.increment_with_floor(key, 1, TTL)

    assert cache.redis_client.ttl(namespaced_key) <= 30


def test_clamping_to_zero_keeps_the_expiry_it_already_had(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    cache.redis_client.expire(namespaced_key, 30)

    assert cache.increment_with_floor(key, -5, TTL) == 0
    assert cache.redis_client.ttl(namespaced_key) <= 30


@pytest.mark.asyncio
async def test_the_async_counter_behaves_the_same_way(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    assert await cache.async_increment_with_floor(key, 2, TTL) == 2
    assert await cache.async_batch_get_counts([key]) == (2,)

    cache.redis_client.expire(namespaced_key, 30)

    assert await cache.async_increment_with_floor(key, -9, TTL) == 0
    assert cache.redis_client.ttl(namespaced_key) <= 30
