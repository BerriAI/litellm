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


def test_traffic_refreshes_a_counters_expiry(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    assert cache.redis_client.ttl(namespaced_key) > TTL - 60

    cache.redis_client.expire(namespaced_key, 30)
    cache.increment_with_floor(key, 1, TTL)

    assert cache.redis_client.ttl(namespaced_key) > 30


def test_clamping_to_zero_deletes_the_counter(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    cache.redis_client.expire(namespaced_key, 30)

    assert cache.increment_with_floor(key, -5, TTL) == 0
    assert cache.redis_client.exists(namespaced_key) == 0


@pytest.mark.asyncio
async def test_the_async_counter_behaves_the_same_way(counter: tuple[RedisCache, str, str]) -> None:
    cache, key, namespaced_key = counter

    assert await cache.async_increment_with_floor(key, 2, TTL) == 2
    assert await cache.async_batch_get_counts([key]) == (2,)

    cache.redis_client.expire(namespaced_key, 30)
    assert await cache.async_increment_with_floor(key, 1, TTL) == 3
    assert cache.redis_client.ttl(namespaced_key) > 30

    assert await cache.async_increment_with_floor(key, -9, TTL) == 0
    assert cache.redis_client.exists(namespaced_key) == 0


def test_ongoing_traffic_keeps_an_in_flight_counter_from_expiring(
    counter: tuple[RedisCache, str, str],
) -> None:
    """Under load, #48994's fixed-from-creation TTL let the counter expire while
    requests were still in flight. Refreshing the TTL on every touch keeps the
    count aligned with the real in-flight set for as long as the deployment is busy.
    """
    import time

    cache, key, namespaced_key = counter
    short_ttl = 2

    for _ in range(3):
        cache.increment_with_floor(key, 1, short_ttl)
    assert cache.redis_client.get(namespaced_key) == b"3"

    # Keep touching for longer than the TTL. Old behavior would expire the key and
    # the next +1/-1 pair would rebuild from zero; new behavior keeps the count at 3.
    deadline = time.monotonic() + short_ttl + 1.0
    while time.monotonic() < deadline:
        cache.increment_with_floor(key, 1, short_ttl)
        cache.increment_with_floor(key, -1, short_ttl)
        time.sleep(0.2)

    assert cache.redis_client.get(namespaced_key) == b"3"
    assert cache.redis_client.ttl(namespaced_key) > 0
