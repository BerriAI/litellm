"""Least-busy routing keeps its in-flight counters in Redis, and the clamp at zero plus the
create-once TTL both live inside a Lua script. Nothing but a real Redis runs that script, so
these are the only tests that fail when the script itself is wrong."""

import os
import uuid
from typing import Final

import pytest
from dotenv import load_dotenv

load_dotenv()

from litellm.caching.redis_cache import RedisCache

TTL: Final = 600


@pytest.fixture
def counter():
    cache: Final = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    key: Final = f"lit7039-{uuid.uuid4()}"
    yield cache, key, cache.check_and_fix_namespace(key=key)
    cache.delete_cache(key)


def test_a_counter_adds_every_increment_and_reads_back_what_it_holds(counter):
    cache, key, _ = counter

    assert cache.increment_with_floor(key, 3, TTL) == 3
    assert cache.increment_with_floor(key, 2, TTL) == 5
    assert cache.batch_get_counts([key]) == (5,)


def test_a_decrement_past_zero_leaves_the_counter_at_zero(counter):
    """A worker whose counter expired mid-request decrements a key that is no longer there.
    Without the clamp that deployment reads negative, and least-busy pins every later request
    on it until the count climbs back to zero."""
    cache, key, _ = counter

    assert cache.increment_with_floor(key, 1, TTL) == 1
    assert cache.increment_with_floor(key, -5, TTL) == 0
    assert cache.batch_get_counts([key]) == (0,)


def test_traffic_never_pushes_a_counters_expiry_back_out(counter):
    """The TTL is what releases a count whose worker died mid-request. Rewriting it on every
    touch would keep that stuck count alive for as long as the group takes traffic."""
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    assert cache.redis_client.ttl(namespaced_key) > TTL - 60

    cache.redis_client.expire(namespaced_key, 30)
    cache.increment_with_floor(key, 1, TTL)

    assert cache.redis_client.ttl(namespaced_key) <= 30


def test_clamping_to_zero_keeps_the_expiry_it_already_had(counter):
    cache, key, namespaced_key = counter

    cache.increment_with_floor(key, 1, TTL)
    cache.redis_client.expire(namespaced_key, 30)

    assert cache.increment_with_floor(key, -5, TTL) == 0
    assert cache.redis_client.ttl(namespaced_key) <= 30


@pytest.mark.asyncio
async def test_the_async_counter_behaves_the_same_way(counter):
    cache, key, namespaced_key = counter

    assert await cache.async_increment_with_floor(key, 2, TTL) == 2
    assert await cache.async_batch_get_counts([key]) == (2,)

    cache.redis_client.expire(namespaced_key, 30)

    assert await cache.async_increment_with_floor(key, -9, TTL) == 0
    assert cache.redis_client.ttl(namespaced_key) <= 30
