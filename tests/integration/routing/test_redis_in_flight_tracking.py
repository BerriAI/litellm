import os
import time
import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from litellm.caching.redis_cache import RedisCache

MAX_AGE: Final = 600


@pytest.fixture
def in_flight() -> Iterator[tuple[RedisCache, str, str]]:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    key: Final = f"in-flight-{uuid.uuid4()}"
    yield cache, key, cache.check_and_fix_namespace(key=key)
    cache.delete_cache(key)


def test_requests_finishing_after_newer_ones_started_leave_the_newer_ones_counted(
    in_flight: tuple[RedisCache, str, str],
) -> None:
    cache, key, _ = in_flight
    now: Final = time.time()

    for call_id in ("old-1", "old-2", "old-3", "new-1", "new-2", "new-3"):
        cache.track_in_flight(key, call_id, now, MAX_AGE)
    for call_id in ("old-1", "old-2", "old-3"):
        cache.untrack_in_flight(key, call_id)
    cache.track_in_flight(key, "new-4", now, MAX_AGE)

    assert cache.batch_count_in_flight([key], since=now - MAX_AGE) == (4,)


def test_a_request_older_than_max_age_ages_out_without_touching_the_rest(
    in_flight: tuple[RedisCache, str, str],
) -> None:
    cache, key, namespaced_key = in_flight
    now: Final = time.time()

    cache.track_in_flight(key, "orphan", now - MAX_AGE - 1, MAX_AGE)
    cache.track_in_flight(key, "running", now - MAX_AGE + 60, MAX_AGE)
    cache.track_in_flight(key, "new", now, MAX_AGE)

    assert cache.batch_count_in_flight([key], since=now - MAX_AGE) == (2,)
    assert cache.redis_client.zcard(namespaced_key) == 2
    assert 0 < cache.redis_client.ttl(namespaced_key) <= MAX_AGE


@pytest.mark.asyncio
async def test_the_async_twins_share_the_same_state(in_flight: tuple[RedisCache, str, str]) -> None:
    cache, key, _ = in_flight
    now: Final = time.time()

    await cache.async_track_in_flight(key, "async-1", now, MAX_AGE)
    cache.track_in_flight(key, "sync-1", now, MAX_AGE)
    await cache.async_untrack_in_flight(key, "sync-1")

    assert await cache.async_batch_count_in_flight([key], since=now - MAX_AGE) == (1,)
