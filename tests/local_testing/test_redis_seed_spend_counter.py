"""The spend-counter seed decides between set, add and keep inside a Lua script, so only a real
Redis can show the script itself is wrong."""

import asyncio
import os
import uuid
from typing import Final

import pytest
from dotenv import load_dotenv

load_dotenv()

import litellm
from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def counter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "default_redis_ttl", 600)
    cache: Final = RedisCache(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))
    key: Final = f"spend:user:seed-{uuid.uuid4()}"
    yield cache, key, cache.check_and_fix_namespace(key=key)
    cache.delete_cache(key)


def test_a_cold_counter_is_seeded_with_the_base_and_gets_an_expiry(counter):
    cache, key, namespaced_key = counter

    assert asyncio.run(cache.async_seed_spend_counter(key=key, base=6.0)) == 6.0
    assert 0 < cache.redis_client.ttl(namespaced_key) <= 600


def test_concurrent_seeds_count_the_base_once(counter):
    cache, key, _ = counter

    async def seed_five_times() -> tuple[float, ...]:
        return tuple(await asyncio.gather(*(cache.async_seed_spend_counter(key=key, base=6.0) for _ in range(5))))

    assert asyncio.run(seed_five_times()) == (6.0,) * 5


def test_spend_that_landed_before_the_seed_is_kept_on_top_of_the_base(counter):
    cache, key, namespaced_key = counter
    cache.redis_client.incrbyfloat(namespaced_key, 0.5)

    assert asyncio.run(cache.async_seed_spend_counter(key=key, base=6.0)) == 6.5


def test_a_counter_already_at_or_above_the_base_is_left_alone(counter):
    cache, key, namespaced_key = counter
    cache.redis_client.set(namespaced_key, 9.5)

    assert asyncio.run(cache.async_seed_spend_counter(key=key, base=6.0)) == 9.5
