import asyncio
from typing import Final
from unittest.mock import patch

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed

DB_SPEND: Final = 2777.16
COUNTER_KEY: Final = "spend:org:test-org"


@pytest.mark.asyncio
async def test_coalesced_no_redis_does_not_double_when_seeded_during_db_read():
    """
    Regression test for the cold-seed race (LIT-5516): with no Redis, if a
    concurrent task (e.g. reservation reconcile) seeds the in-memory counter
    while coalesced() is awaiting the DB read, the blind increment doubled
    the counter to 2x db_spend and falsely tripped budget enforcement.
    """
    cache: Final = DualCache(default_in_memory_ttl=60)
    db_read_started: Final = asyncio.Event()
    db_read_release: Final = asyncio.Event()

    async def slow_from_db(prisma_client, counter_key):
        db_read_started.set()
        await db_read_release.wait()
        return DB_SPEND

    async def seed_during_db_read():
        await db_read_started.wait()
        cache.in_memory_cache.set_cache(key=COUNTER_KEY, value=DB_SPEND)
        db_read_release.set()

    with patch.object(SpendCounterReseed, "from_db", side_effect=slow_from_db):
        result, _ = await asyncio.gather(
            SpendCounterReseed.coalesced(
                prisma_client=None,
                spend_counter_cache=cache,
                counter_key=COUNTER_KEY,
            ),
            seed_during_db_read(),
        )

    assert result == DB_SPEND
    final_value: Final = cache.in_memory_cache.get_cache(key=COUNTER_KEY)
    assert float(final_value) == DB_SPEND


@pytest.mark.asyncio
async def test_coalesced_no_redis_seeds_cold_counter():
    cache: Final = DualCache(default_in_memory_ttl=60)

    async def from_db(prisma_client, counter_key):
        return DB_SPEND

    with patch.object(SpendCounterReseed, "from_db", side_effect=from_db):
        result: Final = await SpendCounterReseed.coalesced(
            prisma_client=None,
            spend_counter_cache=cache,
            counter_key=COUNTER_KEY,
        )

    assert result == DB_SPEND
    assert float(cache.in_memory_cache.get_cache(key=COUNTER_KEY)) == DB_SPEND
