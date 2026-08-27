import asyncio
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed

DB_SPEND: Final = 2777.16
COUNTER_KEY: Final = "spend:org:test-org"


class _OrgRow:
    spend: Final = DB_SPEND


class _FakeOrgTable:
    def __init__(self, read_started: asyncio.Event, read_release: asyncio.Event):
        self.read_started: Final = read_started
        self.read_release: Final = read_release

    async def find_unique(self, where: dict[str, str]) -> _OrgRow:
        self.read_started.set()
        await self.read_release.wait()
        return _OrgRow()


class _FakeDB:
    def __init__(self, table: _FakeOrgTable):
        self.litellm_organizationtable: Final = table


class _FakePrisma:
    def __init__(self, table: _FakeOrgTable):
        self.db: Final = _FakeDB(table)


def _fake_prisma(read_started: asyncio.Event, read_release: asyncio.Event) -> _FakePrisma:
    return _FakePrisma(_FakeOrgTable(read_started=read_started, read_release=read_release))


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

    async def seed_during_db_read():
        await db_read_started.wait()
        cache.in_memory_cache.set_cache(key=COUNTER_KEY, value=DB_SPEND)
        db_read_release.set()

    result, _ = await asyncio.gather(
        SpendCounterReseed.coalesced(
            prisma_client=_fake_prisma(db_read_started, db_read_release),
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
    db_read_started: Final = asyncio.Event()
    db_read_release: Final = asyncio.Event()
    db_read_release.set()

    result: Final = await SpendCounterReseed.coalesced(
        prisma_client=_fake_prisma(db_read_started, db_read_release),
        spend_counter_cache=cache,
        counter_key=COUNTER_KEY,
    )

    assert result == DB_SPEND
    assert float(cache.in_memory_cache.get_cache(key=COUNTER_KEY)) == DB_SPEND
