import asyncio
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

from litellm.proxy.agent_endpoints.external_agents.cache import MetadataCache


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "mapped-user"])
async def test_metadata_caches_values_and_misses_until_ttl_expires(value: str | None) -> None:
    clock: Final = Mock(return_value=0)
    cache: Final = MetadataCache[str | None](clock=clock)
    load: Final = AsyncMock(return_value=value)
    assert await cache.get("actor", load) == value
    assert await cache.get("actor", load) == value
    assert load.await_count == 1
    clock.return_value = 21
    load.return_value = "updated-user"
    assert await cache.get("actor", load) == "updated-user"
    assert load.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_metadata_misses_share_one_database_read() -> None:
    cache: Final = MetadataCache[str]()
    load: Final = AsyncMock(return_value="source")
    values: Final = await asyncio.gather(*(cache.get("source", load) for _ in range(200)))
    assert tuple(values) == ("source",) * 200
    assert load.await_count == 1


@pytest.mark.asyncio
async def test_invalidation_does_not_let_an_inflight_read_repopulate_old_metadata() -> None:
    cache: Final = MetadataCache[str]()
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def old_value() -> str:
        started.set()
        await release.wait()
        return "old-policy"

    pending: Final = asyncio.create_task(cache.get("policy", old_value))
    await started.wait()
    cache.invalidate()
    updated: Final = AsyncMock(return_value="new-policy")
    assert await cache.get("policy", updated) == "new-policy"
    release.set()
    assert await pending == "old-policy"
    assert await cache.get("policy", updated) == "new-policy"
    assert updated.await_count == 1


@pytest.mark.asyncio
async def test_failed_metadata_read_is_not_cached() -> None:
    cache: Final = MetadataCache[str]()
    load: Final = AsyncMock(side_effect=[RuntimeError("database unavailable"), "recovered"])
    with pytest.raises(RuntimeError, match="database unavailable"):
        await cache.get("source", load)
    assert await cache.get("source", load) == "recovered"
    assert load.await_count == 2


@pytest.mark.asyncio
async def test_metadata_capacity_evicts_oldest_entry() -> None:
    cache: Final = MetadataCache[str](capacity=2)
    cache.put("one", "first")
    cache.put("two", "second")
    cache.put("three", "third")
    load: Final = AsyncMock(return_value="reloaded")
    assert await cache.get("three", load) == "third"
    assert await cache.get("one", load) == "reloaded"
    assert load.await_count == 1
