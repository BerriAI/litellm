import asyncio
from typing import Final

import pytest
from mcp.types import ListToolsResult

from litellm.proxy._experimental.mcp_server.tools_list_cache import (
    SharedToolsListCache,
    ToolsListStore,
    tools_list_cache_ttl,
)


class MemoryStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.reads = 0
        self.writes = 0

    async def get(self, key: str) -> str | None:
        self.reads += 1
        return self.values.get(key)

    async def set(self, key: str, value: str, ttl_seconds: float) -> None:
        self.writes += 1
        self.values[key] = value

    async def set_nx(self, key: str, value: str, ttl_seconds: float) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def incr(self, key: str) -> int:
        value: Final = int(self.values.get(key, "0")) + 1
        self.values[key] = str(value)
        return value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FetchCounter:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, ttl_ms: int | None = None) -> ListToolsResult:
        self.calls += 1
        return ListToolsResult(tools=[]) if ttl_ms is None else ListToolsResult(tools=[], ttl_ms=ttl_ms)


class RefreshFetch:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self) -> ListToolsResult:
        self.calls += 1
        if self.calls > 1:
            self.started.set()
            await self.release.wait()
        return ListToolsResult(tools=[])


class FailOnceFetch:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> ListToolsResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("upstream unavailable")
        return ListToolsResult(tools=[])


class WriteNotifyingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.written = asyncio.Event()

    async def set(self, key: str, value: str, ttl_seconds: float) -> None:
        await super().set(key, value, ttl_seconds)
        self.written.set()


@pytest.mark.asyncio
async def test_shared_cache_coalesces_two_instances_and_honors_explicit_upstream_ttl() -> None:
    store: Final = MemoryStore()
    clock: Final = Clock(10)
    first_cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    second_cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    fetches: Final = FetchCounter()

    first: Final = await first_cache.get("server", "digest", "full", lambda: fetches.fetch(1000))
    second: Final = await second_cache.get("server", "digest", "full", lambda: fetches.fetch(1000))
    assert first is not second
    assert fetches.calls == 1

    clock.value = 12.1
    await first_cache.get("server", "digest", "full", lambda: fetches.fetch(1000))
    assert fetches.calls == 2


@pytest.mark.asyncio
async def test_zero_ttl_bypasses_store_and_explicit_upstream_zero_is_not_cached() -> None:
    store: Final = MemoryStore()
    cache: Final = SharedToolsListCache(lambda: store, lambda: 0)
    fetches: Final = FetchCounter()

    await cache.get("server", "digest", "full", fetches.fetch)
    await cache.get("server", "digest", "full", fetches.fetch)
    assert fetches.calls == 2
    assert store.reads == 0
    assert store.writes == 0

    enabled_cache: Final = SharedToolsListCache(lambda: store, lambda: 10)

    await enabled_cache.get("server", "digest", "full", lambda: fetches.fetch(0))
    await enabled_cache.get("server", "digest", "full", lambda: fetches.fetch(0))
    assert fetches.calls == 4
    assert store.writes == 0


@pytest.mark.asyncio
async def test_store_failure_degrades_to_uncached_fetch() -> None:
    class FailingStore(MemoryStore):
        async def get(self, key: str) -> str | None:
            raise RuntimeError("store unavailable")

    store: Final = FailingStore()
    cache: Final = SharedToolsListCache(lambda: store, lambda: 10)
    fetches: Final = FetchCounter()

    await asyncio.gather(
        cache.get("server", "digest", "full", fetches.fetch),
        cache.get("server", "digest", "full", fetches.fetch),
    )
    assert fetches.calls == 2


@pytest.mark.asyncio
async def test_missing_upstream_ttl_uses_configured_ttl() -> None:
    store: Final[ToolsListStore] = MemoryStore()
    clock: Final = Clock(1)
    cache: Final = SharedToolsListCache(lambda: store, lambda: 2, clock)
    fetches: Final = FetchCounter()

    await cache.get("server", "digest", "full", fetches.fetch)
    await cache.get("server", "digest", "full", fetches.fetch)
    assert fetches.calls == 1
    clock.value = 5.1
    await cache.get("server", "digest", "full", fetches.fetch)
    assert fetches.calls == 2


@pytest.mark.asyncio
async def test_upstream_failure_is_not_cached_and_expired_stale_entry_fetches_synchronously() -> None:
    store: Final = MemoryStore()
    clock: Final = Clock(100)
    cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    fetch: Final = FailOnceFetch()

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await cache.get("server", "digest", "full", fetch)
    await cache.get("server", "digest", "full", fetch)
    assert fetch.calls == 2

    clock.value = 120
    await cache.get("server", "digest", "full", fetch)
    assert fetch.calls == 3


@pytest.mark.asyncio
async def test_invalidation_and_configuration_changes_force_cache_misses() -> None:
    store: Final = MemoryStore()
    cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock=lambda: 100)
    fetches: Final = FetchCounter()

    await cache.get("server", "first-digest", "full", fetches.fetch)
    await cache.get("server", "first-digest", "full", fetches.fetch)
    await cache.get("server", "changed-digest", "full", fetches.fetch)
    assert fetches.calls == 2

    cache.invalidate("server")
    await cache.get("server", "changed-digest", "full", fetches.fetch)
    assert fetches.calls == 3


@pytest.mark.asyncio
async def test_store_write_failure_does_not_fail_listing_or_create_a_hit() -> None:
    class FailingWriteStore(MemoryStore):
        async def set(self, key: str, value: str, ttl_seconds: float) -> None:
            raise RuntimeError("store unavailable")

    store: Final = FailingWriteStore()
    cache: Final = SharedToolsListCache(lambda: store, lambda: 10)
    fetches: Final = FetchCounter()

    first: Final = await cache.get("server", "digest", "full", fetches.fetch)
    second: Final = await cache.get("server", "digest", "full", fetches.fetch)

    assert first.tools == []
    assert second.tools == []
    assert fetches.calls == 2


@pytest.mark.asyncio
async def test_stale_refresh_is_single_flight_across_concurrent_cache_instances() -> None:
    store: Final = WriteNotifyingStore()
    clock: Final = Clock(100)
    first_cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    second_cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    fetch: Final = RefreshFetch()
    await first_cache.get("server", "digest", "full", fetch)

    store.written.clear()
    clock.value = 109
    results: Final = await asyncio.gather(
        *(cache.get("server", "digest", "full", fetch) for cache in (first_cache, second_cache) for _ in range(10))
    )
    await asyncio.wait_for(fetch.started.wait(), timeout=2)
    assert all(result.tools == [] for result in results)
    assert fetch.calls == 2

    fetch.release.set()
    await asyncio.wait_for(store.written.wait(), timeout=2)
    await asyncio.sleep(0)
    assert store.writes == 2


@pytest.mark.asyncio
async def test_background_refresh_failure_keeps_serving_the_stale_entry() -> None:
    class FailingRefreshFetch:
        def __init__(self) -> None:
            self.calls = 0
            self.failed = asyncio.Event()

        async def __call__(self) -> ListToolsResult:
            self.calls += 1
            if self.calls > 1:
                self.failed.set()
                raise RuntimeError("refresh unavailable")
            return ListToolsResult(tools=[])

    store: Final = MemoryStore()
    clock: Final = Clock(100)
    cache: Final = SharedToolsListCache(lambda: store, lambda: 10, clock)
    fetch: Final = FailingRefreshFetch()

    await cache.get("server", "digest", "full", fetch)
    clock.value = 109
    stale: Final = await cache.get("server", "digest", "full", fetch)
    await asyncio.wait_for(fetch.failed.wait(), timeout=2)
    await asyncio.sleep(0)
    still_stale: Final = await cache.get("server", "digest", "full", fetch)

    assert stale.tools == []
    assert still_stale.tools == []
    assert fetch.calls == 2
    await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("ui_value", "env_value", "expected"),
    (
        (5, "10", 5.0),
        (0, "10", 0.0),
        (None, "10", 10.0),
        (None, None, 0.0),
        (-1, "10", 0.0),
        ("nan", "10", 0.0),
    ),
)
def test_cache_ttl_prefers_ui_value_and_disables_invalid_values(
    ui_value: object | None, env_value: str | None, expected: float
) -> None:
    assert tools_list_cache_ttl(ui_value, env_value) == expected
