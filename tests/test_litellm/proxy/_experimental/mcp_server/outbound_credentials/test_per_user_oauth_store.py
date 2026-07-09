import asyncio

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InvalidatableOAuthTokenStore,
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
    ServerLookup,
)


class _RecordingStore:
    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self.calls: list[tuple[str, str]] = []
        self.invalidations: list[tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        self.calls.append((user_id, server_id))
        return OAuthToken(access_token=self._access_token)

    async def invalidate(self, user_id: str, server_id: str) -> None:
        self.invalidations.append((user_id, server_id))


class _BlockingStore:
    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, str]] = []
        self.invalidations: list[tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        self.calls.append((user_id, server_id))
        self.started.set()
        await self.release.wait()
        return OAuthToken(access_token=self._access_token)

    async def invalidate(self, user_id: str, server_id: str) -> None:
        self.invalidations.append((user_id, server_id))


class _RedisAvailability:
    def __init__(self) -> None:
        self.available = False

    def __call__(self) -> bool:
        return self.available


async def _wait_for_call_count(store: _BlockingStore, count: int) -> None:
    for _ in range(100):
        if len(store.calls) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {count} calls, saw {len(store.calls)}")


@pytest.mark.asyncio
async def test_lazy_store_rebuilds_when_redis_becomes_available() -> None:
    local_store = _RecordingStore("local")
    redis_store = _RecordingStore("redis")
    redis_available = _RedisAvailability()
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        if redis_available.available:
            return redis_store, True
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=redis_available,
    )

    first = await store.fetch("u", "s")
    redis_available.available = True
    second = await store.fetch("u", "s")
    third = await store.fetch("u", "s")

    assert first is not None and first.access_token == "local"
    assert second is not None and second.access_token == "redis"
    assert third is not None and third.access_token == "redis"
    assert build_calls == 2
    assert local_store.calls == [("u", "s")]
    assert redis_store.calls == [("u", "s"), ("u", "s")]


@pytest.mark.asyncio
async def test_lazy_store_allows_concurrent_local_fetches_without_redis() -> None:
    local_store = _BlockingStore("local")
    redis_available = _RedisAvailability()
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=redis_available,
    )

    first_fetch = asyncio.create_task(store.fetch("u1", "s1"))
    second_fetch = asyncio.create_task(store.fetch("u2", "s2"))
    await asyncio.wait_for(_wait_for_call_count(local_store, 2), timeout=1)

    local_store.release.set()
    first, second = await asyncio.gather(first_fetch, second_fetch)

    assert first is not None and first.access_token == "local"
    assert second is not None and second.access_token == "local"
    assert build_calls == 1
    assert local_store.calls == [("u1", "s1"), ("u2", "s2")]


@pytest.mark.asyncio
async def test_lazy_store_waits_for_in_flight_local_fetch_before_redis_rebuild() -> None:
    local_store = _BlockingStore("local")
    redis_store = _RecordingStore("redis")
    redis_available = _RedisAvailability()

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        if redis_available.available:
            return redis_store, True
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=redis_available,
    )

    first_fetch = asyncio.create_task(store.fetch("u", "s"))
    await local_store.started.wait()

    redis_available.available = True
    second_fetch = asyncio.create_task(store.fetch("u", "s"))
    await asyncio.sleep(0)

    assert redis_store.calls == []

    local_store.release.set()
    first = await first_fetch
    second = await second_fetch

    assert first is not None and first.access_token == "local"
    assert second is not None and second.access_token == "redis"
    assert local_store.calls == [("u", "s")]
    assert redis_store.calls == [("u", "s")]


@pytest.mark.asyncio
async def test_lazy_store_invalidate_builds_chain_and_delegates() -> None:
    local_store = _RecordingStore("local")
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=_RedisAvailability(),
    )

    await store.invalidate("u", "s")

    assert build_calls == 1
    assert local_store.invalidations == [("u", "s")]


@pytest.mark.asyncio
async def test_lazy_store_invalidate_reaches_the_store_fetch_reads() -> None:
    local_store = _RecordingStore("local")
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        return local_store, False

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=_RedisAvailability(),
    )

    await store.fetch("u", "s")
    await store.invalidate("u", "s")

    assert build_calls == 1
    assert local_store.calls == [("u", "s")]
    assert local_store.invalidations == [("u", "s")]


@pytest.mark.asyncio
async def test_lazy_store_invalidate_works_after_redis_chain_is_built() -> None:
    redis_store = _RecordingStore("redis")
    redis_available = _RedisAvailability()
    redis_available.available = True
    build_calls = 0

    def build_store(_server_lookup: ServerLookup) -> tuple[InvalidatableOAuthTokenStore, bool]:
        nonlocal build_calls
        build_calls += 1
        return redis_store, True

    def server_lookup(_server_id: str) -> None:
        return None

    store = LazyPerUserOAuthTokenStore(
        server_lookup,
        store_builder=build_store,
        redis_available=redis_available,
    )

    await store.fetch("u", "s")
    await store.invalidate("u", "s")

    assert build_calls == 1
    assert redis_store.invalidations == [("u", "s")]
