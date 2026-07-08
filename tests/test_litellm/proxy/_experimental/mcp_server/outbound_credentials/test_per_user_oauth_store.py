import asyncio

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    OAuthTokenStore,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
    ServerLookup,
)


class _RecordingStore:
    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self.calls: list[tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        self.calls.append((user_id, server_id))
        return OAuthToken(access_token=self._access_token)


class _BlockingStore:
    def __init__(self, access_token: str) -> None:
        self._access_token = access_token
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        self.calls.append((user_id, server_id))
        self.started.set()
        await self.release.wait()
        return OAuthToken(access_token=self._access_token)


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

    def build_store(_server_lookup: ServerLookup) -> tuple[OAuthTokenStore, bool]:
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

    def build_store(_server_lookup: ServerLookup) -> tuple[OAuthTokenStore, bool]:
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

    def build_store(_server_lookup: ServerLookup) -> tuple[OAuthTokenStore, bool]:
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


class _MutableSourceStore:
    """DB stand-in whose stored credential can change underneath the cache (re-auth / revoke)."""

    def __init__(self, token: OAuthToken | None) -> None:
        self.token = token

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        return self.token


@pytest.mark.asyncio
async def test_invalidate_drops_cached_token_after_credential_change() -> None:
    """Revoke/re-auth regression: the cache serves the old token until TTL, so after the stored
    credential changes the egress keeps using the revoked token (upstream 401s) unless the entry
    is invalidated. ``invalidate`` must make the next fetch read the new stored credential."""
    source = _MutableSourceStore(OAuthToken(access_token="old"))

    def build_store(_server_lookup: ServerLookup) -> tuple[CachedOAuthTokenStore, bool]:
        return CachedOAuthTokenStore(source, default_ttl_seconds=300.0), False

    store = LazyPerUserOAuthTokenStore(
        lambda _server_id: None,
        store_builder=build_store,
        redis_available=lambda: False,
    )

    first = await store.fetch("u", "s")
    source.token = OAuthToken(access_token="new")
    stale = await store.fetch("u", "s")

    await store.invalidate("u", "s")
    fresh = await store.fetch("u", "s")

    assert first is not None and first.access_token == "old"
    assert stale is not None and stale.access_token == "old"
    assert fresh is not None and fresh.access_token == "new"


@pytest.mark.asyncio
async def test_invalidate_drops_cached_token_after_revoke() -> None:
    """After a revoke the stored credential is gone; a cached positive token must not outlive it."""
    source = _MutableSourceStore(OAuthToken(access_token="revoked-later"))

    def build_store(_server_lookup: ServerLookup) -> tuple[CachedOAuthTokenStore, bool]:
        return CachedOAuthTokenStore(source, default_ttl_seconds=300.0), False

    store = LazyPerUserOAuthTokenStore(
        lambda _server_id: None,
        store_builder=build_store,
        redis_available=lambda: False,
    )

    cached = await store.fetch("u", "s")
    source.token = None
    await store.invalidate("u", "s")
    after_revoke = await store.fetch("u", "s")

    assert cached is not None
    assert after_revoke is None
