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


@pytest.mark.asyncio
async def test_enforcement_invalidates_cached_legacy_credentials_before_use():
    from litellm.types.mcp import MCPAuth, MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPOAuthIdentityBinding, MCPServer

    server = MCPServer(
        server_id="srv",
        name="srv",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth_identity_binding=MCPOAuthIdentityBinding(
            mode="enforce",
            issuer="https://idp.example.com",
            audiences=["client"],
        ),
    )
    cached = _RecordingStore("belongs-to-bob")

    store = LazyPerUserOAuthTokenStore(
        lambda server_id: server,
        store_builder=lambda lookup: (cached, False),
        redis_available=lambda: False,
    )
    assert await store.fetch("alice", "srv") is None
    assert cached.calls == [("alice", "srv")]
    assert cached.invalidations == [("alice", "srv")]


@pytest.mark.asyncio
async def test_enforced_cache_hit_avoids_credential_read_and_rejects_changed_policy(monkeypatch):
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.oauth_identity_binding import current_binding_proof
    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv",
        name="srv",
        transport="http",
        auth_type="oauth2",
        oauth_identity_binding={
            "mode": "enforce",
            "issuer": "https://idp.example",
            "audiences": ["client"],
            "caller_field": "user_id",
            "principal_claim": "sub",
        },
    )
    proof = await current_binding_proof(server.oauth_identity_binding, "alice", "srv")
    read = AsyncMock(return_value={"access_token": "alice-token", "identity_binding_proof": proof})
    monkeypatch.setattr(module, "_read_credential", read)
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    store = LazyPerUserOAuthTokenStore(lambda _: server, redis_available=lambda: False)
    assert (await store.fetch("alice", "srv")).access_token == "alice-token"
    assert (await store.fetch("alice", "srv")).access_token == "alice-token"
    read.assert_awaited_once_with("alice", "srv")
    server.oauth_identity_binding = server.oauth_identity_binding.model_copy(update={"audiences": ["changed"]})
    assert await store.fetch("alice", "srv") is None
    read.assert_awaited_once()
    assert await store.fetch("alice", "srv") is None
    assert read.await_count == 2


@pytest.mark.asyncio
async def test_expired_unverified_credential_never_reaches_refresh(monkeypatch):
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv",
        name="srv",
        transport="http",
        auth_type="oauth2",
        token_url="https://idp.example/token",
        oauth_identity_binding={"mode": "enforce", "issuer": "https://idp.example", "audiences": ["client"]},
    )
    read = AsyncMock(
        return_value={"access_token": "bob", "refresh_token": "bob-refresh", "expires_at": "2000-01-01T00:00:00Z"}
    )
    post = AsyncMock()
    monkeypatch.setattr(module, "_read_credential", read)
    monkeypatch.setattr(module, "_post_token_endpoint", post)
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    store = LazyPerUserOAuthTokenStore(lambda _: server, redis_available=lambda: False)
    assert await store.fetch("alice", "srv") is None
    post.assert_not_awaited()
