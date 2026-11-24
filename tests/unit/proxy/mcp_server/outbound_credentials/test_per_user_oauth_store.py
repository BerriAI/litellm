import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final

import pytest

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    InvalidatableOAuthTokenStore,
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
    ServerLookup,
)

if TYPE_CHECKING:
    from respx import MockRouter


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


@pytest.fixture
def credential_database(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy import proxy_server

    prisma = MagicMock()
    table = prisma.db.litellm_mcpusercredentials
    table.upsert = AsyncMock()
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setenv("LITELLM_SALT_KEY", "credential-persistence-test-salt")
    return table


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes, expected", [(None, None), ((), []), (("read",), ["read"])])
async def test_credential_persistence_preserves_known_and_unknown_scopes(credential_database, scopes, expected):
    from litellm.proxy.mcp_server.db import decode_oauth_payload
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _persist_credential

    await _persist_credential("alice", "srv", "access", "refresh", 60, scopes)
    row = credential_database.upsert.call_args.kwargs["data"]["create"]
    payload = decode_oauth_payload(row["credential_b64"])
    assert payload is not None and payload.get("scopes") == expected
    assert row["user_id"] == "alice" and row["server_id"] == "srv"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prior_scopes, response_scope, expected",
    [((), None, []), (("read", "write"), "", []), (("read", "write"), None, ["read", "write"])],
)
async def test_native_refresh_persists_actual_grant_through_storage_adapter(
    credentials_prisma: SimpleNamespace,
    prior_scopes,
    response_scope,
    expected,
):
    from litellm.proxy.mcp_server.db import (
        get_user_oauth_credential,
        record_oauth_scope_refusal,
        store_user_oauth_credential,
    )
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeRefresher
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _persist_credential
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv",
        name="srv",
        transport="http",
        auth_type="oauth2",
        token_url="https://idp.example/token",
        client_id="client",
        client_secret="secret",
    )

    async def token_endpoint(url, form, headers):
        assert url == server.token_url and form["grant_type"] == "refresh_token"
        return {"access_token": "rotated", **({"scope": response_scope} if response_scope is not None else {})}

    await store_user_oauth_credential(credentials_prisma, "alice", "srv", "old", refresh_token="refresh")
    await record_oauth_scope_refusal(credentials_prisma, "alice", "srv", "gist")
    refresher = AuthorizationCodeRefresher(lambda _: server, token_endpoint, _persist_credential)
    refreshed = await refresher.refresh("alice", "srv", OAuthToken("old", refresh_token="refresh", scopes=prior_scopes))
    assert refreshed is not None and refreshed.scopes == tuple(expected)
    payload = await get_user_oauth_credential(credentials_prisma, "alice", "srv")
    assert payload is not None and payload.get("scopes") == expected
    assert payload["access_token"] == "rotated"
    assert payload["needs_scopes"] == ["gist"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, body, marked",
    [
        (400, {"error": "invalid_grant"}, True),
        (401, {"error": "invalid_grant"}, True),
        (400, {"error": "invalid_client"}, False),
        (200, {"error": "server_error"}, False),
        (400, {"error": "invalid_request"}, True),
        (401, {"error_description": "bad credentials"}, True),
        (503, {"error": "invalid_grant"}, False),
        (401, {"error": "invalid_grant", "access_token": "unusable", "refresh_token": "unusable"}, False),
        (400, {"access_token": "unusable"}, False),
    ],
)
async def test_a_refused_refresh_asks_for_sign_in_without_calling_the_provider_again(
    credentials_prisma: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: "MockRouter",
    status: int,
    body: Mapping[str, object],
    marked: bool,
) -> None:
    import httpx

    from litellm.proxy.mcp_server.db import get_user_oauth_credential, oauth_grant_state, store_user_oauth_credential
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import build_per_user_oauth_token_store
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    server: Final = MCPServer(
        server_id="srv",
        name="srv",
        transport="http",
        auth_type="oauth2",
        token_url="https://idp.example/token",
        client_id="client",
        client_secret="secret",
    )
    route: Final = respx_mock.post("https://idp.example/token").mock(return_value=httpx.Response(status, json=body))
    await store_user_oauth_credential(credentials_prisma, "alice", "srv", "old", refresh_token="dead", expires_in=-60)
    store: Final = build_per_user_oauth_token_store(lambda _: server)

    assert await store.fetch("alice", "srv") is None
    assert await store.fetch("alice", "srv") is None
    saved: Final = await get_user_oauth_credential(credentials_prisma, "alice", "srv")
    assert oauth_grant_state(saved) == ("absent" if marked else "refreshable")
    assert saved is not None and saved["access_token"] == "old", "an error response never replaces the saved login"
    assert route.call_count == (1 if marked else 2), "a refused refresh token must not be presented again"


@pytest.mark.asyncio
@pytest.mark.parametrize("database", ["down", "absent"])
async def test_a_refusal_that_cannot_be_recorded_still_asks_for_sign_in(
    monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeRefresher
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import (
        _persist_credential,
        _record_refresh_refusal,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    table: Final = SimpleNamespace(find_unique=AsyncMock(side_effect=RuntimeError("database down")))
    unreachable: Final = SimpleNamespace(db=SimpleNamespace(litellm_mcpusercredentials=table), writer_db=None)
    monkeypatch.setattr(proxy_server, "prisma_client", unreachable if database == "down" else None)
    server: Final = MCPServer(
        server_id="srv", name="srv", transport="http", auth_type="oauth2", token_url="https://idp.example/token"
    )

    async def refused(url: str, form: Mapping[str, str], headers: Mapping[str, str]) -> Mapping[str, object]:
        return {"error": "invalid_grant"}

    refresher: Final = AuthorizationCodeRefresher(
        lambda _: server, refused, _persist_credential, record_refusal=_record_refresh_refusal
    )
    token: Final = await refresher.refresh("alice", "srv", OAuthToken("old", refresh_token="dead"))

    assert token is None, "a refusal must still end in the sign-in challenge, never a 500"


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

    from litellm.proxy.mcp_server.oauth_identity_binding import current_binding_proof
    from litellm.proxy.mcp_server.outbound_credentials import per_user_oauth_store as module
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

    from litellm.proxy.mcp_server.outbound_credentials import per_user_oauth_store as module
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
