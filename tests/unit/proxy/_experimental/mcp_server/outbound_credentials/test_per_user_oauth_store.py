import asyncio

import httpx
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


@pytest.fixture
def credential_database(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy import proxy_server

    prisma = MagicMock()
    table = prisma.db.litellm_mcpusercredentials
    table.upsert = AsyncMock()
    table.find_unique = AsyncMock(return_value=None)
    table.delete_many = AsyncMock(return_value=1)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setenv("LITELLM_SALT_KEY", "credential-persistence-test-salt")
    return table


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes, expected", [(None, None), ((), []), (("read",), ["read"])])
async def test_credential_persistence_preserves_known_and_unknown_scopes(credential_database, scopes, expected):
    from litellm.proxy._experimental.mcp_server.db import decode_oauth_payload
    from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import _persist_credential

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
    credential_database,
    prior_scopes,
    response_scope,
    expected,
):
    from litellm.proxy._experimental.mcp_server.db import decode_oauth_payload
    from litellm.proxy._experimental.mcp_server.outbound_credentials.authz_code_refresher import (
        AuthorizationCodeRefresher,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import (
        _persist_credential,
        _purge_credential,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok
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
        return Ok({"access_token": "rotated", **({"scope": response_scope} if response_scope is not None else {})})

    refresher = AuthorizationCodeRefresher(
        lambda _: server, token_endpoint, _persist_credential, purge=_purge_credential
    )
    refreshed = await refresher.refresh("alice", "srv", OAuthToken("old", refresh_token="refresh", scopes=prior_scopes))
    assert refreshed is not None and refreshed.scopes == tuple(expected)
    payload = decode_oauth_payload(credential_database.upsert.call_args.kwargs["data"]["create"]["credential_b64"])
    assert payload is not None and payload.get("scopes") == expected
    assert payload["access_token"] == "rotated"


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


async def _stored_row(credential_database, refresh_token):
    from types import SimpleNamespace

    from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import _persist_credential

    await _persist_credential("alice", "srv", "expired-access", refresh_token, -60, ("read",))
    credential_b64 = credential_database.upsert.call_args.kwargs["data"]["create"]["credential_b64"]
    return SimpleNamespace(user_id="alice", server_id="srv", credential_b64=credential_b64)


def _refreshable_server():
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id="srv",
        name="srv",
        transport="http",
        auth_type="oauth2",
        oauth2_flow="authorization_code",
        token_url="https://idp.example/token",
        client_id="client",
        client_secret="secret",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["invalid_grant", "invalid_token"])
async def test_revoked_refresh_token_deletes_the_stored_credential(
    monkeypatch, respx_mock, credential_database, error_code
):
    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    row = await _stored_row(credential_database, "revoked-refresh")
    credential_database.find_unique.return_value = row
    upstream = respx_mock.post("https://idp.example/token").respond(400, json={"error": error_code})

    store = LazyPerUserOAuthTokenStore(lambda _: _refreshable_server(), redis_available=lambda: False)
    assert await store.fetch("alice", "srv") is None
    assert upstream.call_count == 1
    credential_database.delete_many.assert_awaited_once_with(
        where={"user_id": "alice", "server_id": "srv", "credential_b64": row.credential_b64}
    )


@pytest.mark.asyncio
async def test_revoked_refresh_completes_when_redis_comes_up_mid_fetch(monkeypatch, respx_mock, credential_database):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    credential_database.find_unique.return_value = await _stored_row(credential_database, "revoked-refresh")
    redis_available = _RedisAvailability()

    def revoke_while_redis_comes_up(_request: httpx.Request) -> httpx.Response:
        redis_available.available = True
        return httpx.Response(400, json={"error": "invalid_grant"})

    respx_mock.post("https://idp.example/token").mock(side_effect=revoke_while_redis_comes_up)
    store = LazyPerUserOAuthTokenStore(lambda _: _refreshable_server(), redis_available=redis_available)
    monkeypatch.setattr(global_mcp_server_manager, "_per_user_oauth_token_store", store)

    assert await asyncio.wait_for(store.fetch("alice", "srv"), timeout=5) is None
    credential_database.delete_many.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status_code": 503, "json": {"error": "temporarily_unavailable"}},
        {"status_code": 400, "json": {"error": "invalid_client"}},
        {"status_code": 200, "text": "not json"},
        {"status_code": 200, "json": ["not", "an", "object"]},
        {"status_code": 400, "json": ["invalid_grant"]},
        {"status_code": 400, "text": "invalid_grant"},
    ],
)
async def test_transient_refresh_failure_keeps_the_stored_credential_for_retry(
    monkeypatch, respx_mock, credential_database, response
):
    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    credential_database.find_unique.return_value = await _stored_row(credential_database, "live-refresh")
    upstream = respx_mock.post("https://idp.example/token").respond(**response)

    store = LazyPerUserOAuthTokenStore(lambda _: _refreshable_server(), redis_available=lambda: False)
    assert await store.fetch("alice", "srv") is None
    upstream.respond(200, json={"access_token": "recovered", "expires_in": 3600})
    recovered = await store.fetch("alice", "srv")
    assert recovered is not None and recovered.access_token == "recovered"
    assert upstream.call_count == 2
    credential_database.delete_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
async def test_unreachable_token_endpoint_keeps_the_stored_credential(
    monkeypatch, respx_mock, credential_database, transport_error
):
    from litellm.proxy._experimental.mcp_server.outbound_credentials import per_user_oauth_store as module

    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setattr(module, "_runtime_backend_and_coordinator", lambda: (None, None, False))
    credential_database.find_unique.return_value = await _stored_row(credential_database, "live-refresh")
    respx_mock.post("https://idp.example/token").mock(side_effect=transport_error)

    store = LazyPerUserOAuthTokenStore(lambda _: _refreshable_server(), redis_available=lambda: False)
    assert await store.fetch("alice", "srv") is None
    credential_database.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_purge_skips_a_credential_already_replaced_by_reauthorization(monkeypatch, credential_database):
    from litellm.proxy._experimental.mcp_server.outbound_credentials.per_user_oauth_store import _purge_credential

    credential_database.find_unique.return_value = await _stored_row(credential_database, "fresh-refresh")
    await _purge_credential("alice", "srv", "revoked-refresh")
    credential_database.delete_many.assert_not_awaited()
