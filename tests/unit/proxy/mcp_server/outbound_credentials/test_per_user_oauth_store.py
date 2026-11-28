import asyncio
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    InvalidatableOAuthTokenStore,
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import (
    LazyPerUserOAuthTokenStore,
    ServerLookup,
    _decode_refresh_error,
    _encode_refresh_error,
)
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.outbound_credentials.types import CredError


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
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import (
        AuthorizationCodeRefresher,
        AuthorizationCodeTokenSuccess,
    )
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
        return AuthorizationCodeTokenSuccess(
            body={"access_token": "rotated", **({"scope": response_scope} if response_scope is not None else {})}
        )

    await store_user_oauth_credential(credentials_prisma, "alice", "srv", "old", refresh_token="refresh")
    await record_oauth_scope_refusal(credentials_prisma, "alice", "srv", "gist")
    refresher = AuthorizationCodeRefresher(lambda _: server, token_endpoint, _persist_credential)
    refreshed = await refresher.refresh("alice", "srv", OAuthToken("old", refresh_token="refresh", scopes=prior_scopes))
    assert isinstance(refreshed, Ok)
    assert refreshed.ok.scopes == tuple(expected)
    payload = await get_user_oauth_credential(credentials_prisma, "alice", "srv")
    assert payload is not None and payload.get("scopes") == expected
    assert payload["access_token"] == "rotated"
    assert payload["needs_scopes"] == ["gist"]


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

    async def refresh_now(
        self, user_id: str, server_id: str, failed_access_token: str
    ) -> Result[OAuthToken, CredError]:
        self.calls.append((user_id, server_id))
        self.started.set()
        await self.release.wait()
        return Ok(OAuthToken(access_token=self._access_token))

    async def peek(self, user_id: str, server_id: str) -> OAuthToken | None:
        return await self.fetch(user_id, server_id)

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
async def test_cancelled_local_fetch_still_blocks_the_redis_rebuild_until_it_finishes() -> None:
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
    first_fetch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_fetch

    redis_available.available = True
    second_fetch = asyncio.create_task(store.fetch("u", "s"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert redis_store.calls == []

    local_store.release.set()
    second = await second_fetch

    assert second is not None and second.access_token == "redis"
    assert redis_store.calls == [("u", "s")]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["refresh_now", "peek"])
async def test_cancelled_local_reactive_read_still_blocks_the_redis_rebuild_until_it_finishes(operation: str) -> None:
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

    refresh = asyncio.create_task(
        store.refresh_now("u", "s", "rejected") if operation == "refresh_now" else store.peek("u", "s")
    )
    await local_store.started.wait()
    refresh.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh

    redis_available.available = True
    fetch = asyncio.create_task(store.fetch("u", "s"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert redis_store.calls == []

    local_store.release.set()
    fetched = await fetch

    assert fetched is not None and fetched.access_token == "redis"
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


@pytest.mark.asyncio
async def test_token_endpoint_success_is_typed_with_the_json_body(monkeypatch):
    import httpx

    from litellm.llms.custom_httpx import http_handler
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeTokenSuccess
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _post_token_endpoint

    class Client:
        async def post(self, url, headers, data):
            request = httpx.Request("POST", url)
            return httpx.Response(200, json={"access_token": "new"}, request=request)

    monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda llm_provider: Client())
    outcome = await _post_token_endpoint("https://idp.example/token", {}, {})
    assert isinstance(outcome, AuthorizationCodeTokenSuccess)
    assert outcome.body == {"access_token": "new"}


@pytest.mark.asyncio
async def test_token_endpoint_http_denial_preserves_status_and_error_body(monkeypatch):
    import httpx

    from litellm.llms.custom_httpx import http_handler
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeTokenDenied
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _post_token_endpoint

    class Client:
        async def post(self, url, headers, data):
            request = httpx.Request("POST", url)
            response = httpx.Response(400, json={"error": "invalid_grant"}, request=request)
            response.raise_for_status()

    monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda llm_provider: Client())
    outcome = await _post_token_endpoint("https://idp.example/token", {}, {})
    assert isinstance(outcome, AuthorizationCodeTokenDenied)
    assert outcome.status_code == 400
    assert outcome.body == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_transport_failure_is_typed_unreachable(monkeypatch):
    from litellm.llms.custom_httpx import http_handler
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeTokenUnreachable
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _post_token_endpoint

    class Client:
        async def post(self, url, headers, data):
            raise TimeoutError("connect timed out")

    monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda llm_provider: Client())
    outcome = await _post_token_endpoint("https://idp.example/token", {}, {})
    assert isinstance(outcome, AuthorizationCodeTokenUnreachable)
    assert outcome.detail == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_body", [b"<html>error</html>", b'["not-an-object"]'])
async def test_token_endpoint_malformed_200_body_is_typed_denied(monkeypatch, raw_body: bytes):
    import httpx

    from litellm.llms.custom_httpx import http_handler
    from litellm.proxy.mcp_server.outbound_credentials.authz_code_refresher import AuthorizationCodeTokenDenied
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _post_token_endpoint

    class Client:
        async def post(self, url, headers, data):
            request = httpx.Request("POST", url)
            return httpx.Response(200, content=raw_body, request=request)

    monkeypatch.setattr(http_handler, "get_async_httpx_client", lambda llm_provider: Client())
    outcome = await _post_token_endpoint("https://idp.example/token", {}, {})
    assert isinstance(outcome, AuthorizationCodeTokenDenied)
    assert outcome.status_code == 200
    assert outcome.body == {}
    assert outcome.detail == "token endpoint returned a non-JSON-object body"


def test_redis_availability_reads_the_shared_usage_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.outbound_credentials.per_user_oauth_store import _redis_cache_is_available

    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    assert _redis_cache_is_available() is True
    monkeypatch.setattr(proxy_server, "redis_usage_cache", None)
    assert _redis_cache_is_available() is False


@pytest.mark.asyncio
async def test_assembled_store_shares_cached_tokens_through_redis_usage_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.outbound_credentials import per_user_oauth_store

    class _FakeRedisUsageCache:
        def __init__(self) -> None:
            self.data: dict[str, object] = {}

        async def async_get_cache(self, key: str) -> object:
            return self.data.get(key)

        async def async_set_cache(self, key: str, value: object, ttl: int | None = None) -> None:
            self.data[key] = value

        async def async_delete_cache(self, key: str) -> None:
            self.data.pop(key, None)

        def init_async_client(self) -> object:
            return object()

        def check_and_fix_namespace(self, key: str) -> str:
            return key

    redis_cache: Final = _FakeRedisUsageCache()
    db_reads: Final[list[tuple[str, str]]] = []

    async def _fake_read_credential(
        user_id: str,
        server_id: str,
    ) -> dict[str, object]:
        db_reads.append((user_id, server_id))
        return {"access_token": "shared-tok", "expires_at": 9999999999.0}

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key")
    monkeypatch.setattr(proxy_server, "redis_usage_cache", redis_cache)
    monkeypatch.setattr(per_user_oauth_store, "_read_credential", _fake_read_credential)

    store: Final = per_user_oauth_store.build_per_user_oauth_token_store(lambda _sid: None)
    first: Final = await store.fetch("user-1", "srv-1")
    second: Final = await store.fetch("user-1", "srv-1")

    assert first is not None and first.access_token == "shared-tok"
    assert second is not None and second.access_token == "shared-tok"
    assert db_reads == [("user-1", "srv-1")]
    assert len(redis_cache.data) == 1


def test_refresh_error_publication_roundtrips_the_exact_failure_class():
    for original in (
        Error(CredError.of_misconfigured("OAuth client authentication failed; check the server client configuration.")),
        Error(CredError.of_precondition_required("OAuth identity binding did not match.")),
    ):
        encoded: Final = _encode_refresh_error(original)
        assert encoded is not None
        decoded: Final = _decode_refresh_error(encoded)
        assert decoded == original


def test_refresh_error_publication_never_encodes_a_token():
    assert _encode_refresh_error(Ok(OAuthToken(access_token="secret-access-token"))) is None


@pytest.mark.asyncio
async def test_lazy_store_peek_enforces_identity_binding_and_invalidates_mismatch():
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv",
        name="srv",
        url="https://mcp.example/mcp",
        transport="http",
        auth_type="oauth2",
        oauth_identity_binding={"mode": "enforce", "issuer": "https://idp.example", "audiences": ["client"]},
    )
    invalidated: list[tuple[str, str]] = []

    class FakeReactiveStore:
        async def fetch(self, user_id: str, server_id: str):
            return OAuthToken(access_token="cached")

        async def fetch_result(self, user_id: str, server_id: str):
            return Ok(OAuthToken(access_token="cached"))

        async def peek(self, user_id: str, server_id: str):
            return OAuthToken(access_token="cached", identity_binding_proof=None)

        async def refresh_now(self, user_id: str, server_id: str, failed_access_token: str):
            return Error(CredError.of_unauthorized("expired"))

        async def invalidate(self, user_id: str, server_id: str):
            invalidated.append((user_id, server_id))

    store = LazyPerUserOAuthTokenStore(
        lambda _: server,
        store_builder=lambda _: (FakeReactiveStore(), False),
        redis_available=lambda: False,
    )
    assert await store.peek("alice", "srv") is None
    assert invalidated == [("alice", "srv")]
