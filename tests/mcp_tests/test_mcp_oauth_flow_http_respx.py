"""
HTTP-level integration tests for MCP OAuth.

Covers two layers:
1. Management broker endpoints (mcp_management_endpoints.py) — the routes
   actually modified by the auth-gate regression fix. These tests use
   ASGITransport to hit the management router directly and assert that
   /server/oauth/{id}/authorize and /server/oauth/{id}/token do NOT
   require an API key (regression: they previously returned 401 for browsers).

2. Discoverable OAuth router (discoverable_endpoints.py) — end-to-end
   authorize → callback → token flow mocked with respx.
"""

from __future__ import annotations

import urllib.parse
from typing import Iterator
from unittest.mock import patch

import httpx
import litellm
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from litellm.types.mcp import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def mock_mcp_client_ip() -> Iterator[None]:
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


@pytest.fixture
def oauth_asgi_app(monkeypatch) -> Iterator[FastAPI]:
    monkeypatch.setenv("LITELLM_SALT_KEY", "integration-test-salt-key-32chars")
    # Outbound token exchange must use httpx so respx can intercept (not aiohttp).
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        router as discoverable_oauth_router,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    server = MCPServer(
        server_id="mock-oauth-srv",
        name="mock_oauth",
        server_name="mock_oauth",
        alias="mock_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="upstream-client",
        client_secret="upstream-secret",
        authorization_url="https://mock-idp.example/oauth/authorize",
        token_url="https://mock-idp.example/oauth/token",
        scopes=["openid"],
        oauth2_flow="client_credentials",
    )
    global_mcp_server_manager.registry[server.server_id] = server

    app = FastAPI()
    app.include_router(discoverable_oauth_router)
    try:
        yield app
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_authorize_redirect_uri_to_upstream_is_proxy_callback_not_client_loopback(
    oauth_asgi_app: FastAPI,
) -> None:
    transport = ASGITransport(app=oauth_asgi_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test", follow_redirects=False
    ) as client:
        r = await client.get(
            "/mock_oauth/authorize",
            params={
                "client_id": "upstream-client",
                "redirect_uri": "http://127.0.0.1:60108/ui/mcp/oauth/callback",
                "state": "plain-client-state",
                "code_challenge": "challenge",
                "code_challenge_method": "S256",
            },
        )
    assert r.status_code in (301, 302, 303, 307, 308)
    loc = r.headers["location"]
    assert loc.startswith("https://mock-idp.example/oauth/authorize")
    q = urllib.parse.urlparse(loc).query
    parsed = urllib.parse.parse_qs(q)
    upstream_redirect = parsed["redirect_uri"][0]
    assert upstream_redirect == "http://proxy.test/callback"
    assert "challenge" == parsed["code_challenge"][0]
    encrypted_state = parsed["state"][0]
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        decode_state_hash,
    )

    state_data = decode_state_hash(encrypted_state)
    assert state_data["original_state"] == "plain-client-state"
    assert (
        state_data["client_redirect_uri"]
        == "http://127.0.0.1:60108/ui/mcp/oauth/callback"
    )


@pytest.mark.asyncio
async def test_authorize_rejects_non_loopback_client_redirect_uri(
    oauth_asgi_app: FastAPI,
) -> None:
    transport = ASGITransport(app=oauth_asgi_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test", follow_redirects=False
    ) as client:
        r = await client.get(
            "/mock_oauth/authorize",
            params={
                "client_id": "upstream-client",
                "redirect_uri": "https://attacker.example/capture",
                "state": "x",
            },
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_callback_redirects_to_client_loopback_with_upstream_code(
    oauth_asgi_app: FastAPI,
) -> None:
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        encode_state_with_base_url,
    )

    client_cb = "http://127.0.0.1:7777/oauth/callback"
    base_no_query = "http://127.0.0.1:7777/oauth/callback"
    state_token = encode_state_with_base_url(
        base_url=base_no_query,
        original_state="csrf-token-9",
        code_challenge="cc",
        code_challenge_method="S256",
        client_redirect_uri=client_cb,
    )
    transport = ASGITransport(app=oauth_asgi_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test", follow_redirects=False
    ) as client:
        r = await client.get(
            "/callback",
            params={"code": "upstream-auth-code", "state": state_token},
        )
    assert r.status_code in (301, 302, 303, 307, 308)
    loc = r.headers["location"]
    assert loc.startswith("http://127.0.0.1:7777/oauth/callback")
    q = urllib.parse.urlparse(loc).query
    parsed = urllib.parse.parse_qs(q)
    assert parsed["code"][0] == "upstream-auth-code"
    assert parsed["state"][0] == "csrf-token-9"


@pytest.mark.asyncio
async def test_callback_rejects_non_loopback_in_decrypted_state(
    oauth_asgi_app: FastAPI,
) -> None:
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash",
        return_value={
            "original_state": "ok",
            "client_redirect_uri": "https://evil.com/y",
            "base_url": "https://evil.com/y",
        },
    ):
        transport = ASGITransport(app=oauth_asgi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy.test", follow_redirects=False
        ) as client:
            r = await client.get(
                "/callback",
                params={"code": "c", "state": "opaque"},
            )
    assert r.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_token_exchange_posts_proxy_callback_redirect_uri_to_upstream(
    oauth_asgi_app: FastAPI,
    respx_mock,
) -> None:
    captured: dict = {}

    def on_request(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "access_token": "at-upstream",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    respx_mock.post("https://mock-idp.example/oauth/token").mock(side_effect=on_request)

    transport = ASGITransport(app=oauth_asgi_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://proxy.test", follow_redirects=False
    ) as client:
        r = await client.post(
            "/mock_oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": "code-from-upstream",
                "client_id": "upstream-client",
                "client_secret": "upstream-secret",
                "code_verifier": "verifier",
                "redirect_uri": "ignored-by-litellm-for-upstream-exchange",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"
    assert r.headers.get("pragma") == "no-cache"
    body = r.json()
    assert body["access_token"] == "at-upstream"

    parsed = urllib.parse.parse_qs(captured.get("body", ""))
    assert parsed["grant_type"][0] == "authorization_code"
    assert parsed["redirect_uri"][0] == "http://proxy.test/callback"
    assert parsed["code"][0] == "code-from-upstream"
    assert parsed["code_verifier"][0] == "verifier"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_token_exchange_applies_token_validation_rules(
    oauth_asgi_app: FastAPI,
    respx_mock,
) -> None:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    srv = global_mcp_server_manager.registry["mock-oauth-srv"]
    prev_validation = getattr(srv, "token_validation", None)
    srv.token_validation = {"org_id": "expected-org"}
    try:
        respx_mock.post("https://mock-idp.example/oauth/token").respond(
            200,
            json={
                "access_token": "tok",
                "token_type": "Bearer",
                "expires_in": 60,
                "org_id": "wrong-org",
            },
        )
        transport = ASGITransport(app=oauth_asgi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy.test", follow_redirects=False
        ) as client:
            r = await client.post(
                "/mock_oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": "c",
                    "client_id": "upstream-client",
                    "client_secret": "upstream-secret",
                    "code_verifier": "v",
                },
            )
        assert r.status_code == 403
        err = r.json()
        assert err["detail"]["error"] == "token_validation_failed"
    finally:
        srv.token_validation = prev_validation


# ---------------------------------------------------------------------------
# Regression + security tests: management broker endpoints
#
# Three cases are verified against the exact routes in mcp_management_endpoints.py:
#
#   1. Nonexistent server_id → 404 (not 401, which would mean auth was re-added)
#   2. Global-registry server + no API key → 403 (unauthenticated callers must
#      not be able to invoke the OAuth broker for globally configured servers,
#      as doing so would use the proxy's stored client_secret)
#   3. Temp-session server + no API key → passes access check (browser OAuth
#      flow; temp sessions are admin-created and scoped to this flow).
#      Covered by test_management_broker_authorize_unauthenticated_temp_session_passes.
# ---------------------------------------------------------------------------


@pytest.fixture
def management_asgi_app(monkeypatch) -> FastAPI:
    monkeypatch.setenv("LITELLM_SALT_KEY", "integration-test-salt-key-32chars")
    from litellm.proxy.management_endpoints.mcp_management_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_management_broker_authorize_requires_no_api_key(
    management_asgi_app: FastAPI,
) -> None:
    """Nonexistent server → 404, not 401 (auth gate must not be present)."""
    transport = ASGITransport(app=management_asgi_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        r = await client.get(
            "/v1/mcp/server/oauth/nonexistent-server-id/authorize",
            params={
                "redirect_uri": "http://127.0.0.1:8080/callback",
                "state": "regression-test-state",
                "response_type": "code",
                "code_challenge": "abc123",
                "code_challenge_method": "S256",
                "client_id": "test-client",
            },
        )
    assert r.status_code != 401, (
        "Got 401 — user_api_key_auth was re-added to /authorize. " f"Response: {r.text}"
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_management_broker_token_requires_no_api_key(
    management_asgi_app: FastAPI,
) -> None:
    """Nonexistent server → 404, not 401 (auth gate must not be present)."""
    transport = ASGITransport(app=management_asgi_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        follow_redirects=False,
    ) as client:
        r = await client.post(
            "/v1/mcp/server/oauth/nonexistent-server-id/token",
            data={
                "grant_type": "authorization_code",
                "code": "test-code",
                "redirect_uri": "http://127.0.0.1:8080/callback",
                "client_id": "test-client",
            },
        )
    assert r.status_code != 401, (
        "Got 401 — user_api_key_auth was re-added to /token. " f"Response: {r.text}"
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_management_broker_rejects_unauthenticated_access_to_global_registry_server(
    management_asgi_app: FastAPI,
) -> None:
    """
    Unauthenticated callers must not reach the OAuth broker for a global-registry
    server. Allowing it would let anyone invoke the proxy's OAuth broker using the
    server's stored client_secret, while authenticated non-admins without allowlist
    access get 403 — an unintended privilege inversion.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.types.mcp import MCPAuth, MCPTransport

    server = MCPServer(
        server_id="global-oauth-srv",
        name="global_oauth",
        server_name="global_oauth",
        alias="global_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="client",
        client_secret="secret",
        authorization_url="https://idp.example/oauth/authorize",
        token_url="https://idp.example/oauth/token",
    )
    global_mcp_server_manager.registry["global-oauth-srv"] = server
    try:
        transport = ASGITransport(app=management_asgi_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            r = await client.get(
                "/v1/mcp/server/oauth/global-oauth-srv/authorize",
                params={
                    "redirect_uri": "http://127.0.0.1:8080/callback",
                    "state": "test",
                    "response_type": "code",
                    "code_challenge": "abc",
                    "code_challenge_method": "S256",
                    "client_id": "client",
                },
            )
        assert r.status_code == 403, (
            f"Expected 403 for unauthenticated access to a global-registry server, "
            f"got {r.status_code}. Response: {r.text}"
        )
    finally:
        global_mcp_server_manager.registry.pop("global-oauth-srv", None)


@pytest.mark.asyncio
async def test_management_broker_authorize_unauthenticated_temp_session_passes(
    management_asgi_app: FastAPI,
) -> None:
    """
    Positive path: a temp-cached MCP OAuth server must allow GET /authorize with
    no API key (browser redirect), yielding a redirect to the upstream IdP — not
    401/403 from the broker gate.
    """
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mcp_mod
    from litellm.proxy.management_endpoints.mcp_management_endpoints import (
        _cache_temporary_mcp_server,
    )

    server_id = "temp-broker-oauth-success-001"
    server = MCPServer(
        server_id=server_id,
        name="temp_oauth",
        server_name="temp_oauth",
        alias="temp_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="upstream-client",
        client_secret="upstream-secret",
        authorization_url="https://idp.example/oauth/authorize",
        token_url="https://idp.example/oauth/token",
    )
    _cache_temporary_mcp_server(server, ttl_seconds=300)
    try:
        transport = ASGITransport(app=management_asgi_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            r = await client.get(
                f"/v1/mcp/server/oauth/{server_id}/authorize",
                params={
                    "redirect_uri": "http://127.0.0.1:8080/callback",
                    "state": "browser-oauth-state",
                    "response_type": "code",
                    "code_challenge": "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
                    "code_challenge_method": "S256",
                    "client_id": "upstream-client",
                },
            )
        assert r.status_code != 401, (
            "Unauthenticated temp-session authorize must not hit master-key 401. "
            f"body={r.text}"
        )
        assert r.status_code != 403, (
            "Unauthenticated temp-session authorize must not be blocked as global. "
            f"body={r.text}"
        )
        assert r.status_code in (
            302,
            303,
            307,
        ), f"expected redirect to upstream IdP, got {r.status_code}: {r.text}"
        location = r.headers.get("location") or ""
        assert "idp.example" in location
    finally:
        mcp_mod._temporary_mcp_servers.pop(server_id, None)
