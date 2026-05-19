"""
Unit tests for the BYOK OAuth 2.1 authorization server endpoints.

Covers:
- _verify_pkce helper
- OAuth metadata discovery endpoints
- Authorization GET / POST endpoints
- Token endpoint (PKCE verification, credential storage, JWT issuance)
- 401 challenge in execute_mcp_tool (_check_byok_credential)
"""

import base64
import hashlib
import json
import time
import uuid
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
    _byok_auth_codes,
    _verify_pkce,
    router,
)
from litellm.proxy._types import MCPTransport

# ---------------------------------------------------------------------------
# _verify_pkce
# ---------------------------------------------------------------------------


def _make_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def test_verify_pkce_valid():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    challenge = _make_challenge(verifier)
    assert _verify_pkce(verifier, challenge) is True


def test_verify_pkce_invalid():
    assert _verify_pkce("wrong_verifier", _make_challenge("right_verifier")) is False


def test_verify_pkce_tampered_challenge():
    verifier = "test_verifier_value"
    challenge = _make_challenge(verifier)
    # Flip one character to tamper with the challenge
    tampered = challenge[:-1] + ("A" if challenge[-1] != "A" else "B")
    assert _verify_pkce(verifier, tampered) is False


# ---------------------------------------------------------------------------
# Minimal FastAPI app for testing the router
# ---------------------------------------------------------------------------

from fastapi import FastAPI

from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
    _byok_session_auth,
)
from litellm.proxy._types import UserAPIKeyAuth

_test_app = FastAPI()
_test_app.include_router(router)


@pytest.fixture
def client():
    """Test client with a fixed authenticated user (bypasses the session
    cookie check by overriding the dep)."""
    _test_app.dependency_overrides[_byok_session_auth] = lambda: UserAPIKeyAuth(
        api_key="hashed", user_id="user-123"
    )
    try:
        yield TestClient(_test_app, raise_server_exceptions=False)
    finally:
        _test_app.dependency_overrides.pop(_byok_session_auth, None)


@pytest.fixture
def unauthenticated_client():
    """Test client with no dependency override — the real ``_byok_session_auth``
    runs, which checks the ``token`` cookie and falls back to
    ``user_api_key_auth``. With neither set, both paths fail → 401."""
    yield TestClient(_test_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# OAuth metadata endpoints
# ---------------------------------------------------------------------------


def test_oauth_authorization_server_metadata(client):
    resp = client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    data = resp.json()
    assert "issuer" in data
    assert data["authorization_endpoint"].endswith("/v1/mcp/oauth/authorize")
    assert data["token_endpoint"].endswith("/v1/mcp/oauth/token")
    assert "S256" in data["code_challenge_methods_supported"]


def test_oauth_protected_resource_metadata(client):
    resp = client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 200
    data = resp.json()
    assert "resource" in data
    assert "authorization_servers" in data
    assert len(data["authorization_servers"]) == 1


# ---------------------------------------------------------------------------
# Authorization GET endpoint
# ---------------------------------------------------------------------------


def test_authorize_get_returns_html(client):
    resp = client.get(
        "/v1/mcp/oauth/authorize",
        params={
            "client_id": "test-client",
            "redirect_uri": "http://127.0.0.1:3000/callback",
            "response_type": "code",
            "code_challenge": "abc123",
            "code_challenge_method": "S256",
            "state": "xyz",
            "server_id": "my-server",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # The button text is HTML-entity-escaped in the template
    assert "Connect &amp; Authorize" in resp.text
    # Hidden fields should be embedded
    assert "my-server" in resp.text
    assert "abc123" in resp.text


def test_authorize_get_missing_redirect_uri(client):
    resp = client.get(
        "/v1/mcp/oauth/authorize",
        params={
            "response_type": "code",
            "code_challenge": "abc",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_authorize_get_wrong_response_type(client):
    resp = client.get(
        "/v1/mcp/oauth/authorize",
        params={
            "redirect_uri": "http://127.0.0.1:3000/cb",
            "response_type": "token",
            "code_challenge": "abc",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_authorize_get_rejects_non_loopback_redirect_uri(client):
    """GET /authorize validates redirect_uri up front so the user sees
    the rejection before typing an API key into the HTML form — matches
    the POST handler's rule and avoids the ``user fills form → POST 400
    with no form state`` UX."""
    resp = client.get(
        "/v1/mcp/oauth/authorize",
        params={
            "redirect_uri": "https://attacker.example.com/cb",
            "response_type": "code",
            "code_challenge": "abc",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Authorization POST endpoint
# ---------------------------------------------------------------------------


def test_authorize_post_creates_code_and_redirects(client):
    verifier = "my_code_verifier_that_is_long_enough_43chars"
    challenge = _make_challenge(verifier)

    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "client_id": "user-123",
            "redirect_uri": "http://127.0.0.1:3000/callback",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st_abc",
            "server_id": "server-xyz",
            "api_key": "sk-supersecretkey",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert "code=" in location
    assert "st_abc" in location

    # Extract the code from the redirect URL
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(location).query)
    code = qs["code"][0]
    assert code in _byok_auth_codes
    entry = _byok_auth_codes[code]
    assert entry["api_key"] == "sk-supersecretkey"
    assert entry["server_id"] == "server-xyz"
    assert entry["user_id"] == "user-123"
    assert entry["code_challenge"] == challenge


def test_authorize_post_unsupported_method(client):
    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "client_id": "u",
            "redirect_uri": "https://example.com/cb",
            "code_challenge": "abc",
            "code_challenge_method": "plain",
            "state": "",
            "server_id": "s",
            "api_key": "key",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------


def _insert_code(
    api_key: str,
    server_id: str,
    user_id: str,
    challenge: str,
    redirect_uri: str,
    ttl: int = 300,
) -> str:
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": api_key,
        "server_id": server_id,
        "user_id": user_id,
        "code_challenge": challenge,
        "redirect_uri": redirect_uri,
        "expires_at": time.time() + ttl,
    }
    return code


@pytest.mark.asyncio
async def test_token_endpoint_success():
    """Happy path: valid code + PKCE → credential stored → JWT returned."""
    verifier = "my_test_code_verifier_value_long_enough_yes"
    challenge = _make_challenge(verifier)
    code = _insert_code(
        api_key="sk-myapikey",
        server_id="server-1",
        user_id="user-42",
        challenge=challenge,
        redirect_uri="https://example.com/cb",
    )

    mock_prisma = MagicMock()
    mock_store = AsyncMock()
    test_master_key = "test_master_key_value"

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.store_user_credential",
            mock_store,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.router",
        ),
    ):
        # Import the actual handler function directly
        from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
            byok_token,
        )

        mock_request = MagicMock()
        # Patch module-level globals in the function's module
        with patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.store_user_credential",
            mock_store,
        ):
            import litellm.proxy._experimental.mcp_server.byok_oauth_endpoints as mod

            original_prisma = None
            original_master_key = None

            # Temporarily inject our test values
            with (
                patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
                patch("litellm.proxy.proxy_server.master_key", test_master_key),
            ):
                result = await byok_token(
                    request=mock_request,
                    grant_type="authorization_code",
                    code=code,
                    redirect_uri="https://example.com/cb",
                    code_verifier=verifier,
                    client_id="user-42",
                )

    assert result.status_code == 200
    # RFC 6749 §5.1 / OAuth 2.1 draft-15 §4.1.3: token responses MUST NOT
    # be cached, as the body contains an access token.
    assert result.headers["cache-control"] == "no-store"
    assert result.headers["pragma"] == "no-cache"
    data = json.loads(result.body)
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 3600

    # Verify JWT payload
    import jwt as pyjwt

    payload = pyjwt.decode(data["access_token"], test_master_key, algorithms=["HS256"])
    assert payload["user_id"] == "user-42"
    assert payload["server_id"] == "server-1"
    assert payload["type"] == "byok_session"

    # Auth code was consumed
    assert code not in _byok_auth_codes

    # store_user_credential was called
    mock_store.assert_awaited_once_with(
        prisma_client=mock_prisma,
        user_id="user-42",
        server_id="server-1",
        credential="sk-myapikey",
    )


@pytest.mark.asyncio
async def test_token_endpoint_invalid_code():
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", "key"),
    ):
        result = await byok_token(
            request=mock_request,
            grant_type="authorization_code",
            code="nonexistent-code",
            redirect_uri="",
            code_verifier="anything",
            client_id="u",
        )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_expired_code():
    verifier = "exp_verifier_that_is_long_enough_to_be_valid"
    challenge = _make_challenge(verifier)
    code = _insert_code(
        api_key="key",
        server_id="s",
        user_id="u",
        challenge=challenge,
        redirect_uri="http://127.0.0.1:3000/cb",
        ttl=-10,  # already expired
    )

    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", "key"),
    ):
        result = await byok_token(
            request=mock_request,
            grant_type="authorization_code",
            code=code,
            redirect_uri="http://127.0.0.1:3000/cb",
            code_verifier=verifier,
            client_id="u",
        )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_wrong_verifier():
    verifier = "correct_verifier_value_that_is_long_enough"
    challenge = _make_challenge(verifier)
    code = _insert_code(
        api_key="key",
        server_id="s",
        user_id="u",
        challenge=challenge,
        redirect_uri="http://127.0.0.1:3000/cb",
    )

    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", "key"),
    ):
        result = await byok_token(
            request=mock_request,
            grant_type="authorization_code",
            code=code,
            redirect_uri="http://127.0.0.1:3000/cb",
            code_verifier="wrong_verifier_value_that_wont_match",
            client_id="u",
        )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_token_endpoint_unsupported_grant_type():
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", "key"),
    ):
        result = await byok_token(
            request=mock_request,
            grant_type="client_credentials",
            code="any",
            redirect_uri="",
            code_verifier="v",
            client_id="u",
        )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "unsupported_grant_type"}


# ---------------------------------------------------------------------------
# _check_byok_credential  (the 401 challenge in execute_mcp_tool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_byok_credential_not_byok():
    """Non-BYOK servers should pass through without any DB check."""
    from litellm.proxy._experimental.mcp_server.server import _check_byok_credential
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="s1",
        name="normal-server",
        transport=MCPTransport.http,
        is_byok=False,
    )
    # Should not raise
    await _check_byok_credential(server, None)


@pytest.mark.asyncio
async def test_check_byok_credential_no_user_id():
    """BYOK server with no user identity → 401."""
    from litellm.proxy._experimental.mcp_server.server import _check_byok_credential
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="byok-1",
        name="byok-server",
        transport=MCPTransport.http,
        is_byok=True,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _check_byok_credential(server, None)

    assert exc_info.value.status_code == 401
    assert "WWW-Authenticate" in (exc_info.value.headers or {})  # type: ignore[operator]
    assert "byok_auth_required" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_check_byok_credential_missing_credential():
    """BYOK server with a known user but no stored credential → 401."""
    from litellm.proxy._experimental.mcp_server.server import _check_byok_credential
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="byok-2",
        name="byok-server",
        transport=MCPTransport.http,
        is_byok=True,
    )
    user_auth = UserAPIKeyAuth(user_id="user-99", api_key="sk-test")

    mock_prisma = MagicMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_user_credential",
            new=AsyncMock(return_value=None),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _check_byok_credential(server, user_auth)

    assert exc_info.value.status_code == 401
    detail: Any = exc_info.value.detail
    assert detail["error"] == "byok_auth_required"
    assert detail["server_id"] == "byok-2"
    headers = exc_info.value.headers or {}
    assert "WWW-Authenticate" in headers  # type: ignore[operator]
    assert "oauth-protected-resource" in headers["WWW-Authenticate"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_check_byok_credential_has_credential():
    """BYOK server with a valid stored credential → no error raised."""
    from litellm.proxy._experimental.mcp_server.server import _check_byok_credential
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="byok-3",
        name="byok-server",
        transport=MCPTransport.http,
        is_byok=True,
    )
    user_auth = UserAPIKeyAuth(user_id="user-77", api_key="sk-test")

    mock_prisma = MagicMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_user_credential",
            new=AsyncMock(return_value="some-credential-value"),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        # Should not raise
        await _check_byok_credential(server, user_auth)


@pytest.mark.asyncio
async def test_check_byok_credential_db_unavailable_fails_closed():
    """BYOK server with no prisma_client → 503, not silent pass.

    Regression for GHSA-6762: previously returned silently, bypassing the
    ownership check during DB outage windows.
    """
    from litellm.proxy._experimental.mcp_server.server import _check_byok_credential
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="byok-4",
        name="byok-server",
        transport=MCPTransport.http,
        is_byok=True,
    )
    user_auth = UserAPIKeyAuth(user_id="user-55", api_key="sk-test")

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        with pytest.raises(HTTPException) as exc_info:
            await _check_byok_credential(server, user_auth)

    assert exc_info.value.status_code == 503
    detail: Any = exc_info.value.detail
    assert detail["error"] == "byok_auth_unavailable"
    assert detail["server_id"] == "byok-4"


# ---------------------------------------------------------------------------
# Security regression tests for AO2kf_-9 / GHSA-jg3h:
# Unauthenticated /v1/mcp/oauth/authorize previously allowed an attacker to
# stamp `user_id = client_id` into the auth-code record, overwriting any
# victim's stored BYOK credential at /token.
# ---------------------------------------------------------------------------


def test_authorize_post_rejects_unauthenticated(unauthenticated_client):
    resp = unauthenticated_client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "client_id": "victim-user-id",
            "redirect_uri": "http://127.0.0.1:3000/cb",
            "code_challenge": "abc",
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "attacker-key",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 401


def test_authorize_post_ignores_spec_compliant_client_id(client):
    """An MCP client following OAuth 2.1 semantics sends client_id as its
    *application* identifier (e.g. "claude-desktop"). The stored user_id
    must come from the authenticated session regardless — the form's
    client_id is informational only."""
    verifier = "verifier_value_long_enough_to_be_valid_43chars"
    challenge = _make_challenge(verifier)

    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            # Fixture authenticates as "user-123"; the client identifies
            # itself as "claude-desktop" per OAuth 2.1 — unrelated to user.
            "client_id": "claude-desktop",
            "redirect_uri": "http://127.0.0.1:3000/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "upstream-key",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302

    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(resp.headers["location"]).query)
    code = qs["code"][0]
    # Identity = authenticated session, not the form's client_id.
    assert _byok_auth_codes[code]["user_id"] == "user-123"


def test_authorize_post_binds_code_to_authenticated_user_id(client):
    """Ensure the stored auth-code record uses the authenticated user_id,
    NOT the form's client_id, as the identity the token endpoint will trust."""
    verifier = "verifier_value_long_enough_to_be_valid_43chars"
    challenge = _make_challenge(verifier)

    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            # client_id omitted — must still bind to the authenticated user.
            "redirect_uri": "http://127.0.0.1:3000/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "legit-key",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302

    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(resp.headers["location"]).query)
    code = qs["code"][0]
    assert _byok_auth_codes[code]["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_token_endpoint_rejects_missing_user_id_in_code_record():
    """Defense in depth: if a code record somehow lacks user_id (older
    format / manual DB write), /token must reject rather than fall back to
    the form's client_id."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_test_missing_user_id_path"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "user_id": "",  # missing / empty
        "expires_at": time.time() + 60,
    }

    result = await byok_token(
        request=MagicMock(),
        grant_type="authorization_code",
        code=code,
        redirect_uri="http://127.0.0.1:3000/cb",
        code_verifier=verifier,
        client_id="attacker-chosen",
    )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


def _authorize_post_with_cookie(client, cookie_jwt: str, api_key: str = "upstream-key"):
    verifier = "verifier_cookie_auth_long_enough_to_be_valid"
    return client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "redirect_uri": "http://127.0.0.1:3000/cb",
            "code_challenge": _make_challenge(verifier),
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": api_key,
        },
        cookies={"token": cookie_jwt},
        follow_redirects=False,
    )


def test_authorize_post_accepts_ui_session_cookie(unauthenticated_client):
    """Browser flow: the native HTML form doesn't add Authorization. Instead
    the user's UI session cookie ``token`` carries a master-key-signed JWT
    whose ``user_id`` + ``login_method`` claims authenticate the POST."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "test-master-key"):
        cookie_jwt = _jwt.encode(
            {
                "user_id": "browser-user-42",
                "login_method": "sso",
                "exp": int(time.time()) + 3600,
            },
            "test-master-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(unauthenticated_client, cookie_jwt)
    assert resp.status_code == 302

    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(resp.headers["location"]).query)
    code = qs["code"][0]
    assert _byok_auth_codes[code]["user_id"] == "browser-user-42"


def test_authorize_post_rejects_cookie_signed_with_wrong_key(unauthenticated_client):
    """A cookie JWT signed with a different key than the proxy's master_key
    must not grant access — otherwise an attacker who can forge a JWT
    against any key could impersonate any user."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "real-master-key"):
        forged = _jwt.encode(
            {"user_id": "victim-user", "login_method": "sso"},
            "attacker-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(unauthenticated_client, forged, api_key="k")
    assert resp.status_code == 401


def test_authorize_post_rejects_replayed_byok_session_token(unauthenticated_client):
    """Regression: the /token endpoint itself issues master-key-signed JWTs
    with ``type="byok_session"`` + ``user_id`` (for MCP-client use). Those
    tokens must not be accepted here — otherwise an attacker with any
    byok_session token could replay it as a Cookie and re-authorize BYOK
    writes without a valid UI session."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "real-master-key"):
        byok_session = _jwt.encode(
            {
                "user_id": "any-user",
                "server_id": "sid",
                "type": "byok_session",
            },
            "real-master-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(
            unauthenticated_client, byok_session, api_key="k"
        )
    assert resp.status_code == 401


def test_authorize_post_rejects_non_loopback_redirect_uri(client):
    """OAuth 2.1 §4.1.2.1 + RFC 8252: native-app redirects must be loopback.
    A public HTTPS callback from an MCP client would let that client capture
    the issued code after a legitimate user enters their API key, so we
    reject anything that isn't 127.0.0.1/localhost/::1."""
    verifier = "verifier_non_loopback_redirect_uri_test_long"
    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "redirect_uri": "https://attacker.example.com/cb",
            "code_challenge": _make_challenge(verifier),
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "k",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_token_endpoint_rejects_redirect_uri_mismatch():
    """RFC 6749 §4.1.3 / OAuth 2.1 §4.1.3: if redirect_uri was sent at
    /authorize, the /token redirect_uri MUST match exactly."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_redirect_mismatch_test_long"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "user_id": "u",
        "expires_at": time.time() + 60,
    }

    result = await byok_token(
        request=MagicMock(),
        grant_type="authorization_code",
        code=code,
        redirect_uri="http://127.0.0.1:9999/cb",  # different port
        code_verifier=verifier,
        client_id="",
    )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


def test_authorize_post_rejects_cookie_missing_login_method(unauthenticated_client):
    """Defense in depth: a master-key-signed JWT with only ``user_id`` is not
    a valid UI session (UI tokens always carry ``login_method``). Accepting
    it would expand the cookie surface to include any master-key-signed
    JWT in the system, which is exactly what the byok_session-replay
    regression above protects against."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "real-master-key"):
        malformed = _jwt.encode(
            {"user_id": "some-user"},
            "real-master-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(
            unauthenticated_client, malformed, api_key="k"
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_token_endpoint_accepts_spec_compliant_client_id():
    """Per OAuth 2.1, the /token client_id is the client application
    identifier, not the user. It must not be cross-checked against the
    record's user_id — that cross-check would break spec-compliant MCP
    clients that pass e.g. client_id="claude-desktop"."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_spec_compliant_long_enough_43chars"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "user_id": "real-authenticated-user",
        "expires_at": time.time() + 60,
    }

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.store_user_credential",
            new=AsyncMock(),
        ),
        patch("litellm.proxy.proxy_server.master_key", "test-master"),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        result = await byok_token(
            request=MagicMock(),
            grant_type="authorization_code",
            code=code,
            redirect_uri="http://127.0.0.1:3000/cb",
            code_verifier=verifier,
            client_id="claude-desktop",  # OAuth app id, unrelated to user
        )
    # Token should be issued successfully.
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_token_endpoint_rejects_client_id_mismatch():
    """RFC 6749 §4.1.3: if the authorization request was bound to a
    client_id, the token request must submit the same value. An attacker
    who steals a code from another client (different native app) can't
    redeem it."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_client_id_mismatch_long_enough!"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "client_id": "legitimate-client",
        "user_id": "u",
        "expires_at": time.time() + 60,
    }

    result = await byok_token(
        request=MagicMock(),
        grant_type="authorization_code",
        code=code,
        redirect_uri="http://127.0.0.1:3000/cb",
        code_verifier=verifier,
        client_id="attacker-client",
    )
    assert result.status_code == 400
    assert json.loads(result.body) == {"error": "invalid_grant"}


def test_authorize_post_accepts_ipv4_loopback_range(client):
    """RFC 8252 §7.3 / RFC 5735: ``127.0.0.0/8`` is loopback — a string
    match on ``127.0.0.1`` would miss ``127.0.0.2`` and break clients that
    pick a loopback alias."""
    verifier = "verifier_for_127002_loopback_test_long_enough"
    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "redirect_uri": "http://127.0.0.2:3000/cb",
            "code_challenge": _make_challenge(verifier),
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "k",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302


def test_authorize_post_accepts_ipv6_loopback_full_form(client):
    """RFC 4291: full IPv6 loopback ``0:0:0:0:0:0:0:1`` must be accepted
    equivalently to ``::1``."""
    verifier = "verifier_for_ipv6_full_loopback_test_long_enough"
    resp = client.post(
        "/v1/mcp/oauth/authorize",
        data={
            "redirect_uri": "http://[0:0:0:0:0:0:0:1]:3000/cb",
            "code_challenge": _make_challenge(verifier),
            "code_challenge_method": "S256",
            "state": "s",
            "server_id": "sid",
            "api_key": "k",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302


@pytest.mark.asyncio
async def test_token_endpoint_missing_master_key_preserves_code_and_db():
    """If master_key is unset, /token must reject BEFORE consuming the
    code or writing the credential — otherwise a misconfigured deploy
    burns the code and persists the key with no way for the user to
    retrieve a session token without restarting the flow."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_missing_master_key_test_long_!"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "client_id": "",
        "user_id": "u",
        "expires_at": time.time() + 60,
    }

    mock_store = AsyncMock()
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.store_user_credential",
            mock_store,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        result = await byok_token(
            request=MagicMock(),
            grant_type="authorization_code",
            code=code,
            redirect_uri="http://127.0.0.1:3000/cb",
            code_verifier=verifier,
            client_id="",
        )

    assert result.status_code == 500
    assert json.loads(result.body) == {"error": "server_error"}
    # Code still present — user can retry once master_key is configured.
    assert code in _byok_auth_codes
    # Credential never written — no inconsistent DB state.
    mock_store.assert_not_awaited()


def test_authorize_post_rejects_cookie_without_exp(unauthenticated_client):
    """Defense-in-depth: UI session cookies must carry an ``exp`` claim
    so a leaked cookie has a bounded lifetime. A master-key-signed JWT
    without ``exp`` is rejected at decode time (PyJWT
    ``options={"require": ["exp"]}``)."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "real-master-key"):
        no_exp = _jwt.encode(
            {"user_id": "u", "login_method": "sso"},
            "real-master-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(unauthenticated_client, no_exp, api_key="k")
    assert resp.status_code == 401


def test_authorize_post_rejects_expired_cookie(unauthenticated_client):
    """An expired UI session cookie is rejected, not accepted as valid."""
    import jwt as _jwt

    with patch("litellm.proxy.proxy_server.master_key", "real-master-key"):
        expired = _jwt.encode(
            {
                "user_id": "u",
                "login_method": "sso",
                "exp": int(time.time()) - 60,  # 1 minute ago
            },
            "real-master-key",
            algorithm="HS256",
        )
        resp = _authorize_post_with_cookie(unauthenticated_client, expired, api_key="k")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_token_endpoint_accepts_oauth21_client_omitting_redirect_uri():
    """OAuth 2.1 draft-15 §4.1.3 dropped the redirect_uri requirement at
    the token endpoint. A strict OAuth 2.1 client will omit the value —
    LiteLLM must accept that and rely on PKCE + client_id binding for
    the security role redirect_uri played under RFC 6749.

    Enforcement still fires when the client DOES submit a value that
    disagrees with the record (see test_token_endpoint_rejects_redirect_uri_mismatch).
    """
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        byok_token,
    )

    verifier = "verifier_for_oauth21_no_redirect_uri_omit_ok!"
    challenge = _make_challenge(verifier)
    code = str(uuid.uuid4())
    _byok_auth_codes[code] = {
        "api_key": "k",
        "server_id": "sid",
        "code_challenge": challenge,
        "redirect_uri": "http://127.0.0.1:3000/cb",
        "client_id": "claude-desktop",
        "user_id": "u",
        "expires_at": time.time() + 60,
    }

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.byok_oauth_endpoints.store_user_credential",
            new=AsyncMock(),
        ),
        patch("litellm.proxy.proxy_server.master_key", "test-master"),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        result = await byok_token(
            request=MagicMock(),
            grant_type="authorization_code",
            code=code,
            redirect_uri="",  # OAuth 2.1 client omits it
            code_verifier=verifier,
            client_id="claude-desktop",
        )
    assert result.status_code == 200


def test_validate_loopback_redirect_uri_rejects_fragment():
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_loopback_redirect_uri,
    )

    with pytest.raises(HTTPException) as exc:
        validate_loopback_redirect_uri("http://127.0.0.1:3000/cb#code=1")
    assert exc.value.status_code == 400


def test_validate_loopback_redirect_uri_rejects_malformed_cleanly():
    """Malformed / unparseable URIs should surface as 400 invalid_request,
    not a 500 from an unhandled exception inside ip_address()."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_loopback_redirect_uri,
    )

    # Netloc that parses but whose host is neither "localhost" nor a valid IP.
    with pytest.raises(HTTPException) as exc:
        validate_loopback_redirect_uri("http://[not-an-ip]/cb")
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# validate_trusted_redirect_uri — same-origin + loopback + env allowlist
# ---------------------------------------------------------------------------


def _make_trusted_request(base_url: str = "https://llm.example.com/"):
    """Build a request-like object whose same-origin is ``base_url``.

    ``get_request_base_url`` defers to ``request.base_url`` unless the
    caller is a trusted proxy, so passing the target origin as
    ``base_url`` is sufficient here — no X-Forwarded headers needed.
    """
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.base_url = base_url
    mock.headers = {}
    return mock


def test_validate_trusted_redirect_uri_accepts_same_origin():
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://llm.example.com/")
    validate_trusted_redirect_uri(req, "https://llm.example.com/ui/mcp/callback")


def test_validate_trusted_redirect_uri_same_origin_normalizes_default_port():
    """Regression: a load balancer that sets X-Forwarded-Port: 443 would
    otherwise produce a proxy_base of ``https://llm.example.com:443``
    which wouldn't literally match the browser's port-less ``llm.example.com``
    redirect_uri even though both represent the same origin."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    # Proxy base with explicit :443 — redirect_uri without a port.
    req = _make_trusted_request("https://llm.example.com:443/")
    validate_trusted_redirect_uri(req, "https://llm.example.com/cb")

    # And the symmetric case — redirect_uri has the explicit port.
    req2 = _make_trusted_request("https://llm.example.com/")
    validate_trusted_redirect_uri(req2, "https://llm.example.com:443/cb")


def test_validate_trusted_redirect_uri_accepts_loopback():
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://llm.example.com/")
    for uri in (
        "http://localhost:3000/cb",
        "http://127.0.0.1:3000/cb",
        "http://127.0.0.55/cb",
        "http://[::1]/cb",
    ):
        validate_trusted_redirect_uri(req, uri)


def test_validate_trusted_redirect_uri_rejects_cross_origin_by_default(
    monkeypatch,
):
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.delenv("MCP_TRUSTED_REDIRECT_ORIGINS", raising=False)
    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://attacker.example.net/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_rejects_fragment_and_bad_scheme():
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://llm.example.com/")
    for uri in (
        "https://llm.example.com/cb#frag",  # fragment
        "ftp://llm.example.com/cb",  # unsupported scheme
        "https:///no-netloc",  # missing netloc
    ):
        with pytest.raises(HTTPException) as exc:
            validate_trusted_redirect_uri(req, uri)
        assert exc.value.status_code == 400, uri


def test_validate_trusted_redirect_uri_rejects_scheme_mismatch_on_same_host():
    """Regression: an attacker who can serve http on the proxy's own
    host (e.g. by MITMing an unencrypted LAN hop) must not be able to
    pass same-origin validation — scheme must match as well as host."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "http://llm.example.com/ui/callback")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_rejects_userinfo(monkeypatch):
    """VERIA finding: an attacker can hide the real destination host in
    the post-``@`` portion of the URL, while the pre-``@`` userinfo is
    styled to look like an allowlisted host. Without an explicit
    username/password check, a wildcard allowlist that splits the raw
    netloc on ``:`` sees ``app.example.com`` and accepts; the browser
    then navigates to ``attacker.example`` with the authorization code.

    Reject userinfo at every tier — same-origin, loopback, exact-entry
    allowlist, and wildcard allowlist — so the bypass is closed on
    every path through the validator.
    """
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    # (1) Wildcard allowlist — the original VERIA vector, including the
    # ``:443`` inside userinfo that makes the raw netloc split deceptive.
    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    for uri in (
        "https://app.example.com:443@attacker.example/cb",
        "https://app.example.com@attacker.example/cb",
    ):
        with pytest.raises(HTTPException) as exc:
            validate_trusted_redirect_uri(req, uri)
        assert exc.value.status_code == 400, uri

    # (2) Exact-entry allowlist — same class of bypass, different path.
    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "app.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(
            req, "https://app.example.com@attacker.example/cb"
        )
    assert exc.value.status_code == 400

    # (3) Same-origin path — userinfo that mimics the proxy's host.
    monkeypatch.delenv("MCP_TRUSTED_REDIRECT_ORIGINS", raising=False)
    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(
            req, "https://llm.example.com@attacker.example/cb"
        )
    assert exc.value.status_code == 400

    # (4) Loopback path — userinfo that mimics 127.0.0.1.
    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "http://127.0.0.1@attacker.example/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_rejects_backslash_in_netloc(monkeypatch):
    """VERIA finding: urlparse keeps backslashes in ``netloc``, but
    browsers normalize ``\\`` to ``/`` on http(s) URLs and treat it as
    the start of the path. An allowlist of ``*.example.com`` would
    accept ``https://attacker.net\\app.example.com/cb`` (the raw netloc
    ends with ``.example.com``) while the browser navigates to
    ``attacker.net`` and delivers the authorization code there.

    Reject on every path through the validator — same-origin,
    exact-entry, and wildcard — by bouncing the netloc before any
    matching runs.
    """
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    # (1) Wildcard allowlist — the VERIA vector.
    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://attacker.net\\app.example.com/cb")
    assert exc.value.status_code == 400

    # (2) Exact-entry allowlist — same split, different match path.
    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "app.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://attacker.net\\app.example.com/cb")
    assert exc.value.status_code == 400

    # (3) Same-origin path — backslash that mimics the proxy's host.
    monkeypatch.delenv("MCP_TRUSTED_REDIRECT_ORIGINS", raising=False)
    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://attacker.net\\llm.example.com/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_allowlist_entry_with_default_port(monkeypatch):
    """Regression: operators who write ``app.example.com:443`` in
    ``MCP_TRUSTED_REDIRECT_ORIGINS`` (natural when copy-pasting from a
    browser address bar or load-balancer log) must still match a
    port-less redirect_uri. The redirect_uri's ``:443`` is normalized
    away for the same-origin compare; the allowlist side has to apply
    the same normalization or the comparison is asymmetric and silently
    fails."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        _parse_trusted_redirect_origins,
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "app.example.com:443")
    # Verify the parse step itself drops the default port.
    assert _parse_trusted_redirect_origins() == ["app.example.com"]

    req = _make_trusted_request("https://llm.example.com/")
    # Port-less redirect_uri — should match the :443 env entry.
    validate_trusted_redirect_uri(req, "https://app.example.com/cb")
    # Explicit :443 on both sides — should still match.
    validate_trusted_redirect_uri(req, "https://app.example.com:443/cb")
    # Non-default port on the redirect_uri — must NOT match a default-port entry.
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://app.example.com:8443/cb")


def test_validate_trusted_redirect_uri_accepts_exact_allowlisted_host(monkeypatch):
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv(
        "MCP_TRUSTED_REDIRECT_ORIGINS",
        "app.example.com, https://other.example.com/",
    )
    req = _make_trusted_request("https://llm.example.com/")
    # Exact allowlisted host — accepted.
    validate_trusted_redirect_uri(req, "https://app.example.com/oauth/cb")
    # Path component on the env entry should be stripped at parse time;
    # the URL still resolves to an allowlisted host.
    validate_trusted_redirect_uri(req, "https://other.example.com/anything")
    # An unrelated host still fails.
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://different.example.com/cb")


def test_validate_trusted_redirect_uri_allowlist_rejects_http_even_on_listed_host(
    monkeypatch,
):
    """An attacker must not be able to elevate to the allowlist by
    serving http:// on the listed host — only https is accepted for
    non-loopback allowlist entries."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "app.example.com")
    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "http://app.example.com/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_wildcard_allowlist(monkeypatch):
    """``*.suffix`` entries match any strictly-deeper subdomain of
    ``suffix`` but must not match the bare suffix, nor unrelated domains
    that happen to end with the same characters."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")

    # Direct subdomain — accepted.
    validate_trusted_redirect_uri(req, "https://app.example.com/cb")
    # Nested subdomain — accepted.
    validate_trusted_redirect_uri(req, "https://foo.bar.example.com/cb")

    # Bare suffix — NOT accepted (wildcard requires a proper subdomain).
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://example.com/cb")

    # Similar-looking domain that isn't a subdomain — NOT accepted.
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://evil-example.com/cb")
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://example.com.attacker.net/cb")


def test_validate_trusted_redirect_uri_wildcard_rejects_http(monkeypatch):
    """The https-only gate applies to wildcard entries too."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "http://app.example.com/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_wildcard_host_with_port_still_matches(
    monkeypatch,
):
    """Wildcard entries don't express port constraints — an allowlisted
    subdomain should match regardless of explicit port on the URL."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*.example.com")
    req = _make_trusted_request("https://llm.other-proxy.com/")
    validate_trusted_redirect_uri(req, "https://app.example.com:8443/cb")


def test_validate_trusted_redirect_uri_accepts_ipv6_loopback_with_default_port():
    """IPv6 loopback with explicit ``:443`` on an ``https`` URL should
    still match — exercises ``_strip_default_port``'s IPv6 branch."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://[::1]/")
    validate_trusted_redirect_uri(req, "https://[::1]:443/cb")


def test_validate_trusted_redirect_uri_tolerates_malformed_env_entries(monkeypatch):
    """Operators occasionally mis-type env values (empty items, bare
    ``*.``, non-numeric ports). None of those should raise; unmatched
    entries must simply fail to grant access while well-formed entries
    in the same list continue to work."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv(
        "MCP_TRUSTED_REDIRECT_ORIGINS",
        ",  ,*.,  foo:notaport,  app.example.com",
    )
    req = _make_trusted_request("https://llm.example.com/")
    # Well-formed entry still works.
    validate_trusted_redirect_uri(req, "https://app.example.com/cb")
    # Bare ``*.`` grants nothing.
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://example.com/cb")
    # Non-numeric port entry is ignored (doesn't grant access).
    with pytest.raises(HTTPException):
        validate_trusted_redirect_uri(req, "https://foo.example.net/cb")


def test_validate_trusted_redirect_uri_rejects_wildcard_entry_with_dot_leading_suffix(
    monkeypatch,
):
    """A wildcard entry like ``*..example.com`` has a suffix that starts
    with ``.``, which would otherwise match ``anything.example.com`` via
    the ``host.endswith("." + suffix)`` branch by accepting a netloc
    whose own leading ``.`` makes it look like a deeper subdomain.
    Operators who mistype an extra dot should get an ignored entry, not
    a broader match than they intended."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    monkeypatch.setenv("MCP_TRUSTED_REDIRECT_ORIGINS", "*..example.com")
    req = _make_trusted_request("https://llm.example.com/")

    # None of these should resolve against the malformed wildcard entry.
    for uri in (
        "https://app.example.com/cb",
        "https://foo.bar.example.com/cb",
        "https://example.com/cb",
    ):
        with pytest.raises(HTTPException) as exc:
            validate_trusted_redirect_uri(req, uri)
        assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_falls_through_when_origin_lookup_fails():
    """If ``get_request_base_url`` can't determine the proxy's origin,
    same-origin is skipped silently but loopback + allowlist paths are
    still reachable."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    class _ExplodingRequest:
        # Accessing ``.base_url`` is what ``get_request_base_url``
        # reaches for first; raising here lets us exercise the swallowed-
        # error fallback without monkey-patching imports.
        base_url = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        headers: dict = {}

    req = _ExplodingRequest()
    # Loopback still accepted despite origin lookup failure.
    validate_trusted_redirect_uri(req, "http://127.0.0.1:3000/cb")


def test_strip_default_port_empty_netloc():
    """``_strip_default_port("", "")`` should round-trip — validator
    rejects empty-netloc URLs upstream so this is purely a defensive
    contract on the helper itself."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import _strip_default_port

    assert _strip_default_port("https", "") == ""


def test_strip_default_port_handles_non_numeric_port():
    """Raw netloc with a non-numeric port is returned unchanged. Reached
    in practice when a malformed ``Host`` header survives upstream
    parsing — we stay out of its way rather than 500ing."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import _strip_default_port

    assert _strip_default_port("https", "foo.com:bar") == "foo.com:bar"
    assert _strip_default_port("https", "[::1]:bar") == "[::1]:bar"


def test_validate_trusted_redirect_uri_rejects_public_ip_without_allowlist():
    """A redirect_uri whose host is a public IP (parseable by
    ``ip_address`` but not loopback) must fail all three tiers and 400."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _make_trusted_request("https://llm.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://1.2.3.4/cb")
    assert exc.value.status_code == 400


def test_parse_trusted_redirect_origins_drops_bare_path_entries(monkeypatch):
    """``/foo`` has a scheme-less leading slash and would strip to the
    empty string — drop silently rather than allowlisting empty
    origins."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        _parse_trusted_redirect_origins,
    )

    monkeypatch.setenv(
        "MCP_TRUSTED_REDIRECT_ORIGINS", "https:///, /foo, app.example.com"
    )
    # The two malformed entries drop out; only the real host survives.
    assert _parse_trusted_redirect_origins() == ["app.example.com"]
