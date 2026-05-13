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


def _mock_request_with_base_url(base_url: str):
    req = MagicMock()
    req.base_url = base_url
    req.headers = {}
    return req


def test_validate_trusted_redirect_uri_accepts_same_origin():
    """UI OAuth flow: redirect_uri on the proxy's own origin is allowed."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _mock_request_with_base_url("https://proxy.example.com/")
    # Should not raise.
    validate_trusted_redirect_uri(
        req, "https://proxy.example.com/ui/mcp/oauth/callback"
    )


def test_validate_trusted_redirect_uri_accepts_loopback():
    """Native MCP client flow: loopback is still allowed."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _mock_request_with_base_url("https://proxy.example.com/")
    validate_trusted_redirect_uri(req, "http://127.0.0.1:3000/cb")
    validate_trusted_redirect_uri(req, "http://localhost:3000/cb")


def test_validate_trusted_redirect_uri_rejects_external_origin():
    """An attacker-controlled origin must still be rejected."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _mock_request_with_base_url("https://proxy.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://attacker.example.com/cb")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_rejects_scheme_mismatch():
    """https→http (or vice versa) on the same host is not same-origin."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _mock_request_with_base_url("https://proxy.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "http://proxy.example.com/ui/callback")
    assert exc.value.status_code == 400


def test_validate_trusted_redirect_uri_rejects_fragment():
    from litellm.proxy._experimental.mcp_server.oauth_utils import (
        validate_trusted_redirect_uri,
    )

    req = _mock_request_with_base_url("https://proxy.example.com/")
    with pytest.raises(HTTPException) as exc:
        validate_trusted_redirect_uri(req, "https://proxy.example.com/ui/cb#code=1")
    assert exc.value.status_code == 400
