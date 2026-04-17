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

_test_app = FastAPI()
_test_app.include_router(router)


@pytest.fixture
def client():
    return TestClient(_test_app, raise_server_exceptions=False)


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
            "redirect_uri": "https://client.example.com/callback",
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
            "redirect_uri": "https://example.com/cb",
            "response_type": "token",
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
            "redirect_uri": "https://client.example.com/callback",
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
    body = result.body
    import json

    data = json.loads(body)
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
    with pytest.raises(HTTPException) as exc_info:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.master_key", "key"),
        ):
            await byok_token(
                request=mock_request,
                grant_type="authorization_code",
                code="nonexistent-code",
                redirect_uri="",
                code_verifier="anything",
                client_id="u",
            )
    assert exc_info.value.status_code == 400
    assert "invalid_grant" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_token_endpoint_expired_code():
    verifier = "exp_verifier_that_is_long_enough_to_be_valid"
    challenge = _make_challenge(verifier)
    code = _insert_code(
        api_key="key",
        server_id="s",
        user_id="u",
        challenge=challenge,
        redirect_uri="https://cb",
        ttl=-10,  # already expired
    )

    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.master_key", "key"),
        ):
            await byok_token(
                request=mock_request,
                grant_type="authorization_code",
                code=code,
                redirect_uri="",
                code_verifier=verifier,
                client_id="u",
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_token_endpoint_wrong_verifier():
    verifier = "correct_verifier_value_that_is_long_enough"
    challenge = _make_challenge(verifier)
    code = _insert_code(
        api_key="key",
        server_id="s",
        user_id="u",
        challenge=challenge,
        redirect_uri="https://cb",
    )

    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.master_key", "key"),
        ):
            await byok_token(
                request=mock_request,
                grant_type="authorization_code",
                code=code,
                redirect_uri="",
                code_verifier="wrong_verifier_value_that_wont_match",
                client_id="u",
            )
    assert exc_info.value.status_code == 400
    assert "invalid_grant" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_token_endpoint_unsupported_grant_type():
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import byok_token

    mock_request = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.master_key", "key"),
        ):
            await byok_token(
                request=mock_request,
                grant_type="client_credentials",
                code="any",
                redirect_uri="",
                code_verifier="v",
                client_id="u",
            )
    assert exc_info.value.status_code == 400
    assert "unsupported_grant_type" in str(exc_info.value.detail)


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
