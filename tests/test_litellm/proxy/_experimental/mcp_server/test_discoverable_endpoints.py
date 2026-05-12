"""Tests for MCP OAuth discoverable endpoints"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# Fixture to mock IP address check for all MCP tests
# This prevents tests from failing due to IP-based access control
@pytest.fixture(autouse=True)
def mock_mcp_client_ip():
    """Mock IPAddressUtils.get_mcp_client_ip to return None for all tests.

    This bypasses IP-based access control in tests, since the MCP server's
    available_on_public_internet defaults to False and mock requests don't
    have proper client IP context.
    """
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
        return_value=None,
    ):
        yield


def _mock_callback_request(base_url: str = "http://localhost:3000/"):
    """Return a MagicMock Request for callback/authorize same-origin tests.

    The callback handler only uses ``request`` to compute the proxy's own
    base URL via ``get_request_base_url`` (which reads ``request.base_url``
    and trusted ``X-Forwarded-*`` headers). A simple MagicMock with the
    right attributes is sufficient.
    """
    req = MagicMock()
    req.base_url = base_url
    req.headers = {}
    return req


@pytest.fixture
def trust_xff():
    """Force ``IPAddressUtils.is_request_from_trusted_proxy`` to True.

    Tests that exercise X-Forwarded-* parsing logic opt into this fixture.
    The trust gate's own behaviour is covered by
    ``test_get_request_base_url_xff_trust_gate``.
    """
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.is_request_from_trusted_proxy",
        return_value=True,
    ):
        yield


@pytest.mark.asyncio
async def test_authorize_endpoint_includes_response_type():
    """Test that authorize endpoint includes response_type=code parameter (fixes #15684)"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    # Mock the encryption functions to avoid needing a signing key
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        # Call authorize endpoint
        response = await authorize(
            request=mock_request,
            client_id="test_client_id",
            mcp_server_name="test_oauth",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="test_state",
        )

    # Verify response is a redirect
    assert response.status_code == 307  # FastAPI RedirectResponse default

    # Verify response_type is in the redirect URL
    assert "response_type=code" in response.headers["location"]
    assert "https://provider.com/oauth/authorize" in response.headers["location"]
    assert "client_id=test_client_id" in response.headers["location"]
    assert "scope=read+write" in response.headers["location"]


@pytest.mark.asyncio
async def test_authorize_endpoint_preserves_existing_query_params():
    """Test that authorize endpoint merges OAuth params with existing query params in authorization_url"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()

    # Authorization URL already has query params (e.g. multi-tenant OAuth)
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize?tenant=system",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        response = await authorize(
            request=mock_request,
            client_id="test_client_id",
            mcp_server_name="test_oauth",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="test_state",
        )

    location = response.headers["location"]

    # Must NOT have double '?' — existing params must be merged correctly
    assert (
        location.count("?") == 1
    ), f"Expected exactly one '?' in URL but got {location.count('?')}: {location}"
    assert "tenant=system" in location
    assert "client_id=test_client_id" in location
    assert "response_type=code" in location
    assert "scope=read+write" in location


@pytest.mark.asyncio
async def test_authorize_endpoint_forwards_pkce_parameters():
    """Test that authorize endpoint forwards PKCE parameters (code_challenge and code_challenge_method)"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server (simulating Google OAuth)
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="669428968603-test.apps.googleusercontent.com",
        client_secret="GOCSPX-test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive", "openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"
    mock_request.headers = {}

    # Mock the encryption function
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state_with_pkce"

        # Call authorize endpoint with PKCE parameters
        response = await authorize(
            request=mock_request,
            client_id="669428968603-test.apps.googleusercontent.com",
            mcp_server_name="google_mcp",
            redirect_uri="http://localhost:60108/callback",
            state="test_client_state",
            code_challenge="x6YH_qgwbvOzbsHDuL1sW9gYkR9-gObUiIB5RkPwxDk",
            code_challenge_method="S256",
        )

    # Verify response is a redirect
    assert response.status_code == 307

    # Verify PKCE parameters are included in the redirect URL
    location = response.headers["location"]
    assert "https://accounts.google.com/o/oauth2/v2/auth" in location
    assert "code_challenge=x6YH_qgwbvOzbsHDuL1sW9gYkR9-gObUiIB5RkPwxDk" in location
    assert "code_challenge_method=S256" in location
    assert "client_id=669428968603-test.apps.googleusercontent.com" in location
    assert "response_type=code" in location


@pytest.mark.asyncio
async def test_token_endpoint_forwards_code_verifier():
    """Test that token endpoint forwards code_verifier for PKCE flow"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="669428968603-test.apps.googleusercontent.com",
        client_secret="GOCSPX-test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive", "openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"
    mock_request.headers = {}

    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "ya29.test_access_token",
        "token_type": "Bearer",
        "expires_in": 3599,
        "scope": "openid email https://www.googleapis.com/auth/drive",
    }
    mock_response.raise_for_status = MagicMock()

    # Mock the async httpx client with AsyncMock for async methods
    from unittest.mock import AsyncMock

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_async_client = MagicMock()
        # Use AsyncMock for the async post method
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_async_client

        # Call token endpoint with code_verifier
        response = await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="4/test_authorization_code",
            redirect_uri="http://localhost:60108/callback",
            client_id="669428968603-test.apps.googleusercontent.com",
            mcp_server_name="google_mcp",
            client_secret="GOCSPX-test_secret",
            code_verifier="test_code_verifier_from_client",
        )

    # Verify that the token endpoint was called with code_verifier
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args

    # Check the data parameter includes code_verifier
    assert call_args[1]["data"]["code_verifier"] == "test_code_verifier_from_client"
    assert call_args[1]["data"]["code"] == "4/test_authorization_code"
    assert (
        call_args[1]["data"]["client_id"]
        == "669428968603-test.apps.googleusercontent.com"
    )
    assert call_args[1]["data"]["client_secret"] == "GOCSPX-test_secret"
    assert call_args[1]["data"]["grant_type"] == "authorization_code"

    # Verify response
    response_data = response.body
    import json

    token_data = json.loads(response_data)
    assert token_data["access_token"] == "ya29.test_access_token"
    assert token_data["token_type"] == "Bearer"


@pytest.mark.asyncio
async def test_register_client_without_mcp_server_name_returns_dummy():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry to ensure no OAuth2 servers exist (otherwise resolver would find one)
    global_mcp_server_manager.registry.clear()

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
        new=AsyncMock(return_value={}),
    ):
        result = await register_client(request=mock_request)

    assert result == {
        "client_id": "dummy_client",
        "client_secret": "dummy",
        "redirect_uris": ["https://proxy.litellm.example/callback"],
    }


@pytest.mark.asyncio
async def test_register_client_returns_existing_server_credentials():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="stored_server",
        name="stored_server",
        server_name="stored_server",
        alias="stored_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="existing-client",
        client_secret="existing-secret",
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
            new=AsyncMock(return_value={}),
        ):
            result = await register_client(
                request=mock_request, mcp_server_name=oauth2_server.server_name
            )
    finally:
        global_mcp_server_manager.registry.clear()

    assert result == {
        "client_id": "stored_server",
        "client_secret": "dummy",
        "redirect_uris": ["https://proxy.litellm.example/callback"],
    }


@pytest.mark.asyncio
async def test_register_client_remote_registration_success():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="remote_server",
        name="remote_server",
        server_name="remote_server",
        alias="remote_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id=None,
        client_secret=None,
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
        registration_url="https://provider.example/oauth/register",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    request_payload = {
        "client_name": "Litellm Proxy",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "client_id": "generated-client",
        "client_secret": "generated-secret",
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
                new=AsyncMock(return_value=request_payload),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
                return_value=mock_async_client,
            ),
        ):
            response = await register_client(
                request=mock_request, mcp_server_name=oauth2_server.server_name
            )
    finally:
        global_mcp_server_manager.registry.clear()

    import json

    assert response.status_code == 200
    payload = json.loads(response.body.decode("utf-8"))
    assert payload == mock_response.json.return_value

    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args
    assert call_args.args[0] == oauth2_server.registration_url
    assert call_args.kwargs["headers"] == {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    assert call_args.kwargs["json"]["redirect_uris"] == [
        "https://proxy.litellm.example/callback"
    ]
    assert call_args.kwargs["json"]["grant_types"] == request_payload["grant_types"]
    assert (
        call_args.kwargs["json"]["token_endpoint_auth_method"]
        == request_payload["token_endpoint_auth_method"]
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_authorize_endpoint_respects_x_forwarded_proto():
    """Test that authorize endpoint uses X-Forwarded-Proto header to construct correct redirect_uri"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm.example.com/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy

    # Mock the encryption functions
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        # Call authorize endpoint
        response = await authorize(
            request=mock_request,
            client_id="test_client_id",
            mcp_server_name="test_oauth",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="test_state",
        )

    # Verify redirect URL uses HTTPS in the redirect_uri parameter
    location = response.headers["location"]

    # The redirect_uri parameter sent to the OAuth provider should use HTTPS
    assert (
        "redirect_uri=https%3A%2F%2Flitellm.example.com%2Fcallback" in location
        or "redirect_uri=https://litellm.example.com/callback" in location
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_token_endpoint_respects_x_forwarded_proto():
    """Test that token endpoint uses X-Forwarded-Proto header for redirect_uri"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm-proxy.example.com/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy

    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "test_token",
        "token_type": "Bearer",
        "expires_in": 3599,
    }
    mock_response.raise_for_status = MagicMock()

    # Mock the async httpx client
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_async_client

        await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="test_code",
            redirect_uri="http://localhost:60108/callback",
            client_id="test_client_id",
            mcp_server_name="google_mcp",
            client_secret="test_secret",
        )

    # Verify that the redirect_uri sent to the provider uses HTTPS
    call_args = mock_async_client.post.call_args
    assert (
        call_args[1]["data"]["redirect_uri"]
        == "https://litellm-proxy.example.com/callback"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_oauth_protected_resource_respects_x_forwarded_proto():
    """Test that oauth_protected_resource_mcp uses X-Forwarded-Proto for URLs"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_protected_resource_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")
    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm.example.com/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy

    # Call the endpoint
    response = await oauth_protected_resource_mcp(
        request=mock_request,
        mcp_server_name="test_oauth",
    )

    # Verify response uses HTTPS URLs
    assert response["authorization_servers"][0].startswith(
        "https://litellm.example.com/"
    )
    assert response["scopes_supported"] == oauth2_server.scopes


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_oauth_authorization_server_respects_x_forwarded_proto():
    """Test that oauth_authorization_server_mcp uses X-Forwarded-Proto for URLs"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_authorization_server_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")
    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm.example.com/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy

    # Call the endpoint
    response = await oauth_authorization_server_mcp(
        request=mock_request,
        mcp_server_name="test_oauth",
    )

    # Verify response uses HTTPS URLs
    assert response["authorization_endpoint"].startswith("https://litellm.example.com/")
    assert response["token_endpoint"].startswith("https://litellm.example.com/")
    assert response["registration_endpoint"].startswith("https://litellm.example.com/")
    assert response["grant_types_supported"] == ["authorization_code", "refresh_token"]
    assert response["scopes_supported"] == oauth2_server.scopes


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_register_client_respects_x_forwarded_proto():
    """Test that register_client uses X-Forwarded-Proto for redirect_uris"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry to ensure no OAuth2 servers exist (otherwise resolver would find one)
    global_mcp_server_manager.registry.clear()

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://proxy.litellm.example/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
        new=AsyncMock(return_value={}),
    ):
        result = await register_client(request=mock_request)

    # Verify the redirect_uris use HTTPS
    assert result == {
        "client_id": "dummy_client",
        "client_secret": "dummy",
        "redirect_uris": ["https://proxy.litellm.example/callback"],
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_authorize_endpoint_respects_x_forwarded_host():
    """Test that authorize endpoint uses X-Forwarded-Host and X-Forwarded-Proto to construct correct redirect_uri"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request simulating nginx proxy:
    # Internal: http://localhost:8888/github/mcp
    # External: https://proxy.example.com/github/mcp
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://localhost:8888/github/mcp"
    mock_request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "proxy.example.com",
    }

    # Mock the encryption functions
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        # Call authorize endpoint
        response = await authorize(
            request=mock_request,
            client_id="test_client_id",
            mcp_server_name="test_oauth",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="test_state",
        )

    # Verify redirect URL uses the forwarded host and scheme
    location = response.headers["location"]

    # The redirect_uri parameter should use the external URL
    assert (
        "redirect_uri=https%3A%2F%2Fproxy.example.com%2Fgithub%2Fmcp%2Fcallback"
        in location
        or "redirect_uri=https://proxy.example.com/github/mcp/callback" in location
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("trust_xff")
async def test_token_endpoint_respects_x_forwarded_host():
    """Test that token endpoint uses X-Forwarded-Host and X-Forwarded-Proto for redirect_uri"""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock request simulating nginx proxy without port in host
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://localhost:8888/github/mcp"
    mock_request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "proxy.example.com",
    }

    # Mock httpx client response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "test_token",
        "token_type": "Bearer",
        "expires_in": 3599,
    }
    mock_response.raise_for_status = MagicMock()

    # Mock the async httpx client
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_async_client

        await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="test_code",
            redirect_uri="http://localhost:60108/callback",
            client_id="test_client_id",
            mcp_server_name="google_mcp",
            client_secret="test_secret",
        )

    # Verify that the redirect_uri sent to the provider uses the external URL
    call_args = mock_async_client.post.call_args
    assert (
        call_args[1]["data"]["redirect_uri"]
        == "https://proxy.example.com/github/mcp/callback"
    )


@pytest.mark.parametrize(
    "base_url,x_forwarded_proto,x_forwarded_host,x_forwarded_port,expected_url",
    [
        # Case 1: No forwarded headers - use original URL as-is (no trailing slash)
        (
            "http://localhost:4000/",
            None,
            None,
            None,
            "http://localhost:4000",
        ),
        # Case 2: Only X-Forwarded-Proto - change scheme only
        (
            "http://localhost:4000/",
            "https",
            None,
            None,
            "https://localhost:4000",
        ),
        # Case 3: X-Forwarded-Proto + X-Forwarded-Host - change scheme and host
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com",
            None,
            "https://proxy.example.com",
        ),
        # Case 4: X-Forwarded-Host with port included in host header
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com:8080",
            None,
            "https://proxy.example.com:8080",
        ),
        # Case 5: X-Forwarded-Host + X-Forwarded-Port as separate headers
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com",
            "8443",
            "https://proxy.example.com:8443",
        ),
        # Case 6: Only X-Forwarded-Host without proto - use original scheme
        (
            "http://localhost:4000/",
            None,
            "proxy.example.com",
            None,
            "http://proxy.example.com",
        ),
        # Case 7: Only X-Forwarded-Port without host - preserves original port if present
        # (This is safer behavior - X-Forwarded-Port alone is unusual)
        (
            "http://localhost:4000/",
            None,
            None,
            "8443",
            "http://localhost:4000",  # Original port preserved when already present
        ),
        # Case 8: Complex internal URL with path (path is preserved)
        (
            "http://localhost:8888/github/mcp",
            "https",
            "proxy.example.com",
            None,
            "https://proxy.example.com/github/mcp",
        ),
        # Case 9: IPv6 address in X-Forwarded-Host (should not treat :: as port separator)
        (
            "http://localhost:4000/",
            "https",
            "[2001:db8::1]",
            None,
            "https://[2001:db8::1]",
        ),
        # Case 10: IPv6 address with port
        (
            "http://localhost:4000/",
            "https",
            "[2001:db8::1]:8080",
            None,
            "https://[2001:db8::1]:8080",
        ),
        # Case 11: X-Forwarded-Host already has port, X-Forwarded-Port also provided (host wins)
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com:9000",
            "8443",
            "https://proxy.example.com:9000",
        ),
        # Case 12: Standard proxy setup (most common case)
        (
            "http://127.0.0.1:8888/",
            "https",
            "chatproxy.company.com",
            None,
            "https://chatproxy.company.com",
        ),
        # Case 13: Internal URL already has port, X-Forwarded-Port does NOT override
        # (safer behavior - preserves original port when X-Forwarded-Host not provided)
        (
            "http://localhost:4000/",
            None,
            None,
            "443",
            "http://localhost:4000",  # Original port preserved
        ),
        # Case 14: Original URL with existing port in netloc, X-Forwarded-Host replaces it
        (
            "http://internal.local:8888/",
            "https",
            "external.com",
            None,
            "https://external.com",
        ),
    ],
)
def test_get_request_base_url_comprehensive(
    base_url, x_forwarded_proto, x_forwarded_host, x_forwarded_port, expected_url
):
    """Comprehensive test for get_request_base_url with various header combinations.

    These cases exercise the X-Forwarded-* parsing logic, so the trust gate
    is patched True; the gate's own behaviour is covered by the
    ``test_get_request_base_url_xff_trust_gate`` matrix below.
    """
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            get_request_base_url,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = base_url

    headers = {}
    if x_forwarded_proto:
        headers["X-Forwarded-Proto"] = x_forwarded_proto
    if x_forwarded_host:
        headers["X-Forwarded-Host"] = x_forwarded_host
    if x_forwarded_port:
        headers["X-Forwarded-Port"] = x_forwarded_port

    def mock_get(header_name, default=None):
        return headers.get(header_name, default)

    mock_request.headers.get = mock_get

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.is_request_from_trusted_proxy",
        return_value=True,
    ):
        result = get_request_base_url(mock_request)

    assert result == expected_url, (
        f"Expected '{expected_url}' but got '{result}'\n"
        f"Input: base_url={base_url}, "
        f"X-Forwarded-Proto={x_forwarded_proto}, "
        f"X-Forwarded-Host={x_forwarded_host}, "
        f"X-Forwarded-Port={x_forwarded_port}"
    )


@pytest.mark.parametrize(
    "general_settings,direct_ip,expect_xff_honoured",
    [
        # Default: use_x_forwarded_for not set -> ignore X-Forwarded-* entirely.
        ({}, "127.0.0.1", False),
        # XFF enabled, no trusted ranges -> still ignored (no way to tell a trusted
        # reverse proxy from a direct attacker).
        ({"use_x_forwarded_for": True}, "127.0.0.1", False),
        # XFF enabled, ranges set, but caller IP outside any range -> ignored.
        (
            {
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
            "203.0.113.5",
            False,
        ),
        # XFF enabled, caller in trusted range -> headers honoured.
        (
            {
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
            },
            "10.0.0.7",
            True,
        ),
        # Loopback example (common dev / single-host deploy).
        (
            {
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": ["127.0.0.0/8"],
            },
            "127.0.0.1",
            True,
        ),
    ],
)
def test_get_request_base_url_xff_trust_gate(
    general_settings, direct_ip, expect_xff_honoured
):
    """Verify the X-Forwarded-* trust gate.

    With XFF poisoning attempted, the helper must return either the literal
    base_url (gate denies) or the forwarded URL (gate allows), never the
    forwarded URL when the gate denies.
    """
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            get_request_base_url,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://localhost:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = direct_ip

    headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "attacker.example.com",
    }
    mock_request.headers.get = lambda name, default=None: headers.get(name, default)
    mock_request.headers.__contains__ = lambda self_, name: name in headers

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        general_settings,
        create=True,
    ):
        result = get_request_base_url(mock_request)

    if expect_xff_honoured:
        assert result == "https://attacker.example.com"
    else:
        assert result == "http://localhost:4000"


def test_xff_misconfig_warning_emitted_once(caplog):
    """Operators upgrading from the old "always trust X-Forwarded-*" behaviour
    get a one-shot warning when they have ``use_x_forwarded_for`` enabled
    but no ``mcp_trusted_proxy_ranges`` configured. The warning must NOT
    spam every request."""
    try:
        from fastapi import Request

        from litellm.proxy import auth as proxy_auth_pkg  # noqa: F401
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            get_request_base_url,
        )
        from litellm.proxy.auth import ip_address_utils
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Reset the module-level one-shot flag so the test is deterministic.
    ip_address_utils._warned_xff_without_trusted_ranges = False

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://localhost:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = "203.0.113.5"
    headers = {"X-Forwarded-Host": "attacker.example.com"}
    mock_request.headers.get = lambda name, default=None: headers.get(name, default)

    misconfig = {"use_x_forwarded_for": True}

    import logging

    with (
        caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"),
        patch("litellm.proxy.proxy_server.general_settings", misconfig, create=True),
    ):
        for _ in range(3):
            get_request_base_url(mock_request)

    matching = [
        rec for rec in caplog.records if "mcp_trusted_proxy_ranges" in rec.getMessage()
    ]
    assert (
        len(matching) == 1
    ), f"expected exactly one warning, got {len(matching)}: {[r.getMessage() for r in matching]}"


# -------------------------------------------------------------------
# Tests for scopes_supported when mcp_server.scopes is None
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oauth_protected_resource_returns_empty_scopes_when_none():
    """
    When an MCP server exists but has scopes=None (e.g. Atlassian OAuth),
    scopes_supported should be [] not None.
    """
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_protected_resource_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()

    # Create an OAuth2 server with scopes=None (like Atlassian)
    oauth2_server = MCPServer(
        server_id="atlassian_mcp",
        name="atlassian_mcp",
        server_name="atlassian_mcp",
        alias="atlassian_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="atlassian_client_id",
        client_secret="atlassian_secret",
        authorization_url="https://auth.atlassian.com/authorize",
        token_url="https://auth.atlassian.com/oauth/token",
        scopes=None,  # Atlassian doesn't set scopes
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        response = _build_oauth_protected_resource_response(
            request=mock_request,
            mcp_server_name="atlassian_mcp",
            use_standard_pattern=False,
        )
        assert response["scopes_supported"] == []
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_authorization_server_returns_empty_scopes_when_none():
    """
    When an MCP server exists but has scopes=None (e.g. Atlassian OAuth),
    scopes_supported should be [] not None.
    """
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_authorization_server_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()

    # Create an OAuth2 server with scopes=None
    oauth2_server = MCPServer(
        server_id="atlassian_mcp",
        name="atlassian_mcp",
        server_name="atlassian_mcp",
        alias="atlassian_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="atlassian_client_id",
        client_secret="atlassian_secret",
        authorization_url="https://auth.atlassian.com/authorize",
        token_url="https://auth.atlassian.com/oauth/token",
        scopes=None,
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        response = _build_oauth_authorization_server_response(
            request=mock_request,
            mcp_server_name="atlassian_mcp",
        )
        assert response["scopes_supported"] == []
    finally:
        global_mcp_server_manager.registry.clear()


# -------------------------------------------------------------------
# Tests for root-level OAuth endpoint resolution (no server name)
# -------------------------------------------------------------------


def _create_oauth2_server(
    server_id="test_oauth_server",
    name="test_oauth",
    server_name="test_oauth",
    alias="test_oauth",
    client_id="test_client_id",
    client_secret="test_client_secret",
    available_on_public_internet=True,
):
    """Helper to create a mock OAuth2 MCPServer."""
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id=server_id,
        name=name,
        server_name=server_name,
        alias=alias,
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id=client_id,
        client_secret=client_secret,
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        scopes=["read", "write"],
        available_on_public_internet=available_on_public_internet,
    )


@pytest.mark.asyncio
async def test_authorize_root_resolves_single_oauth2_server():
    """When /authorize is hit without server name and exactly 1 OAuth2 server exists, resolve it."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server()
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
        ) as mock_encrypt:
            mock_encrypt.return_value = "mocked_encrypted_state"

            # Call /authorize WITHOUT mcp_server_name, with dummy_client as client_id
            response = await authorize(
                request=mock_request,
                client_id="dummy_client",
                mcp_server_name=None,
                redirect_uri="http://localhost:62646/callback",
                state="test_state",
            )

        # Should resolve to the single OAuth2 server and redirect
        assert response.status_code == 307
        location = response.headers["location"]
        assert "https://provider.com/oauth/authorize" in location
        assert "client_id=test_client_id" in location
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_authorize_root_fails_with_multiple_oauth2_servers():
    """When /authorize is hit without server name and multiple OAuth2 servers exist, return 404."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server1 = _create_oauth2_server(
        server_id="server1", name="server1", server_name="server1", alias="server1"
    )
    server2 = _create_oauth2_server(
        server_id="server2", name="server2", server_name="server2", alias="server2"
    )
    global_mcp_server_manager.registry[server1.server_id] = server1
    global_mcp_server_manager.registry[server2.server_id] = server2

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorize(
                request=mock_request,
                client_id="dummy_client",
                mcp_server_name=None,
                redirect_uri="http://localhost:62646/callback",
                state="test_state",
            )
        assert exc_info.value.status_code == 404
        assert "MCP server not found" in str(exc_info.value.detail)
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_authorize_root_does_not_resolve_private_server_for_external_client():
    """Root /authorize must not auto-select an MCP server hidden from the caller IP."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server(available_on_public_internet=False)
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
            return_value="198.51.100.10",
        ):
            with pytest.raises(HTTPException) as exc_info:
                await authorize(
                    request=mock_request,
                    client_id="dummy_client",
                    mcp_server_name=None,
                    redirect_uri="http://localhost:62646/callback",
                    state="test_state",
                )
        assert exc_info.value.status_code == 404
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_token_root_resolves_single_oauth2_server():
    """When /token is hit without server name and exactly 1 OAuth2 server exists, resolve it."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server()
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "ya29.test_token",
        "token_type": "Bearer",
        "expires_in": 3599,
    }
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_async_client

            # Call /token WITHOUT mcp_server_name
            response = await token_endpoint(
                request=mock_request,
                grant_type="authorization_code",
                code="test_auth_code",
                redirect_uri="http://localhost:62646/callback",
                client_id="dummy_client",
                mcp_server_name=None,
                client_secret=None,
                code_verifier="test_verifier",
            )

        # Should resolve and exchange token with the upstream server
        import json

        token_data = json.loads(response.body)
        assert token_data["access_token"] == "ya29.test_token"

        # Verify it called the correct upstream token URL
        call_args = mock_async_client.post.call_args
        assert call_args.args[0] == "https://provider.com/oauth/token"
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_token_root_does_not_resolve_private_server_for_external_client():
    """Root /token must not exchange codes for a hidden MCP server."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server(available_on_public_internet=False)
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
            return_value="198.51.100.10",
        ):
            with pytest.raises(HTTPException) as exc_info:
                await token_endpoint(
                    request=mock_request,
                    grant_type="authorization_code",
                    code="test_auth_code",
                    redirect_uri="http://localhost:62646/callback",
                    client_id="dummy_client",
                    mcp_server_name=None,
                    client_secret=None,
                    code_verifier="test_verifier",
                )
        assert exc_info.value.status_code == 404
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_register_root_resolves_single_oauth2_server():
    """When /register is hit without server name and exactly 1 OAuth2 server exists, resolve it."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server()
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
            new=AsyncMock(return_value={}),
        ):
            result = await register_client(request=mock_request, mcp_server_name=None)

        # Should resolve to the single server and return its name as client_id
        assert result["client_id"] == "test_oauth"
        assert "redirect_uris" in result
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_register_root_does_not_resolve_private_server_for_external_client():
    """Root /register must not reveal or use a hidden MCP server."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server(available_on_public_internet=False)
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
                return_value="198.51.100.10",
            ),
        ):
            result = await register_client(request=mock_request, mcp_server_name=None)

        assert result["client_id"] == "dummy_client"
        assert result["redirect_uris"] == ["https://llm.example.com/callback"]
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_discovery_root_includes_server_name_prefix():
    """When root discovery is hit and exactly 1 OAuth2 server exists, include server name in URLs."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_authorization_server_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server()
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        # Call with mcp_server_name=None (root discovery)
        response = _build_oauth_authorization_server_response(
            request=mock_request,
            mcp_server_name=None,
        )

        # Should resolve to the single server and include its name in endpoint URLs
        assert "/test_oauth/authorize" in response["authorization_endpoint"]
        assert "/test_oauth/token" in response["token_endpoint"]
        assert "/test_oauth/register" in response["registration_endpoint"]
        assert response["scopes_supported"] == ["read", "write"]
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_discovery_root_does_not_expose_private_server_for_external_client():
    """Root discovery must use caller visibility before adding server-specific metadata."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_authorization_server_response,
            _build_oauth_protected_resource_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    oauth2_server = _create_oauth2_server(available_on_public_internet=False)
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://llm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
            return_value="198.51.100.10",
        ):
            authorization_response = _build_oauth_authorization_server_response(
                request=mock_request,
                mcp_server_name=None,
            )
            resource_response = _build_oauth_protected_resource_response(
                request=mock_request,
                mcp_server_name=None,
                use_standard_pattern=False,
            )

        assert "/test_oauth/" not in authorization_response["authorization_endpoint"]
        assert "/test_oauth/" not in authorization_response["token_endpoint"]
        assert authorization_response["scopes_supported"] == []
        assert resource_response["authorization_servers"] == ["https://llm.example.com"]
        assert resource_response["scopes_supported"] == []
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_callback_redirects_with_state():
    """Test OAuth callback endpoint properly decodes state and redirects to client callback URL."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            callback,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Mock the state decoding
    mock_state_data = {
        "base_url": "http://localhost:3000/ui/mcp/oauth/callback",
        "original_state": "test-uuid-state-123",
        "code_challenge": "test_challenge",
        "code_challenge_method": "S256",
        "client_redirect_uri": "http://localhost:3000/ui/mcp/oauth/callback",
    }

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = mock_state_data

        # Call callback endpoint with code and state
        response = await callback(
            request=_mock_callback_request(),
            code="test_authorization_code_12345",
            state="encrypted_state_value",
        )

        # Should redirect to the client callback URL with code and original state
        assert response.status_code == 302
        assert (
            "http://localhost:3000/ui/mcp/oauth/callback"
            in response.headers["location"]
        )
        assert "code=test_authorization_code_12345" in response.headers["location"]
        assert "state=test-uuid-state-123" in response.headers["location"]

        # Verify state was decoded
        mock_decode.assert_called_once_with("encrypted_state_value")


@pytest.mark.asyncio
async def test_oauth_callback_preserves_client_redirect_uri_query():
    """The callback should append code/state without dropping a client's existing query."""
    try:
        from urllib.parse import parse_qs, urlparse

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            callback,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "http://localhost:3000/ui/mcp/oauth/callback",
            "original_state": "test-uuid-state-123",
            "code_challenge": "test_challenge",
            "code_challenge_method": "S256",
            "client_redirect_uri": (
                "http://localhost:3000/ui/mcp/oauth/callback?session=abc"
            ),
        }

        response = await callback(
            request=_mock_callback_request(),
            code="test_authorization_code_12345",
            state="encrypted_state_value",
        )

    assert response.status_code == 302
    parsed_location = urlparse(response.headers["location"])
    query_params = parse_qs(parsed_location.query)
    assert query_params["session"] == ["abc"]
    assert query_params["code"] == ["test_authorization_code_12345"]
    assert query_params["state"] == ["test-uuid-state-123"]


@pytest.mark.asyncio
async def test_oauth_callback_handles_invalid_state():
    """Test OAuth callback returns error page when state decryption fails."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            callback,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Mock state decoding to raise an exception
    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.side_effect = Exception("Failed to decrypt state")

        # Call callback endpoint with invalid state
        response = await callback(
            request=_mock_callback_request(),
            code="test_code",
            state="invalid_encrypted_state",
        )

        # Should return HTML error page
        assert response.status_code == 200
        assert "Authentication incomplete" in response.body.decode()


@pytest.mark.asyncio
async def test_oauth_callback_accepts_same_origin_ui_redirect():
    """UI OAuth flow: the callback should redirect to the proxy's own UI
    origin when the encrypted state carries a same-origin client_redirect_uri."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "https://proxy.example.com/ui/mcp/oauth/callback",
            "original_state": "state-123",
            "code_challenge": None,
            "code_challenge_method": None,
            "client_redirect_uri": "https://proxy.example.com/ui/mcp/oauth/callback",
        }

        response = await callback(
            request=_mock_callback_request(base_url="https://proxy.example.com/"),
            code="auth-code-123",
            state="encrypted_state",
        )

    assert response.status_code == 302
    assert (
        "https://proxy.example.com/ui/mcp/oauth/callback"
        in response.headers["location"]
    )
    assert "code=auth-code-123" in response.headers["location"]
    assert "state=state-123" in response.headers["location"]


@pytest.mark.asyncio
async def test_oauth_authorize_includes_scopes_from_server_config():
    """Test that authorize endpoint includes scopes from server configuration."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize_with_server,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Create server with specific scopes (e.g., GitLab requires 'ai_workflows')
    oauth_server = MCPServer(
        server_id="gitlab_server",
        name="gitlab",
        server_name="gitlab",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        authorization_url="https://gitlab.com/oauth/authorize",
        token_url="https://gitlab.com/oauth/token",
        client_id="test_client",
        scopes=["api", "read_user", "ai_workflows"],  # GitLab-specific scopes
    )

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "encrypted_state"

        # Call authorize without explicit scope parameter
        response = await authorize_with_server(
            request=mock_request,
            mcp_server=oauth_server,
            client_id="test_client",
            redirect_uri="http://localhost:3000/callback",
            state="test_state",
            code_challenge="test_challenge",
            code_challenge_method="S256",
            response_type="code",
            scope=None,  # No scope in request, should use server's scopes
        )

        # Should redirect with scopes from server config
        assert response.status_code in (307, 302)
        redirect_url = response.headers["location"]
        assert (
            "scope=api+read_user+ai_workflows" in redirect_url
            or "scope=api%20read_user%20ai_workflows" in redirect_url
        )


@pytest.mark.asyncio
async def test_oauth_authorize_prefers_request_scope_over_server_config():
    """Test that explicit scope parameter takes precedence over server configuration."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize_with_server,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    oauth_server = MCPServer(
        server_id="test_server",
        name="test",
        server_name="test",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
        client_id="test_client",
        scopes=["default_scope1", "default_scope2"],
    )

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
    ) as mock_encrypt:
        mock_encrypt.return_value = "encrypted_state"

        # Call authorize WITH explicit scope parameter
        response = await authorize_with_server(
            request=mock_request,
            mcp_server=oauth_server,
            client_id="test_client",
            redirect_uri="http://localhost:3000/callback",
            state="test_state",
            code_challenge="test_challenge",
            code_challenge_method="S256",
            response_type="code",
            scope="custom_scope1 custom_scope2",  # Explicit scope should take precedence
        )

        # Should use the explicit scope, not server config
        assert response.status_code in (307, 302)
        redirect_url = response.headers["location"]
        assert (
            "scope=custom_scope1+custom_scope2" in redirect_url
            or "scope=custom_scope1%20custom_scope2" in redirect_url
        )
        assert "default_scope" not in redirect_url


@pytest.mark.asyncio
async def test_token_endpoint_refresh_token_grant():
    """Test that token endpoint supports refresh_token grant type."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="google_mcp",
        name="google_mcp",
        server_name="google_mcp",
        alias="google_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="test_client_id",
        client_secret="test_secret",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["openid", "email"],
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    # Mock httpx client response with new tokens
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_access_token",
        "token_type": "Bearer",
        "expires_in": 3599,
        "refresh_token": "new_refresh_token",
    }
    mock_response.raise_for_status = MagicMock()

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_async_client

        response = await token_endpoint(
            request=mock_request,
            grant_type="refresh_token",
            code=None,
            redirect_uri=None,
            client_id="test_client_id",
            mcp_server_name="google_mcp",
            client_secret="test_secret",
            refresh_token="rt-test",
            scope="openid email",
        )

    # Verify the POST was called with refresh_token grant data
    mock_async_client.post.assert_called_once()
    call_args = mock_async_client.post.call_args

    assert call_args[1]["data"]["grant_type"] == "refresh_token"
    assert call_args[1]["data"]["refresh_token"] == "rt-test"
    assert call_args[1]["data"]["client_id"] == "test_client_id"
    assert call_args[1]["data"]["client_secret"] == "test_secret"
    assert call_args[1]["data"]["scope"] == "openid email"

    # Verify response contains the new tokens
    import json

    token_data = json.loads(response.body)
    assert token_data["access_token"] == "new_access_token"
    assert token_data["refresh_token"] == "new_refresh_token"


@pytest.mark.asyncio
async def test_token_endpoint_authorization_code_missing_code():
    """Test that authorization_code grant rejects missing code param."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            exchange_token_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()

    server = MCPServer(
        server_id="test_server",
        name="test_server",
        server_name="test_server",
        alias="test_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        token_url="https://example.com/token",
    )
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://proxy.example/"
    mock_request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        await exchange_token_with_server(
            request=mock_request,
            mcp_server=server,
            grant_type="authorization_code",
            code=None,
            redirect_uri="https://example.com/cb",
            client_id="cid",
            client_secret=None,
            code_verifier=None,
        )
    assert exc_info.value.status_code == 400
    assert "code is required" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_authorize_endpoint_rejects_non_loopback_redirect_uri():
    """VERIA-57 root cause B regression. The client-supplied redirect_uri
    is encrypted into the OAuth state and decoded on /callback to 302 the
    user back. A non-loopback value is an open-redirect + code-theft
    primitive — reject with 400 before encoding anything into state."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        client_secret="cs",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        await authorize(
            request=mock_request,
            client_id="cid",
            mcp_server_name="test_oauth",
            redirect_uri="https://attacker.example.com/cb",
            state="s",
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_authorize_endpoint_accepts_ipv4_loopback_range_and_ipv6_full_form():
    """RFC 8252 §7.3 + RFC 4291: full 127.0.0.0/8 and full-form IPv6
    loopback must be accepted — string match on ``127.0.0.1`` alone
    would miss ``127.0.0.2`` and ``0:0:0:0:0:0:0:1``."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="test_oauth_server",
        name="test_oauth",
        server_name="test_oauth",
        alias="test_oauth",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        client_secret="cs",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    for uri in (
        "http://127.0.0.2:3000/cb",
        "http://[0:0:0:0:0:0:0:1]:3000/cb",
        "http://localhost:3000/cb",
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper"
        ) as mock_encrypt:
            mock_encrypt.return_value = "mocked_encrypted_state"
            response = await authorize(
                request=mock_request,
                client_id="cid",
                mcp_server_name="test_oauth",
                redirect_uri=uri,
                state="s",
            )
        assert response.status_code == 307, f"{uri} should be accepted"


@pytest.mark.asyncio
async def test_callback_revalidates_loopback_on_decoded_base_url():
    """VERIA-57 root cause B defense-in-depth: an encrypted state minted
    before the /authorize validation was added has no expiry and stays
    valid. /callback must re-validate the decoded base_url so those
    stale states can't be used as an open-redirect + code-theft
    primitive."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "https://attacker.example.com/cb",
            "original_state": "s",
            "code_challenge": None,
            "code_challenge_method": None,
            "client_redirect_uri": "https://attacker.example.com/cb",
        }
        with pytest.raises(HTTPException) as exc_info:
            await callback(
                request=_mock_callback_request(),
                code="stolen_code",
                state="encrypted_stale_state",
            )
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_callback_revalidates_loopback_on_decoded_client_redirect_uri():
    """If a state contains a full client_redirect_uri, validate that exact sink."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "base_url": "http://localhost:3000/cb",
            "original_state": "s",
            "code_challenge": None,
            "code_challenge_method": None,
            "client_redirect_uri": "https://attacker.example.com/cb",
        }
        with pytest.raises(HTTPException) as exc_info:
            await callback(
                request=_mock_callback_request(),
                code="stolen_code",
                state="encrypted_stale_state",
            )
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_callback_rejects_state_missing_redirect_uri():
    """Malformed state without a redirect target should fail with a structured 400."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        callback,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash"
    ) as mock_decode:
        mock_decode.return_value = {
            "original_state": "s",
            "code_challenge": None,
            "code_challenge_method": None,
        }
        with pytest.raises(HTTPException) as exc_info:
            await callback(
                request=_mock_callback_request(),
                code="code",
                state="encrypted_malformed_state",
            )
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_token_endpoint_sets_no_store_cache_control():
    """RFC 6749 §5.1 / OAuth 2.1 draft-15 §4.1.3: the token response
    contains an access token (and possibly a refresh token) — it MUST
    NOT be cached by intermediaries or the client."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        exchange_token_with_server,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="t",
        name="t",
        server_name="t",
        alias="t",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="cid",
        client_secret="cs",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
    )
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    fake_http_response = MagicMock()
    fake_http_response.json.return_value = {
        "access_token": "tok",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    fake_http_response.raise_for_status = MagicMock()
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock(return_value=fake_http_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
        return_value=fake_http_client,
    ):
        response = await exchange_token_with_server(
            request=mock_request,
            mcp_server=server,
            grant_type="authorization_code",
            code="c",
            redirect_uri="http://127.0.0.1:3000/cb",
            client_id="cid",
            client_secret=None,
            code_verifier=None,
        )

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
