"""Tests for MCP OAuth discoverable endpoints"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.types.mcp import MCPAuth


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
    req.cookies = {}
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
    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
@pytest.mark.parametrize("auth_type_value", ["true_passthrough", "oauth_delegate"])
async def test_authorize_endpoint_allows_client_forwarded_modes(auth_type_value):
    """The browser-only Authorize relays the gateway authorize flow for the client-forwarded
    token modes; the oauth2-only gate must let them through and redirect to the upstream IdP."""
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

    server = MCPServer(
        server_id="test_cf_server",
        name="test_cf",
        server_name="test_cf",
        alias="test_cf",
        transport=MCPTransport.http,
        auth_type=MCPAuth(auth_type_value),
        # Discovery stamps these onto the in-memory registry entry at build time.
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
    )
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
        mock_encrypt.return_value = "mocked_encrypted_state"

        response = await authorize(
            request=mock_request,
            client_id="dcr_client_id",
            mcp_server_name="test_cf",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="test_state",
        )

    assert response.status_code == 307
    assert "https://provider.com/oauth/authorize" in response.headers["location"]
    assert "client_id=dcr_client_id" in response.headers["location"]


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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
    assert location.count("?") == 1, f"Expected exactly one '?' in URL but got {location.count('?')}: {location}"
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
    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
    assert call_args[1]["data"]["client_id"] == "669428968603-test.apps.googleusercontent.com"
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
            result = await register_client(request=mock_request, mcp_server_name=oauth2_server.server_name)
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
            response = await register_client(request=mock_request, mcp_server_name=oauth2_server.server_name)
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
    assert call_args.kwargs["json"]["redirect_uris"] == ["https://proxy.litellm.example/callback"]
    assert call_args.kwargs["json"]["grant_types"] == request_payload["grant_types"]
    assert call_args.kwargs["json"]["token_endpoint_auth_method"] == request_payload["token_endpoint_auth_method"]


@pytest.mark.asyncio
async def test_register_client_persists_dcr_client_identity():
    """A dynamic client registration (RFC 7591) must persist the issued client_id /
    client_secret / token_endpoint_auth_method and the token_url onto the server row so
    autonomous refresh can authenticate as the registered client. Without persistence the
    minted client_id is discarded and the refresh_token grant has no client identity.

    The persist must also stamp oauth2_flow="authorization_code": only the interactive
    flow reaches this persist, and without the stamp the row (client creds + token_url,
    no persisted authorization_url) matches the legacy M2M inference whenever endpoint
    discovery fails at registry build, flipping the server to client_credentials."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "client_id": "generated-client",
        "client_secret": "generated-secret",
        "token_endpoint_auth_method": "client_secret_basic",
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    mock_update = AsyncMock(return_value=MagicMock())
    mock_update_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.update_mcp_server", new=mock_update),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            persist_credentials=True,
        )

    import json

    assert response.status_code == 200
    assert json.loads(response.body.decode("utf-8")) == mock_response.json.return_value

    mock_update.assert_called_once()
    update_data = mock_update.call_args.kwargs["data"]
    assert update_data.server_id == "remote_server"
    assert update_data.token_url == "https://provider.example/oauth/token"
    assert update_data.credentials["client_id"] == "generated-client"
    assert update_data.credentials["client_secret"] == "generated-secret"
    assert update_data.credentials["token_endpoint_auth_method"] == "client_secret_basic"
    assert update_data.credentials["redirect_uris"] == ["https://proxy.litellm.example/callback"]
    assert update_data.oauth2_flow == "authorization_code"

    mock_update_server.assert_called_once()


async def _register_persistence_attempted_for_auth_type(auth_type: MCPAuth) -> bool:
    """Run register_client_with_server with persist_credentials=True for a server of ``auth_type``
    and report whether the DCR result was persisted onto the server row. The client-forwarded token
    modes must skip the persist even on the admin path: writing it stamps oauth2_flow and a
    client_id onto a server whose contract is that the gateway stores nothing, which makes a fresh
    pass-through server read as gateway-authorized. The upstream registration must still be relayed
    to the browser either way, since the caller needs the minted client to run its own flow."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        register_client_with_server,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="pt_server",
        name="pt_server",
        server_name="pt_server",
        alias="pt_server",
        transport=MCPTransport.http,
        auth_type=auth_type,
        client_id=None,
        client_secret=None,
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
        registration_url="https://provider.example/oauth/register",
    )

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "client_id": "generated-client",
        "client_secret": "generated-secret",
        "token_endpoint_auth_method": "none",
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    mock_update = AsyncMock(return_value=MagicMock())

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.update_mcp_server", new=mock_update),
        patch.object(global_mcp_server_manager, "update_server", new=AsyncMock()),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    assert json.loads(response.body.decode("utf-8")) == mock_response.json.return_value
    return mock_update.await_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
async def test_register_client_does_not_persist_for_client_forwarded_modes(auth_type):
    """The admin Authorize path (persist_credentials=True) must not write the DCR client onto a
    true_passthrough / oauth_delegate server row: the browser still receives the registration, but
    the gateway keeps no OAuth client identity for these modes."""
    assert await _register_persistence_attempted_for_auth_type(auth_type) is False


@pytest.mark.asyncio
async def test_register_client_persist_discriminator_oauth2_persists():
    """Guard the no-persist assertion above against vacuity: the same helper run against a genuine
    oauth2 server DOES persist, so a regression that silently disables persistence everywhere (or a
    helper that never reaches the persist) fails here instead of passing both."""
    assert await _register_persistence_attempted_for_auth_type(MCPAuth.oauth2) is True


@pytest.mark.asyncio
async def test_register_client_persists_only_to_its_own_row_when_another_server_shares_the_url():
    """A fresh server must mint and persist its OWN DCR client even when another server row with
    the same upstream URL already holds one: both the reuse lookup and the persist are keyed by
    server_id, never by URL, so OAuth client identity is not transferable between server entries.
    If either side ever falls back to a URL match, this fails: the fresh server would skip the
    upstream registration (adopting the sibling's client) or persist onto the wrong row."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        register_client_with_server,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    shared_url = "https://provider.example/mcp"
    fresh_server = MCPServer(
        server_id="server-b",
        name="server-b",
        server_name="server-b",
        alias="server-b",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        url=shared_url,
        client_id=None,
        client_secret=None,
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
        registration_url="https://provider.example/oauth/register",
    )
    sibling_row_with_client = MagicMock(server_id="server-a", url=shared_url)
    sibling_row_with_client.credentials = {"client_id": "client-a-do-not-adopt"}
    own_row_without_client = MagicMock(server_id="server-b", url=shared_url)
    own_row_without_client.credentials = {}
    rows_by_server_id = {"server-a": sibling_row_with_client, "server-b": own_row_without_client}

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {"client_id": "fresh-client-b", "token_endpoint_auth_method": "none"}
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    mock_update = AsyncMock(return_value=MagicMock())

    async def _get_row(prisma_client, server_id):
        return rows_by_server_id.get(server_id)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.get_mcp_server", new=AsyncMock(side_effect=_get_row)),
        patch("litellm.proxy._experimental.mcp_server.db.update_mcp_server", new=mock_update),
        patch.object(global_mcp_server_manager, "update_server", new=AsyncMock()),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=fresh_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    mock_async_client.post.assert_called_once()
    body = json.loads(response.body.decode("utf-8"))
    assert body["client_id"] == "fresh-client-b"

    mock_update.assert_called_once()
    update_data = mock_update.call_args.kwargs["data"]
    assert update_data.server_id == "server-b"
    assert update_data.credentials["client_id"] == "fresh-client-b"


@pytest.mark.asyncio
async def test_register_client_does_not_clobber_token_url_when_absent():
    """When the in-memory server has no token_url, the DCR persist must omit it from the
    partial update rather than passing None, so exclude_unset leaves the token_url column
    untouched instead of overwriting an existing value with NULL."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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
        token_url=None,
        registration_url="https://provider.example/oauth/register",
    )

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {"client_id": "generated-client"}
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    mock_update = AsyncMock(return_value=MagicMock())
    mock_update_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch("litellm.proxy._experimental.mcp_server.db.update_mcp_server", new=mock_update),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    mock_update.assert_called_once()
    update_data = mock_update.call_args.kwargs["data"]
    assert update_data.credentials["client_id"] == "generated-client"
    assert "token_url" not in update_data.model_fields_set


@pytest.mark.asyncio
async def test_register_client_reuses_persisted_client_id_for_non_admin_when_registry_is_stale():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    persisted_server = MagicMock()
    persisted_server.credentials = {"client_id": "persisted-client"}
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_update_mcp_server = AsyncMock()
    mock_update_server = AsyncMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.update_mcp_server",
            new=mock_update_mcp_server,
        ),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=False,
        )

    assert response["client_id"] == "remote_server"
    assert oauth2_server.client_id == "persisted-client"
    mock_async_client.post.assert_not_called()
    mock_update_mcp_server.assert_not_called()
    mock_update_server.assert_called_once_with(persisted_server)


@pytest.mark.asyncio
async def test_register_client_reuse_refreshes_request_server_when_manager_update_fails():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    persisted_server = MagicMock()
    persisted_server.credentials = {
        "client_id": "persisted-client",
        "client_secret": "persisted-secret",
        "token_endpoint_auth_method": "client_secret_basic",
    }
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()
    mock_update_server = AsyncMock(side_effect=RuntimeError("registry update failed"))

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=False,
        )

    assert response["client_id"] == "remote_server"
    assert oauth2_server.client_id == "persisted-client"
    assert oauth2_server.client_secret == "persisted-secret"
    assert oauth2_server.token_endpoint_auth_method == "client_secret_basic"
    mock_async_client.post.assert_not_called()
    mock_update_server.assert_called_once_with(persisted_server)


@pytest.mark.asyncio
async def test_register_client_returns_reused_client_when_concurrent_persist_wins():
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {"client_id": "generated-client"}
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    persisted_server = MagicMock()
    persisted_server.credentials = {"client_id": "persisted-client"}
    mock_get_mcp_server = AsyncMock(side_effect=[None, persisted_server])
    mock_update_mcp_server = AsyncMock()
    mock_update_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.update_mcp_server",
            new=mock_update_mcp_server,
        ),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    assert response["client_id"] == "remote_server"
    assert oauth2_server.client_id == "persisted-client"
    mock_async_client.post.assert_called_once()
    mock_update_mcp_server.assert_not_called()
    mock_update_server.assert_called_once_with(persisted_server)


def _dcr_redirect_test_server(client_id):
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id="remote_server",
        name="remote_server",
        server_name="remote_server",
        alias="remote_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id=client_id,
        client_secret=None,
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
        registration_url="https://provider.example/oauth/register",
    )


@pytest.mark.asyncio
async def test_register_client_re_registers_when_persisted_redirect_uri_no_longer_matches_origin():
    """A persisted DCR client is bound to the redirect_uri it was registered with. When the
    proxy's resolved public origin changes, every authorize built for the reused client is
    rejected IdP-side and the server is permanently stranded (GH #32473). A positive mismatch
    between the recorded redirect_uris and the current callback must therefore re-register on
    the admin path and persist the replacement client, with the new binding recorded and the
    old client's secret/auth method cleared rather than merged into the new identity."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    oauth2_server = _dcr_redirect_test_server(client_id="stale-client")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "client_id": "fresh-client",
        "redirect_uris": ["https://proxy.litellm.example/callback"],
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    persisted_server = MagicMock()
    persisted_server.credentials = {
        "client_id": "stale-client",
        "client_secret": "stale-secret",
        "token_endpoint_auth_method": "client_secret_basic",
        "redirect_uris": ["https://old-origin.example/callback"],
    }
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_update_mcp_server = AsyncMock(return_value=MagicMock())
    mock_update_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.update_mcp_server",
            new=mock_update_mcp_server,
        ),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    mock_async_client.post.assert_called_once()
    register_payload = mock_async_client.post.call_args.kwargs["json"]
    assert register_payload["redirect_uris"] == ["https://proxy.litellm.example/callback"]

    mock_update_mcp_server.assert_called_once()
    update_data = mock_update_mcp_server.call_args.kwargs["data"]
    assert update_data.credentials["client_id"] == "fresh-client"
    assert update_data.credentials["redirect_uris"] == ["https://proxy.litellm.example/callback"]
    assert update_data.credentials["client_secret"] is None
    assert update_data.credentials["token_endpoint_auth_method"] is None

    assert response.status_code == 200
    assert json.loads(response.body.decode("utf-8"))["client_id"] == "fresh-client"


@pytest.mark.asyncio
async def test_register_client_grandfathers_persisted_client_without_recorded_redirect_uris():
    """Clients persisted before redirect_uris were recorded (and admin-configured clients,
    which never get a recording) have nothing to compare against; treating that as a mismatch
    would re-mint a client_id for every existing install on upgrade and orphan all users'
    refresh tokens for those servers. A missing recording must read as a match: no DCR call,
    no persistence write, existing client returned."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    oauth2_server = _dcr_redirect_test_server(client_id="legacy-client")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()

    persisted_server = MagicMock()
    persisted_server.credentials = {"client_id": "legacy-client"}
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_update_mcp_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.update_mcp_server",
            new=mock_update_mcp_server,
        ),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    mock_async_client.post.assert_not_called()
    mock_update_mcp_server.assert_not_called()
    assert response["client_secret"] == "dummy"
    assert oauth2_server.client_id == "legacy-client"


@pytest.mark.asyncio
async def test_register_client_keeps_persisted_client_when_recorded_redirect_uri_matches_origin():
    """When the recorded redirect_uris still cover the current callback the persisted client
    is valid; re-registering would orphan refresh tokens for no reason."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    oauth2_server = _dcr_redirect_test_server(client_id="kept-client")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()

    persisted_server = MagicMock()
    persisted_server.credentials = {
        "client_id": "kept-client",
        "redirect_uris": ["https://proxy.litellm.example/callback"],
    }
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_update_mcp_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.update_mcp_server",
            new=mock_update_mcp_server,
        ),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=True,
        )

    mock_async_client.post.assert_not_called()
    mock_update_mcp_server.assert_not_called()
    assert response["client_secret"] == "dummy"


@pytest.mark.asyncio
async def test_register_client_non_admin_reuses_persisted_client_despite_redirect_mismatch():
    """Non-persisting callers (the public register routes and non-admin users) must keep
    today's reuse behavior even when the recorded redirect_uris mismatch: re-registering
    without persistence would mint an orphan upstream client on every connect while the
    stored client keeps being used at authorize time. Only the admin path re-registers."""
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client_with_server,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    oauth2_server = _dcr_redirect_test_server(client_id=None)

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://proxy.litellm.example/"
    mock_request.headers = {}

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()

    persisted_server = MagicMock()
    persisted_server.credentials = {
        "client_id": "persisted-client",
        "redirect_uris": ["https://old-origin.example/callback"],
    }
    mock_get_mcp_server = AsyncMock(return_value=persisted_server)
    mock_update_server = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_mcp_server",
            new=mock_get_mcp_server,
        ),
        patch.object(global_mcp_server_manager, "update_server", new=mock_update_server),
    ):
        response = await register_client_with_server(
            request=mock_request,
            mcp_server=oauth2_server,
            client_name="Litellm Proxy",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            persist_credentials=False,
        )

    mock_async_client.post.assert_not_called()
    assert oauth2_server.client_id == "persisted-client"
    assert response["client_secret"] == "dummy"


@pytest.mark.asyncio
async def test_register_client_reuses_existing_client_id_without_re_dcr():
    """A server that already has a client_id (admin-configured or previously DCR'd) must be
    reused, not re-registered, even without a client_secret. A client_id is one-per-application
    in OAuth and shared across users; re-minting per authorize would orphan other users' refresh
    tokens by overwriting the server's client_id."""
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
        client_id="existing-shared-client",
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
        "token_endpoint_auth_method": "none",
    }

    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock()

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
            response = await register_client(request=mock_request, mcp_server_name=oauth2_server.server_name)
    finally:
        global_mcp_server_manager.registry.clear()

    mock_async_client.post.assert_not_called()
    body = response if isinstance(response, dict) else json.loads(response.body.decode("utf-8"))
    assert body["client_secret"] == "dummy"


@pytest.mark.asyncio
async def test_public_register_route_does_not_persist_client_credentials():
    """The unauthenticated root /register route must not persist the DCR result onto the
    server row; only the authenticated management path passes persist_credentials=True. An
    external caller could otherwise bind a caller-controlled client (and leak its secret) to
    a server that has no client yet."""
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
        "client_name": "attacker",
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "client_id": "attacker-client",
        "client_secret": "attacker-secret",
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    mock_update = AsyncMock(return_value=MagicMock())

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
            patch("litellm.proxy.utils.get_prisma_client_or_throw", return_value=MagicMock()),
            patch.object(global_mcp_server_manager, "update_server", new=AsyncMock()),
            patch("litellm.proxy._experimental.mcp_server.db.update_mcp_server", new=mock_update),
        ):
            await register_client(request=mock_request, mcp_server_name=oauth2_server.server_name)
    finally:
        global_mcp_server_manager.registry.clear()

    mock_update.assert_not_called()


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
    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
    assert call_args[1]["data"]["redirect_uri"] == "https://litellm-proxy.example.com/callback"


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
    assert response["authorization_servers"][0].startswith("https://litellm.example.com/")
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
    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
        "redirect_uri=https%3A%2F%2Fproxy.example.com%2Fgithub%2Fmcp%2Fcallback" in location
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
    assert call_args[1]["data"]["redirect_uri"] == "https://proxy.example.com/github/mcp/callback"


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
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com",
            "443",
            "https://proxy.example.com",
        ),
        (
            "http://localhost:4000/",
            "http",
            "proxy.example.com",
            "80",
            "http://proxy.example.com",
        ),
        (
            "http://internal.local/",
            "https",
            None,
            "443",
            "https://internal.local",
        ),
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com",
            "8443",
            "https://proxy.example.com:8443",
        ),
        (
            "http://localhost:4000/",
            "https",
            "proxy.example.com:443",
            None,
            "https://proxy.example.com",
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
def test_get_request_base_url_xff_trust_gate(general_settings, direct_ip, expect_xff_honoured):
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

    matching = [rec for rec in caplog.records if "mcp_trusted_proxy_ranges" in rec.getMessage()]
    assert len(matching) == 1, (
        f"expected exactly one warning, got {len(matching)}: {[r.getMessage() for r in matching]}"
    )


def test_get_request_base_url_honors_proxy_base_url_env(monkeypatch):
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            get_request_base_url,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm-internal:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = "10.0.0.7"
    headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "litellm-internal:4000",
        "X-Forwarded-Port": "9999",
    }
    mock_request.headers.get = lambda name, default=None: headers.get(name, default)

    monkeypatch.setenv("PROXY_BASE_URL", "https://litellm.example.com")
    assert get_request_base_url(mock_request) == "https://litellm.example.com"

    monkeypatch.setenv("PROXY_BASE_URL", "https://litellm.example.com/")
    assert get_request_base_url(mock_request) == "https://litellm.example.com"


def test_validate_trusted_redirect_uri_logs_diagnostic_on_rejection(caplog, monkeypatch):
    try:
        from fastapi import HTTPException, Request

        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            validate_trusted_redirect_uri,
        )
    except ImportError:
        pytest.skip("MCP oauth_utils not available")

    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("MCP_TRUSTED_REDIRECT_ORIGINS", raising=False)

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm-internal:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = "203.0.113.5"
    headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "litellm.example.com",
        "X-Forwarded-Port": "443",
        "Host": "litellm-internal:4000",
    }
    mock_request.headers.get = lambda name, default=None: headers.get(name, default)

    import logging

    with (
        caplog.at_level(logging.WARNING, logger="LiteLLM"),
        patch("litellm.proxy.proxy_server.general_settings", {}, create=True),
    ):
        with pytest.raises(HTTPException) as exc_info:
            validate_trusted_redirect_uri(
                mock_request,
                "https://litellm.example.com/ui/mcp/oauth/callback",
            )
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail.get("error") == "invalid_request"
        assert "error_description" in detail
        assert "redirect_uri origin" in detail["error_description"]
        assert "proxy origin" in detail["error_description"]
        assert "hint" in detail

    matching = [r for r in caplog.records if "rejecting redirect_uri" in r.getMessage()]
    assert len(matching) == 1, (
        f"expected exactly one diagnostic warning, got {[r.getMessage() for r in caplog.records]}"
    )
    msg = matching[0].getMessage()
    assert "https://litellm.example.com/ui/mcp/oauth/callback" in msg
    assert "litellm-internal:4000" in msg
    assert "X-Forwarded-Host" in msg


@pytest.mark.parametrize(
    "bad_value",
    [
        "litellm.example.com",
        "litellm.example.com/",
        "://litellm.example.com",
        "ftp://litellm.example.com",
        "https://",
        "not a url at all",
    ],
)
def test_get_request_base_url_rejects_malformed_proxy_base_url(bad_value, monkeypatch, caplog):
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server import oauth_utils
        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            get_request_base_url,
        )
    except ImportError:
        pytest.skip("MCP oauth_utils not available")

    oauth_utils._warned_invalid_proxy_base_url = None

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm-internal:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.headers.get = lambda name, default=None: default

    monkeypatch.setenv("PROXY_BASE_URL", bad_value)

    import logging

    with (
        caplog.at_level(logging.WARNING, logger="LiteLLM"),
        patch("litellm.proxy.proxy_server.general_settings", {}, create=True),
    ):
        result = get_request_base_url(mock_request)

    assert result == "http://litellm-internal:4000", (
        f"malformed PROXY_BASE_URL={bad_value!r} should be ignored, got {result!r}"
    )
    matching = [r for r in caplog.records if "PROXY_BASE_URL" in r.getMessage() and "ignored" in r.getMessage()]
    assert len(matching) == 1, (
        f"expected one diagnostic for malformed PROXY_BASE_URL, got {[r.getMessage() for r in caplog.records]}"
    )
    assert repr(bad_value) in matching[0].getMessage() or bad_value in matching[0].getMessage()


def test_get_request_base_url_malformed_proxy_base_url_warning_is_one_shot(monkeypatch, caplog):
    try:
        from fastapi import Request

        from litellm.proxy._experimental.mcp_server import oauth_utils
        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            get_request_base_url,
        )
    except ImportError:
        pytest.skip("MCP oauth_utils not available")

    oauth_utils._warned_invalid_proxy_base_url = None

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://litellm-internal:4000/"
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.headers.get = lambda name, default=None: default

    monkeypatch.setenv("PROXY_BASE_URL", "litellm.example.com")

    import logging

    with (
        caplog.at_level(logging.WARNING, logger="LiteLLM"),
        patch("litellm.proxy.proxy_server.general_settings", {}, create=True),
    ):
        for _ in range(5):
            get_request_base_url(mock_request)

    matching = [r for r in caplog.records if "PROXY_BASE_URL" in r.getMessage() and "ignored" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one warning across 5 calls, got {len(matching)}"


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
        response = await _build_oauth_protected_resource_response(
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
    server1 = _create_oauth2_server(server_id="server1", name="server1", server_name="server1", alias="server1")
    server2 = _create_oauth2_server(server_id="server2", name="server2", server_name="server2", alias="server2")
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
            resource_response = await _build_oauth_protected_resource_response(
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
        mock_decode.return_value = mock_state_data

        # Call callback endpoint with code and state
        response = await callback(
            request=_mock_callback_request(),
            code="test_authorization_code_12345",
            state="encrypted_state_value",
        )

        # Should redirect to the client callback URL with code and original state
        assert response.status_code == 302
        assert "http://localhost:3000/ui/mcp/oauth/callback" in response.headers["location"]
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
        mock_decode.return_value = {
            "base_url": "http://localhost:3000/ui/mcp/oauth/callback",
            "original_state": "test-uuid-state-123",
            "code_challenge": "test_challenge",
            "code_challenge_method": "S256",
            "client_redirect_uri": ("http://localhost:3000/ui/mcp/oauth/callback?session=abc"),
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
    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
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
    assert "https://proxy.example.com/ui/mcp/oauth/callback" in response.headers["location"]
    assert "code=auth-code-123" in response.headers["location"]
    assert "state=state-123" in response.headers["location"]


@pytest.mark.asyncio
async def test_authorize_forwards_short_state_and_round_trips_via_cookie(monkeypatch):
    """LIT-4197: the ``state`` sent to the upstream authorization server must be
    a short opaque handle, not the long encrypted OAuth session (some IdPs
    reject an over-long state). The session must instead ride in a per-flow
    HttpOnly cookie so ``/callback`` still recovers the client's original state
    and redirects back to the client's redirect_uri."""
    from http.cookies import SimpleCookie
    from urllib.parse import parse_qs, urlparse

    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _oauth_state_cookie_name,
        authorize_with_server,
        callback,
        decode_state_hash,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    # Real encryption so the cookie value is a genuine encrypted session.
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-for-LIT-4197")

    client_state = "ee230e3dfd4f19c7441941684f39c8a4e0e2c3c61a088e33403df5662b4047b8"
    client_redirect_uri = "http://127.0.0.1:6274/oauth/callback/debug"

    server = MCPServer(
        server_id="leanix_server",
        name="leanix",
        server_name="leanix",
        alias="leanix",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="upstream-client-id",
        authorization_url="https://idp.example.com/oauth/authorize",
        token_url="https://idp.example.com/oauth/token",
    )

    authorize_request = MagicMock(spec=Request)
    authorize_request.base_url = "https://proxy.example.com/"
    authorize_request.headers = {}

    authorize_response = await authorize_with_server(
        request=authorize_request,
        mcp_server=server,
        client_id="upstream-client-id",
        redirect_uri=client_redirect_uri,
        state=client_state,
        code_challenge="challenge",
        code_challenge_method="S256",
    )

    location = authorize_response.headers["location"]
    upstream_state = parse_qs(urlparse(location).query)["state"][0]

    # The upstream must receive a short handle, not the encrypted session blob.
    assert len(upstream_state) <= 64
    assert upstream_state != client_state

    # The encrypted session rides in a per-flow HttpOnly cookie bound to it.
    jar = SimpleCookie()
    jar.load(authorize_response.headers["set-cookie"])
    cookie_name = _oauth_state_cookie_name(upstream_state)
    assert cookie_name in jar
    morsel = jar[cookie_name]
    assert morsel["httponly"]
    assert morsel["samesite"].lower() == "lax"
    assert len(morsel.value) > len(upstream_state)
    session = decode_state_hash(morsel.value)
    assert session["original_state"] == client_state
    assert session["client_redirect_uri"] == client_redirect_uri

    # /callback recovers the original state from the cookie (not the handle) and
    # redirects back to the client with the client's own state.
    callback_request = MagicMock(spec=Request)
    callback_request.base_url = "https://proxy.example.com/"
    callback_request.headers = {}
    callback_request.cookies = {cookie_name: morsel.value}

    callback_response = await callback(
        request=callback_request,
        code="upstream-auth-code",
        state=upstream_state,
    )

    assert callback_response.status_code == 302
    cb_query = parse_qs(urlparse(callback_response.headers["location"]).query)
    assert callback_response.headers["location"].startswith(client_redirect_uri)
    assert cb_query["code"] == ["upstream-auth-code"]
    assert cb_query["state"] == [client_state]

    # The one-time cookie is expired on the callback response so it cannot be replayed.
    cleared = SimpleCookie()
    cleared.load(callback_response.headers["set-cookie"])
    assert cookie_name in cleared
    assert cleared[cookie_name].value == ""
    assert cleared[cookie_name]["max-age"] == "0"


@pytest.mark.asyncio
async def test_callback_error_path_reads_cookie_and_clears_it(monkeypatch):
    """LIT-4197: an IdP error routed through /callback must recover the client's
    original state from the cookie (not the short handle), propagate the error to
    the client's redirect_uri, and expire the one-time cookie."""
    from http.cookies import SimpleCookie
    from urllib.parse import parse_qs, urlparse

    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _oauth_state_cookie_name,
        callback,
        encode_state_with_base_url,
    )

    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-test-salt-for-LIT-4197")

    client_state = "client-original-state-abc"
    client_redirect_uri = "http://127.0.0.1:6274/oauth/callback/debug"
    handle = "shortRelayHandle123"
    encoded_state = encode_state_with_base_url(
        base_url=client_redirect_uri,
        original_state=client_state,
        client_redirect_uri=client_redirect_uri,
    )
    cookie_name = _oauth_state_cookie_name(handle)

    request = MagicMock(spec=Request)
    request.base_url = "https://proxy.example.com/"
    request.headers = {}
    request.cookies = {cookie_name: encoded_state}

    response = await callback(
        request=request,
        error="access_denied",
        error_description="User declined access",
        state=handle,
    )

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith(client_redirect_uri)
    query = parse_qs(urlparse(location).query)
    assert query["error"] == ["access_denied"]
    # The client's own state is echoed back, recovered from the cookie.
    assert query["state"] == [client_state]

    cleared = SimpleCookie()
    cleared.load(response.headers["set-cookie"])
    assert cookie_name in cleared
    assert cleared[cookie_name].value == ""
    assert cleared[cookie_name]["max-age"] == "0"


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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
            "scope=api+read_user+ai_workflows" in redirect_url or "scope=api%20read_user%20ai_workflows" in redirect_url
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper") as mock_encrypt:
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
            "scope=custom_scope1+custom_scope2" in redirect_url or "scope=custom_scope1%20custom_scope2" in redirect_url
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
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

    with patch("litellm.proxy._experimental.mcp_server.discoverable_endpoints.decode_state_hash") as mock_decode:
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


async def _exchange_with_upstream_token_response(upstream_body):
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
    fake_http_response.json.return_value = upstream_body
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
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_token_exchange_omits_expires_in_when_upstream_omits_it():
    """A provider that issues a non-expiring token (e.g. Slack without token
    rotation) returns no ``expires_in``. The exchange must mirror that and omit
    ``expires_in`` rather than fabricate a 1-hour TTL, so the stored credential
    is treated as non-expiring instead of dying after an hour."""
    body = await _exchange_with_upstream_token_response({"access_token": "tok", "token_type": "Bearer"})
    assert "expires_in" not in body


@pytest.mark.asyncio
async def test_token_exchange_passes_through_upstream_expires_in():
    """When the provider does send ``expires_in`` (e.g. Slack with token
    rotation), the exchange forwards the real value unchanged."""
    body = await _exchange_with_upstream_token_response(
        {"access_token": "tok", "token_type": "Bearer", "expires_in": 43200}
    )
    assert body["expires_in"] == 43200


_BRIDGE_CLIENT_REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def _bridge_server(**overrides):
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    fields = {
        "server_id": "bridge_srv",
        "name": "bridge_srv",
        "server_name": "bridge_srv",
        "alias": "bridge_srv",
        "transport": MCPTransport.http,
        "auth_type": MCPAuth.true_passthrough,
        "dcr_bridge": True,
        "authorization_url": "https://provider.com/oauth/authorize",
        "token_url": "https://provider.com/oauth/token",
        "registration_url": "https://provider.com/oauth/register",
        **overrides,
    }
    return MCPServer(**fields)


def _bridge_mock_request():
    from fastapi import Request

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}
    return mock_request


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type_value", ["true_passthrough", "oauth_delegate"])
async def test_authorize_bridge_relay_passes_client_params_verbatim(auth_type_value):
    """The bridge relay arm (registration relayed upstream, no admin-configured client) passes the
    client's client_id, redirect_uri, state, and PKCE through verbatim: the code returns straight
    to the client's own redirect URI, so the gateway sets no state cookie, injects no /callback,
    and applies no gateway-side redirect trust (the upstream enforces its registered binding)."""
    from urllib.parse import parse_qs, urlparse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize_with_server,
    )
    from litellm.types.mcp import MCPAuth

    response = await authorize_with_server(
        request=_bridge_mock_request(),
        mcp_server=_bridge_server(auth_type=MCPAuth(auth_type_value)),
        client_id="dcr-client-123",
        redirect_uri=_BRIDGE_CLIENT_REDIRECT,
        state="client-state",
        code_challenge="chal",
        code_challenge_method="S256",
    )

    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("https://provider.com/oauth/authorize")
    query = parse_qs(urlparse(location).query)
    assert query["client_id"] == ["dcr-client-123"]
    assert query["redirect_uri"] == [_BRIDGE_CLIENT_REDIRECT]
    assert query["state"] == ["client-state"]
    assert query["code_challenge"] == ["chal"]
    assert query["code_challenge_method"] == ["S256"]
    assert "litellm.example.com" not in location
    assert "set-cookie" not in {key.lower() for key in response.headers.keys()}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code_challenge,code_challenge_method",
    [(None, None), ("chal", None), ("chal", "plain"), (None, "S256")],
)
async def test_authorize_bridge_requires_s256_pkce(code_challenge, code_challenge_method):
    """Bridge servers serve unauthenticated public clients, so the PKCE downgrade paths (missing
    challenge, or a method that is not S256; RFC 7636 defaults a missing method to plain) are
    rejected at the gateway on both bridge arms."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize_with_server,
    )

    with pytest.raises(HTTPException) as exc:
        await authorize_with_server(
            request=_bridge_mock_request(),
            mcp_server=_bridge_server(),
            client_id="dcr-client-123",
            redirect_uri=_BRIDGE_CLIENT_REDIRECT,
            state="s",
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

    assert exc.value.status_code == 400
    assert "S256" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_authorize_bridge_short_circuit_keeps_callback_and_redirect_trust():
    """The bridge short-circuit arm (admin-configured OAuth client, upstream only knows the
    gateway callback) keeps the /callback state relay and the gateway redirect trust: a public
    client redirect target is rejected unless ops allowlist it, and a trusted target still routes
    through the gateway callback with the state cookie."""
    from urllib.parse import parse_qs, urlparse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize_with_server,
    )

    short_circuit_server = _bridge_server(client_id="admin-client", registration_url=None)

    with pytest.raises(HTTPException) as exc:
        await authorize_with_server(
            request=_bridge_mock_request(),
            mcp_server=short_circuit_server,
            client_id="ignored",
            redirect_uri=_BRIDGE_CLIENT_REDIRECT,
            state="s",
            code_challenge="chal",
            code_challenge_method="S256",
        )
    assert exc.value.status_code in (400, 403)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper",
        return_value="mocked_encrypted_state",
    ):
        response = await authorize_with_server(
            request=_bridge_mock_request(),
            mcp_server=short_circuit_server,
            client_id="ignored",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="s",
            code_challenge="chal",
            code_challenge_method="S256",
        )

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == ["https://litellm.example.com/callback"]
    assert query["client_id"] == ["admin-client"]


@pytest.mark.asyncio
async def test_authorize_non_bridge_client_forwarded_keeps_pre_bridge_contract():
    """A client-forwarded server without dcr_bridge keeps the pre-bridge behavior: no PKCE
    requirement and the gateway /callback relay (this is the browser-only Authorize path)."""
    from urllib.parse import parse_qs, urlparse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        authorize_with_server,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.encrypt_value_helper",
        return_value="mocked_encrypted_state",
    ):
        response = await authorize_with_server(
            request=_bridge_mock_request(),
            mcp_server=_bridge_server(dcr_bridge=None),
            client_id="cid",
            redirect_uri="http://127.0.0.1:60108/callback",
            state="s",
        )

    assert response.status_code == 307
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == ["https://litellm.example.com/callback"]


async def _bridge_token_post_data(server, redirect_uri):
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        exchange_token_with_server,
    )

    fake_http_response = MagicMock()
    fake_http_response.json.return_value = {"access_token": "tok", "token_type": "Bearer"}
    fake_http_response.raise_for_status = MagicMock()
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock(return_value=fake_http_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
        return_value=fake_http_client,
    ):
        await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri=redirect_uri,
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier="verifier",
        )
    return fake_http_client.post.call_args.kwargs["data"]


@pytest.mark.asyncio
async def test_token_bridge_relay_posts_client_redirect_uri():
    """The bridge relay arm's token exchange sends the client's own redirect_uri upstream (it must
    match the authorize leg) with the caller's public client_id and PKCE verifier."""
    data = await _bridge_token_post_data(_bridge_server(), redirect_uri=_BRIDGE_CLIENT_REDIRECT)

    assert data["redirect_uri"] == _BRIDGE_CLIENT_REDIRECT
    assert data["client_id"] == "dcr-client-123"
    assert data["code_verifier"] == "verifier"
    assert "client_secret" not in data


@pytest.mark.asyncio
async def test_token_bridge_relay_requires_redirect_uri():
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        exchange_token_with_server,
    )

    with pytest.raises(HTTPException) as exc:
        await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=_bridge_server(),
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri=None,
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier="verifier",
        )

    assert exc.value.status_code == 400
    assert "redirect_uri" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_token_non_bridge_keeps_gateway_callback():
    """Without dcr_bridge the token exchange keeps posting the gateway callback as redirect_uri,
    pinning the pre-bridge contract for the browser-only Authorize path."""
    data = await _bridge_token_post_data(_bridge_server(dcr_bridge=None), redirect_uri=_BRIDGE_CLIENT_REDIRECT)

    assert data["redirect_uri"] == "https://litellm.example.com/callback"


def _named_as_metadata_response(server):
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _build_oauth_authorization_server_response,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    global_mcp_server_manager.registry[server.server_id] = server
    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.get_mcp_client_ip",
            return_value=None,
        ):
            return _build_oauth_authorization_server_response(
                request=_bridge_mock_request(),
                mcp_server_name=server.server_name,
            )
    finally:
        global_mcp_server_manager.registry.clear()


def test_oauth_authorization_server_metadata_served_for_bridge_server():
    """Bridge servers get the gateway's AS metadata (the register, authorize, and token relays),
    which is what makes the DCR front door discoverable to standard MCP clients."""
    result = _named_as_metadata_response(_bridge_server())

    assert result["authorization_endpoint"] == "https://litellm.example.com/bridge_srv/authorize"
    assert result["token_endpoint"] == "https://litellm.example.com/bridge_srv/token"
    assert result["registration_endpoint"] == "https://litellm.example.com/bridge_srv/register"


def test_oauth_authorization_server_404_for_non_bridge_client_forwarded_server():
    """Without dcr_bridge a client-forwarded server keeps 404ing AS-metadata discovery: verbatim
    upstream discovery is the contract and the gateway must not advertise itself as its AS."""
    with pytest.raises(HTTPException) as exc:
        _named_as_metadata_response(_bridge_server(dcr_bridge=None))

    assert exc.value.status_code == 404


async def _bridge_register_response(server, request_payload, persist_credentials=False):
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        register_client_with_server,
    )

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "client_id": "upstream-issued-client",
        "redirect_uris": request_payload.get("redirect_uris", []),
        "token_endpoint_auth_method": "none",
    }
    mock_response.raise_for_status = MagicMock()
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._persist_dcr_client_registration",
            new_callable=AsyncMock,
        ) as mock_persist,
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._reuse_persisted_dcr_client_if_available",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        response = await register_client_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            client_name=request_payload.get("client_name", ""),
            grant_types=request_payload.get("grant_types"),
            response_types=request_payload.get("response_types"),
            token_endpoint_auth_method=request_payload.get("token_endpoint_auth_method"),
            persist_credentials=persist_credentials,
            client_redirect_uris=request_payload.get("redirect_uris"),
        )
    return response, mock_async_client, mock_persist


@pytest.mark.asyncio
async def test_register_bridge_relay_forwards_client_redirect_uris():
    """The bridge relay arm registers the client's own redirect_uris upstream with public-client
    defaults and relays the upstream response verbatim, so the upstream AS enforces the redirect
    binding for that client and the auth code never transits the gateway."""
    import json

    response, mock_async_client, _ = await _bridge_register_response(
        _bridge_server(),
        {"client_name": "Claude", "redirect_uris": [_BRIDGE_CLIENT_REDIRECT]},
    )

    posted = mock_async_client.post.call_args.kwargs["json"]
    assert posted["redirect_uris"] == [_BRIDGE_CLIENT_REDIRECT]
    assert posted["grant_types"] == ["authorization_code", "refresh_token"]
    assert posted["response_types"] == ["code"]
    assert posted["token_endpoint_auth_method"] == "none"

    payload = json.loads(response.body.decode("utf-8"))
    assert payload["client_id"] == "upstream-issued-client"


@pytest.mark.asyncio
async def test_register_bridge_relay_requires_redirect_uris():
    with pytest.raises(HTTPException) as exc:
        await _bridge_register_response(_bridge_server(), {"client_name": "Claude"})

    assert exc.value.status_code == 400
    assert "redirect_uris" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_register_bridge_relay_surfaces_upstream_error_not_500():
    """A bridge relay registration the upstream rejects must surface the upstream status and its
    RFC 7591 error body to the client, not a bare 500 that hides the real reason."""
    import httpx

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        register_client_with_server,
    )

    error_response = MagicMock()
    error_response.status_code = 400
    error_response.text = '{"error":"invalid_redirect_uri","error_description":"redirect_uri not allowed"}'
    error_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("bad", request=MagicMock(), response=error_response)
    )
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=error_response)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._reuse_persisted_dcr_client_if_available",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await register_client_with_server(
                request=_bridge_mock_request(),
                mcp_server=_bridge_server(),
                client_name="Claude",
                grant_types=None,
                response_types=None,
                token_endpoint_auth_method=None,
                client_redirect_uris=[_BRIDGE_CLIENT_REDIRECT],
            )

    assert exc.value.status_code == 400
    assert "invalid_redirect_uri" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_register_non_bridge_upstream_error_still_raises_500():
    """Non-bridge DCR keeps its pre-change behavior: raise_for_status propagates so the flag-off
    contract is byte-identical; only the bridge relay arm relays the upstream status."""
    import httpx

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        register_client_with_server,
    )

    error_response = MagicMock()
    error_response.status_code = 400
    error_response.text = '{"error":"invalid_client_metadata"}'
    error_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("bad", request=MagicMock(), response=error_response)
    )
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=error_response)

    oauth2_server = _bridge_server(auth_type=MCPAuth.oauth2, dcr_bridge=None)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=mock_async_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._reuse_persisted_dcr_client_if_available",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await register_client_with_server(
                request=_bridge_mock_request(),
                mcp_server=oauth2_server,
                client_name="Claude",
                grant_types=None,
                response_types=None,
                token_endpoint_auth_method=None,
            )


@pytest.mark.asyncio
async def test_register_bridge_relay_never_persists():
    """Relayed registrations belong to individual clients; persisting one as the server's own DCR
    client would hand every future caller the first client's identity."""
    _, _, mock_persist = await _bridge_register_response(
        _bridge_server(),
        {"client_name": "Claude", "redirect_uris": [_BRIDGE_CLIENT_REDIRECT]},
        persist_credentials=True,
    )

    mock_persist.assert_not_called()


_BRIDGE_MASTER_KEY = "sk-bridge-producer-master-key-0123456789abcdef"


async def _exchange_for_bridge_server(server, upstream_body, key_hash, fake_client_out=None):
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _ResolvedKey,
        exchange_token_with_server,
    )

    fake_http_response = MagicMock()
    fake_http_response.json.return_value = upstream_body
    fake_http_response.raise_for_status = MagicMock()
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock(return_value=fake_http_response)
    # The mint consumes _resolve_active_litellm_key's tagged result: an active key resolves to a
    # _ResolvedKey carrying its hash; a request with no usable credential resolves to "no_active_key".
    resolution = _ResolvedKey(key_hash=key_hash, key=MagicMock()) if key_hash is not None else "no_active_key"
    key_resolver = AsyncMock(return_value=resolution)
    if fake_client_out is not None:
        fake_client_out["client"] = fake_http_client

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=fake_http_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._resolve_active_litellm_key",
            new=key_resolver,
        ),
        patch("litellm.proxy.proxy_server.master_key", _BRIDGE_MASTER_KEY),
    ):
        response = await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier="verifier",
        )
    if server.is_oauth_delegate and server.is_dcr_bridge:
        key_resolver.assert_awaited_once()
    else:
        key_resolver.assert_not_awaited()
    return response


@pytest.mark.asyncio
async def test_oauth_delegate_bridge_token_exchange_mints_envelope_not_raw_token():
    """A dcr_bridge oauth_delegate token exchange returns a gateway-bound envelope, not the raw
    upstream token: the response access_token opens (under the same master-key-derived keys and the
    server_id) to the caller's identity and the upstream Authorization, and the raw upstream token
    never appears in the bearer the client receives."""
    from datetime import datetime, timezone

    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
        BridgeEnvelopeAdmitted,
        envelope_keys_from_master_key,
        resolve_bridge_envelope,
    )
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UPSTREAM-SECRET-TOKEN", "token_type": "Bearer", "expires_in": 3600}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")

    body = json.loads(response.body)
    token = body["access_token"]
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] > 0
    assert token.startswith("llm_env_")
    assert "UPSTREAM-SECRET-TOKEN" not in token
    assert "refresh_token" not in body

    keys = envelope_keys_from_master_key(_BRIDGE_MASTER_KEY)
    opened = resolve_bridge_envelope(token, keys, datetime.now(timezone.utc), server.server_id)
    assert isinstance(opened, BridgeEnvelopeAdmitted)
    assert opened.identity.key_hash == "hashed-litellm-key-77"
    assert opened.upstream_authorization.get_secret_value() == "Bearer UPSTREAM-SECRET-TOKEN"


@pytest.mark.asyncio
async def test_oauth_delegate_bridge_token_exchange_fails_closed_without_litellm_identity():
    """Without a resolvable litellm identity on the token request, the exchange must not mint an
    identity-less envelope. It returns an RFC 6749 §5.2-shaped invalid_request (error at the top
    level, not wrapped in detail) BEFORE exchanging the upstream code, so the single-use code is not
    burned and the client can retry."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UPSTREAM-SECRET-TOKEN", "token_type": "Bearer", "expires_in": 3600}
    captured: dict = {}
    response = await _exchange_for_bridge_server(server, upstream, key_hash=None, fake_client_out=captured)

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_request"
    # identity resolution failed first, so the upstream single-use code was never exchanged (not burned)
    captured["client"].post.assert_not_called()


@pytest.mark.asyncio
async def test_bridge_envelope_too_large_upstream_token_is_502():
    """An upstream token too large to seal into the envelope is an upstream-payload condition, so the
    mint surfaces a 502 (as an RFC 6749 §5.2 error body, not a raised HTTPException) rather than a 500:
    build_bridge_token_response returns EnvelopeTooLarge as a value, _finish_bridge_mint returns the
    "too_large" failure, and _bridge_mint_error_response maps it to a truthful status."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "x" * 40000, "token_type": "Bearer", "expires_in": 3600}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    assert response.status_code == 502
    assert json.loads(response.body)["error"] == "server_error"


@pytest.mark.asyncio
async def test_bridge_envelope_does_not_seal_upstream_refresh_token():
    """The upstream refresh_token is never sealed into the client-held envelope: the edge never
    consumes it and a long-lived upstream credential should not live in the client bearer. The opened
    envelope's grant carries no refresh token even when the upstream returned one, and neither does
    the response body."""
    from datetime import datetime, timezone

    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
        envelope_keys_from_master_key,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
        OpenedEnvelope,
        open_envelope,
    )
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {
        "access_token": "UP",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "UPSTREAM-REFRESH",
    }
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")

    body = json.loads(response.body)
    assert "refresh_token" not in body
    assert "UPSTREAM-REFRESH" not in body["access_token"]
    keys = envelope_keys_from_master_key(_BRIDGE_MASTER_KEY)
    opened = open_envelope(body["access_token"], keys, datetime.now(timezone.utc))
    assert isinstance(opened, OpenedEnvelope)
    assert opened.grant.refresh_token is None


@pytest.mark.asyncio
async def test_bridge_refresh_grant_is_rejected_before_upstream():
    """A bridge oauth_delegate server issues only envelopes and seals no upstream refresh_token, so the
    client never holds one to present. _prepare_bridge_mint rejects the refresh_token grant up front
    with unsupported_grant_type, BEFORE any upstream exchange, so a stray refresh request can never
    rotate or consume the client's upstream refresh credential; renewal is re-running
    authorization_code. This is checked before identity resolution, so it holds even with a valid key."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import exchange_token_with_server
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock()
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=fake_http_client,
        ),
        patch("litellm.proxy.proxy_server.master_key", _BRIDGE_MASTER_KEY),
    ):
        response = await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            grant_type="refresh_token",
            code=None,
            redirect_uri=None,
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier=None,
            refresh_token="client-refresh-token",
        )

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "unsupported_grant_type"
    fake_http_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_bridge_mint_fails_closed_before_upstream_when_master_key_unset():
    """master_key is validated BEFORE the upstream exchange (in _prepare_bridge_mint), so a
    misconfigured gateway returns a 500 server_error without consuming the single-use code, avoiding
    the burn-then-fail the pre-exchange phase exists to prevent. The failure is returned as an RFC 6749
    error body, not raised."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import exchange_token_with_server
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock()
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=fake_http_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._resolve_active_litellm_key",
            new=AsyncMock(return_value="no_active_key"),
        ),
        patch("litellm.proxy.proxy_server.master_key", None),
    ):
        response = await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier="verifier",
        )

    assert response.status_code == 500
    assert json.loads(response.body)["error"] == "server_error"
    fake_http_client.post.assert_not_called()


async def _prepare_only_bridge_exchange(resolver_result):
    """Drive exchange_token_with_server for a bridge oauth_delegate authorization_code request with the
    identity resolver stubbed to a given tagged result, returning (response, post_mock) so a test can
    assert the mapped status and that the single-use code was never exchanged."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import exchange_token_with_server
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock()
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=fake_http_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._resolve_active_litellm_key",
            new=AsyncMock(return_value=resolver_result),
        ),
        patch("litellm.proxy.proxy_server.master_key", _BRIDGE_MASTER_KEY),
    ):
        response = await exchange_token_with_server(
            request=_bridge_mock_request(),
            mcp_server=server,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            client_id="dcr-client-123",
            client_secret=None,
            code_verifier="verifier",
        )
    return response, fake_http_client.post


@pytest.mark.asyncio
async def test_bridge_mint_db_outage_is_503_before_upstream():
    """A DB outage while resolving identity is a retryable gateway failure, so the mint returns 503
    temporarily_unavailable WITHOUT consuming the single-use code, matching how admission statuses the
    same outage on the egress side. Collapsing every resolution failure to None used to blame the
    client with 400 invalid_request for an infrastructure problem."""
    response, post = await _prepare_only_bridge_exchange("unavailable")
    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "temporarily_unavailable"
    post.assert_not_called()


@pytest.mark.asyncio
async def test_bridge_mint_unresolvable_identity_is_500_before_upstream():
    """An unresolvable identity (no DB connection, or an unexpected resolution error) is a gateway
    fault, so the mint returns 500 server_error before the exchange, a status distinct from both the
    caller's 400 and the transient 503, matching admission's 500-vs-503 split for the same conditions."""
    response, post = await _prepare_only_bridge_exchange("unresolvable")
    assert response.status_code == 500
    assert json.loads(response.body)["error"] == "server_error"
    post.assert_not_called()


@pytest.mark.asyncio
async def test_bridge_mint_upstream_expired_lifetime_is_502():
    """An upstream token response reporting an already-elapsed lifetime (a parseable non-positive
    expires_in) is rejected with 502 rather than sealed into an hour-long envelope around a dead
    bearer. Regression for expires_in<=0 silently falling through to the 1h cap."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UP", "token_type": "Bearer", "expires_in": 0}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    assert response.status_code == 502
    assert json.loads(response.body)["error"] == "server_error"


@pytest.mark.asyncio
async def test_bridge_mint_positive_sub_second_lifetime_mints_not_502():
    """A positive fractional expires_in in (0, 1) is a live token, not an elapsed one, so it mints a
    (1s-floored) envelope rather than being truncated to 0 and rejected with 502 after the single-use
    code was already consumed. Regression for classifying a sub-second remaining lifetime as expired."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UP", "token_type": "Bearer", "expires_in": 0.5}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["access_token"].startswith("llm_env_")
    assert body["expires_in"] >= 0


@pytest.mark.asyncio
async def test_bridge_mint_unknown_lifetime_is_capped_not_rejected():
    """An absent or unparseable expires_in leaves the lifetime unknown, which the envelope caps (never
    inventing a longer life than the upstream stated); it is NOT rejected. Only an explicitly-dead
    lifetime fails, so a metadata glitch on an otherwise-valid token still mints a bounded envelope."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UP", "token_type": "Bearer", "expires_in": "not-a-number"}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["access_token"].startswith("llm_env_")
    assert 0 < body["expires_in"] <= 3600


@pytest.mark.asyncio
async def test_bridge_reported_expires_in_does_not_overstate_jwt_exp():
    """The reported expires_in is derived from the envelope JWT's second-truncated exp (rounding the
    elapsed portion up), so the client is never told the bearer lives past the point admission expires
    it. Regression for the sub-second overstatement of the raw (expires_at - now) delta."""
    import time

    import jwt as _jwt

    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UP", "token_type": "Bearer", "expires_in": 300}
    before = int(time.time())
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    body = json.loads(response.body)
    claims = _jwt.decode(body["access_token"].removeprefix("llm_env_"), options={"verify_signature": False})
    # projecting the reported lifetime from a time no later than the mint must not exceed the JWT exp
    assert before + body["expires_in"] <= claims["exp"]


def test_bridge_reported_expires_in_can_be_zero_at_jwt_exp_boundary():
    from datetime import datetime, timezone

    from fastapi.responses import JSONResponse

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _BridgeMintReady,
        _finish_bridge_mint,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (
        envelope_keys_from_master_key,
    )
    from litellm.types.mcp import MCPAuth

    ready = _BridgeMintReady(
        key_hash="hashed-litellm-key-77",
        keys=envelope_keys_from_master_key(_BRIDGE_MASTER_KEY),
    )
    response = _finish_bridge_mint(
        ready=ready,
        mcp_server=_bridge_server(auth_type=MCPAuth.oauth_delegate),
        token_response={"access_token": "UP", "expires_in": 1},
        now=datetime.fromtimestamp(100.25, tz=timezone.utc),
    )

    assert isinstance(response, JSONResponse)
    assert json.loads(response.body)["expires_in"] == 0


def test_classify_upstream_lifetime():
    """expires_in from an IdP may be an int, a float (3600.0), or a numeric string ("3600"); each
    coerces to a positive number of seconds. Absent or unparseable input (bool, non-numeric, NaN/inf,
    oversized) is "unspecified" so the envelope caps it, while a parseable non-positive value is
    "expired": the upstream reporting an already-dead token, which the mint must reject rather than
    silently give the 1h cap."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import _classify_upstream_lifetime

    assert _classify_upstream_lifetime(300) == 300
    assert _classify_upstream_lifetime(300.0) == 300
    assert _classify_upstream_lifetime("300") == 300
    assert _classify_upstream_lifetime("  300  ") == 300
    # explicit, parseable, non-positive -> the upstream says the token is already dead
    assert _classify_upstream_lifetime(0) == "expired"
    assert _classify_upstream_lifetime(-5) == "expired"
    assert _classify_upstream_lifetime(-0.5) == "expired"
    # a positive sub-second lifetime is alive, not elapsed; it clamps up to the envelope's 1s floor
    # rather than truncating to 0 and being misread as expired
    assert _classify_upstream_lifetime(0.5) == 1
    assert _classify_upstream_lifetime(0.001) == 1
    # a positive value >= 1 truncates toward zero (never overstating the stated lifetime)
    assert _classify_upstream_lifetime(1.9) == 1
    # unknown lifetime -> cap (never invent a longer life than the upstream stated)
    assert _classify_upstream_lifetime(None) == "unspecified"
    assert _classify_upstream_lifetime(True) == "unspecified"
    assert _classify_upstream_lifetime("nope") == "unspecified"
    # hostile numerics must not raise (int(float(...)) can OverflowError) -> unspecified
    assert _classify_upstream_lifetime("inf") == "unspecified"
    assert _classify_upstream_lifetime("1e999") == "unspecified"
    assert _classify_upstream_lifetime("-inf") == "unspecified"
    assert _classify_upstream_lifetime("nan") == "unspecified"
    assert _classify_upstream_lifetime(float("inf")) == "unspecified"
    assert _classify_upstream_lifetime(10**400) == "unspecified"


def test_bridge_grant_honors_and_rejects_upstream_lifetime():
    """The grant validator honors a positive lifetime, leaves an unknown one None for the envelope to
    cap, and rejects an explicitly-expired one with "expired_lifetime" so a dead upstream token is
    never sealed into an hour-long envelope."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import _bridge_grant_from_token_response

    def grant(v):
        return _bridge_grant_from_token_response({"access_token": "x", "expires_in": v})

    assert grant(300).expires_in == 300
    assert grant(120.0).expires_in == 120
    # unknown lifetime backs a grant whose expires_in the envelope caps; it is not a rejection
    assert grant("nope").expires_in is None
    assert _bridge_grant_from_token_response({"access_token": "x"}).expires_in is None
    # an explicitly already-dead lifetime is rejected, not silently capped at 1h
    assert grant(0) == "expired_lifetime"
    assert grant(-5) == "expired_lifetime"


@pytest.mark.asyncio
async def test_bridge_token_exchange_honors_short_float_expires_in_ttl():
    """A short float expires_in from the upstream caps the envelope TTL, so the client-held envelope
    does not outlive the upstream token. Before coercion a float was dropped and the envelope
    defaulted to the 1h cap (3600), which would forward a stale bearer after the upstream token
    expired."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"access_token": "UP", "token_type": "Bearer", "expires_in": 120.0}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")
    assert json.loads(response.body)["expires_in"] <= 120


@pytest.mark.asyncio
async def test_oauth_delegate_bridge_token_exchange_missing_access_token_is_502_not_keyerror():
    """When the upstream token response has no access_token, a dcr_bridge oauth_delegate exchange
    returns a clean 502 error body rather than raising a KeyError. _finish_bridge_mint asks
    _bridge_grant_from_token_response for a typed grant, gets None, and returns the "no_upstream_token"
    failure, which maps to 502; nothing indexes token_response["access_token"] on the bridge path."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate)
    upstream = {"token_type": "Bearer", "expires_in": 3600}

    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")

    assert response.status_code == 502
    assert json.loads(response.body)["error"] == "server_error"


@pytest.mark.asyncio
async def test_true_passthrough_bridge_token_exchange_returns_raw_upstream_token():
    """Only oauth_delegate mints. A true_passthrough dcr_bridge server relays the raw upstream token
    to the client, since that mode has no litellm identity to bind and the caller owns the token."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.true_passthrough)
    upstream = {"access_token": "UPSTREAM-SECRET-TOKEN", "token_type": "Bearer", "expires_in": 3600}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")

    body = json.loads(response.body)
    assert body["access_token"] == "UPSTREAM-SECRET-TOKEN"
    assert not body["access_token"].startswith("llm_env_")


@pytest.mark.asyncio
async def test_non_bridge_oauth_delegate_token_exchange_returns_raw_upstream_token():
    """An oauth_delegate server without dcr_bridge keeps the pre-change contract: the raw upstream
    token is returned, so flag-off behavior is byte-identical."""
    from litellm.types.mcp import MCPAuth

    server = _bridge_server(auth_type=MCPAuth.oauth_delegate, dcr_bridge=None)
    upstream = {"access_token": "UPSTREAM-SECRET-TOKEN", "token_type": "Bearer", "expires_in": 3600}
    response = await _exchange_for_bridge_server(server, upstream, key_hash="hashed-litellm-key-77")

    body = json.loads(response.body)
    assert body["access_token"] == "UPSTREAM-SECRET-TOKEN"


async def _exchange_persistence_attempted_for_auth_type(auth_type) -> bool:
    """Run exchange_token_with_server for a server of ``auth_type`` and report whether it attempted
    to persist the exchanged token server-side. The client-forwarded token modes must not persist:
    their contract is that the upstream token stays browser-held, minted/stored/refreshed nowhere."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        exchange_token_with_server,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="t",
        name="t",
        server_name="t",
        alias="t",
        transport=MCPTransport.http,
        auth_type=auth_type,
        client_id="cid",
        client_secret="cs",
        authorization_url="https://provider.com/oauth/authorize",
        token_url="https://provider.com/oauth/token",
    )
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    fake_http_response = MagicMock()
    fake_http_response.json.return_value = {"access_token": "tok", "refresh_token": "r", "token_type": "Bearer"}
    fake_http_response.raise_for_status = MagicMock()
    fake_http_client = MagicMock()
    fake_http_client.post = AsyncMock(return_value=fake_http_response)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
            return_value=fake_http_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._extract_user_id_from_request",
            new_callable=AsyncMock,
            return_value="admin-user",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints._store_per_user_token_server_side",
            new_callable=AsyncMock,
        ) as mock_store,
    ):
        await exchange_token_with_server(
            request=mock_request,
            mcp_server=server,
            grant_type="authorization_code",
            code="c",
            redirect_uri="http://127.0.0.1:3000/cb",
            client_id="cid",
            client_secret=None,
            code_verifier=None,
        )
    return mock_store.await_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
async def test_token_exchange_does_not_persist_for_client_forwarded_modes(auth_type):
    """The browser-only Authorize for true_passthrough / oauth_delegate must not write the upstream
    token to the DB: these modes forward a browser-held token and persist nothing server-side."""
    assert await _exchange_persistence_attempted_for_auth_type(auth_type) is False


@pytest.mark.asyncio
async def test_token_exchange_persists_for_oauth2():
    """Guard the test's own discriminator: a genuine oauth2 (authorization_code) server DOES persist,
    so the passthrough no-persist assertion above is meaningful and not vacuously true."""
    assert await _exchange_persistence_attempted_for_auth_type(MCPAuth.oauth2) is True


# -------------------------------------------------------------------
# OBO (token_exchange) Protected Resource Metadata: discovery must name the
# JWT-auth issuer the client SSOs with, not the gateway.
# -------------------------------------------------------------------

_OBO_RESOURCE = "https://litellm.example.com/mcp/obo_mcp"
_PATCH_ISSUERS = "litellm.proxy._experimental.mcp_server.discoverable_endpoints._jwt_auth_issuers"


def _obo_server(scopes=None):
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id="obo_mcp",
        name="obo_mcp",
        server_name="obo_mcp",
        alias="obo_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        scopes=scopes,
    )


def test_obo_protected_resource_response_names_jwt_issuers():
    """An OBO server's PRM points authorization_servers at the configured JWT issuers (the IdP that
    mints and validates the subject token), with the gateway resource echoed back."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _obo_protected_resource_response,
    )

    with patch(_PATCH_ISSUERS, return_value=["https://idp.example.com"]):
        response = _obo_protected_resource_response(_obo_server(scopes=["read"]), _OBO_RESOURCE)
    assert response == {
        "authorization_servers": ["https://idp.example.com"],
        "resource": _OBO_RESOURCE,
        "scopes_supported": ["read"],
    }


def test_obo_protected_resource_response_scopes_default_empty():
    """A scopeless OBO server reports scopes_supported as [] rather than None."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _obo_protected_resource_response,
    )

    with patch(_PATCH_ISSUERS, return_value=["https://idp.example.com"]):
        response = _obo_protected_resource_response(_obo_server(scopes=None), _OBO_RESOURCE)
    assert response["scopes_supported"] == []


def test_obo_protected_resource_response_falls_back_when_no_issuer():
    """With no JWT issuer configured, the OBO branch returns None so the caller falls back to the
    gateway-default PRM (discovery still works, it just can't name the IdP)."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _obo_protected_resource_response,
    )

    with patch(_PATCH_ISSUERS, return_value=[]):
        assert _obo_protected_resource_response(_obo_server(), _OBO_RESOURCE) is None


def test_obo_protected_resource_response_ignores_non_obo_server():
    """Non-OBO servers are not handled by this branch (returns None -> gateway default)."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _obo_protected_resource_response,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    oauth2_server = MCPServer(
        server_id="oauth2_mcp",
        name="oauth2_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
    )
    assert _obo_protected_resource_response(oauth2_server, _OBO_RESOURCE) is None


@pytest.mark.asyncio
async def test_build_oauth_protected_resource_response_obo_end_to_end():
    """End to end through the response builder: an OBO server's PRM advertises the JWT issuer as
    authorization_servers, proving the extracted branch is wired into the public discovery path."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _build_oauth_protected_resource_response,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    global_mcp_server_manager.registry["obo_mcp"] = _obo_server(scopes=["read"])

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with patch(_PATCH_ISSUERS, return_value=["https://idp.example.com"]):
            response = await _build_oauth_protected_resource_response(
                request=mock_request,
                mcp_server_name="obo_mcp",
                use_standard_pattern=True,
            )
        assert response["authorization_servers"] == ["https://idp.example.com"]
        assert response["resource"] == "https://litellm.example.com/mcp/obo_mcp"
    finally:
        global_mcp_server_manager.registry.clear()


def _token_request(headers):
    """A real Starlette request with case-insensitive headers (matches production)."""
    from starlette.requests import Request

    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/token", "headers": raw, "query_string": b""})


@pytest.fixture
def proxy_globals():
    """Inject the cache/prisma the OAuth token endpoint resolves identity through, and restore
    them afterward. These module globals are the proxy's real wiring points, so setting them is
    dependency injection rather than monkeypatching a class."""
    import litellm.proxy.proxy_server as ps

    saved = (ps.user_api_key_cache, ps.prisma_client)
    try:
        yield ps
    finally:
        ps.user_api_key_cache, ps.prisma_client = saved


@pytest.mark.asyncio
async def test_extract_user_id_reads_x_litellm_api_key_header(proxy_globals):
    """The LiteLLM key arrives on x-litellm-api-key (what Claude Desktop/Code send), not
    Authorization. Reading only Authorization dropped the identity, so the per-user token was
    never stored and the egress 401'd forever. Resolution must honor x-litellm-api-key."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key = "sk-alice-key"
    cache = UserApiKeyCache()
    await cache.async_set_cache(
        hash_token(key),
        UserAPIKeyAuth(token=hash_token(key), user_id="alice"),
        model_type=UserAPIKeyAuth,
    )
    proxy_globals.user_api_key_cache = cache
    proxy_globals.prisma_client = object()

    request = _token_request({"x-litellm-api-key": f"Bearer {key}"})
    assert await _extract_user_id_from_request(request) == "alice"


@pytest.mark.asyncio
async def test_extract_user_id_rehydrates_cross_replica_dict_cache(proxy_globals):
    """Cross-replica, async_get_cache hands back a serialized dict, not a UserAPIKeyAuth.
    Resolution must rehydrate it; the old getattr(cached, "user_id") returned None on a dict,
    which is exactly why a multi-replica gateway never found the stored token."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy._types import hash_token
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key = "sk-alice-key"
    cache = UserApiKeyCache()
    cache.in_memory_cache.set_cache(hash_token(key), {"token": hash_token(key), "user_id": "alice"})
    proxy_globals.user_api_key_cache = cache
    proxy_globals.prisma_client = object()

    request = _token_request({"Authorization": f"Bearer {key}"})
    assert await _extract_user_id_from_request(request) == "alice"


@pytest.mark.asyncio
async def test_extract_user_id_falls_back_to_db_on_cache_miss(proxy_globals):
    """A cache miss must read the key from the DB rather than returning None; the old code did a
    cache-only peek and skipped the DB, so any replica that hadn't just authenticated the key
    failed to store the token."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key = "sk-bob-key"

    class _FakePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            return UserAPIKeyAuth(token=token, user_id="db-bob")

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _FakePrisma()

    request = _token_request({"x-litellm-api-key": key})
    assert await _extract_user_id_from_request(request) == "db-bob"


@pytest.mark.asyncio
async def test_extract_user_id_none_without_litellm_key(proxy_globals):
    """No LiteLLM key on the request resolves to None without consulting the resolver."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = object()

    request = _token_request({"content-type": "application/json"})
    assert await _extract_user_id_from_request(request) is None


@pytest.mark.asyncio
async def test_extract_user_id_rejects_blocked_key(proxy_globals):
    """A blocked LiteLLM key must not resolve an identity. get_key_object returns the DB row without
    checking blocked/expiry (the main auth pipeline does, and the public token endpoint bypasses it),
    so a revoked key could otherwise overwrite the stored per-user OAuth token for its user."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class _FakePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            return UserAPIKeyAuth(token=token, user_id="blocked-user", blocked=True)

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _FakePrisma()

    request = _token_request({"x-litellm-api-key": "sk-blocked-key"})
    assert await _extract_user_id_from_request(request) is None


@pytest.mark.asyncio
async def test_extract_user_id_rejects_expired_key(proxy_globals):
    """An expired LiteLLM key must not resolve an identity, for the same reason as a blocked key."""
    from datetime import datetime, timedelta, timezone

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    class _FakePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            return UserAPIKeyAuth(token=token, user_id="expired-user", expires=expired)

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _FakePrisma()

    request = _token_request({"x-litellm-api-key": "sk-expired-key"})
    assert await _extract_user_id_from_request(request) is None


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_returns_resolved_key_for_active_key(proxy_globals):
    """The dcr_bridge mint seals the hash of the authorizing key so admission can reload the live
    record. For an active key the resolver returns exactly hash_token(key), the same value
    get_key_object and the whole cache/DB layer key the record by, so the sealed reference resolves
    back to this key at admission."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
        _ResolvedKey,
    )
    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key = "sk-alice-key"
    cache = UserApiKeyCache()
    await cache.async_set_cache(
        hash_token(key),
        UserAPIKeyAuth(token=hash_token(key), user_id="alice"),
        model_type=UserAPIKeyAuth,
    )
    proxy_globals.user_api_key_cache = cache
    proxy_globals.prisma_client = object()

    request = _token_request({"x-litellm-api-key": f"Bearer {key}"})
    resolved = await _resolve_active_litellm_key(request)
    assert isinstance(resolved, _ResolvedKey)
    assert resolved.key_hash == hash_token(key)


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_resolves_key_without_user_id(proxy_globals):
    """A valid team-scoped or service-account key has no user_id but is a legitimate credential, so it
    must still resolve to a hash and be able to mint a bridge envelope. Gating the resolver on user_id
    presence wrongly rejected these keys with invalid_request; the active-state gate now checks only
    blocked and expiry, and the key hash (not the user) is what the mint seals. The per-user token
    store still gets no user for such a key, since there is none to key a stored credential by."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _extract_user_id_from_request,
        _resolve_active_litellm_key,
        _ResolvedKey,
    )
    from litellm.proxy._types import UserAPIKeyAuth, hash_token
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key = "sk-team-scoped-key"
    cache = UserApiKeyCache()
    await cache.async_set_cache(
        hash_token(key),
        UserAPIKeyAuth(token=hash_token(key), user_id=None, team_id="team-x"),
        model_type=UserAPIKeyAuth,
    )
    proxy_globals.user_api_key_cache = cache
    proxy_globals.prisma_client = object()

    request = _token_request({"x-litellm-api-key": f"Bearer {key}"})
    resolved = await _resolve_active_litellm_key(request)
    assert isinstance(resolved, _ResolvedKey)
    assert resolved.key_hash == hash_token(key)
    assert await _extract_user_id_from_request(request) is None


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_rejects_blocked_key(proxy_globals):
    """A blocked key must not yield a hash, so no gateway-bound envelope is minted for a revoked key;
    the mint fails closed with invalid_request instead."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class _FakePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            return UserAPIKeyAuth(token=token, user_id="blocked-user", blocked=True)

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _FakePrisma()

    request = _token_request({"x-litellm-api-key": "sk-blocked-key"})
    assert await _resolve_active_litellm_key(request) == "no_active_key"


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_fails_closed_on_malformed_expiry(proxy_globals):
    """A key whose stored expires string does not parse must fail closed to no-hash (the mint then
    returns invalid_request), not surface an unhandled 500. The active-state check runs outside the
    resolver's try, so it must be total over a bad expires rather than letting datetime.fromisoformat
    raise. Before the fix this raised a ValueError instead of returning None."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class _FakePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            return UserAPIKeyAuth(token=token, user_id="u", expires="not-a-parseable-date")

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _FakePrisma()

    request = _token_request({"x-litellm-api-key": "sk-bad-expiry-key"})
    assert await _resolve_active_litellm_key(request) == "no_active_key"


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_no_active_key_without_litellm_key(proxy_globals):
    """No LiteLLM key on the request yields no hash without consulting the resolver."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = object()

    request = _token_request({"content-type": "application/json"})
    assert await _resolve_active_litellm_key(request) == "no_active_key"


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_db_outage_is_unavailable(proxy_globals):
    """A database outage while resolving the presented key is a retryable infrastructure failure, not
    the caller's fault, so the resolver reports "unavailable" (the mint statuses it 503) rather than
    collapsing it to the same value as a missing credential. is_database_service_unavailable_error
    classifies a connection error (an OSError) as an outage, matching admission's egress-side handling."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class _OutagePrisma:
        async def get_data(self, token, table_name, parent_otel_span=None, proxy_logging_obj=None):
            raise ConnectionError("connection refused")

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = _OutagePrisma()

    request = _token_request({"x-litellm-api-key": "sk-during-outage"})
    assert await _resolve_active_litellm_key(request) == "unavailable"


@pytest.mark.asyncio
async def test_resolve_active_litellm_key_no_database_is_unresolvable(proxy_globals):
    """With no database connection configured the gateway cannot verify the presented key at all, so
    the resolver reports "unresolvable" (the mint statuses it 500) instead of blaming the caller.
    Mirrors admission, which 500s a missing prisma_client on the egress side."""
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _resolve_active_litellm_key,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    proxy_globals.user_api_key_cache = UserApiKeyCache()
    proxy_globals.prisma_client = None

    request = _token_request({"x-litellm-api-key": "sk-no-db"})
    assert await _resolve_active_litellm_key(request) == "unresolvable"


@pytest.mark.asyncio
async def test_token_endpoint_uses_client_secret_basic_when_configured():
    """LIT-4091: a server with token_endpoint_auth_method=client_secret_basic must send the
    credentials as an HTTP Basic Authorization header and omit client_secret from the body;
    providers requiring Basic rejected body credentials with invalid_client."""
    import base64
    from unittest.mock import AsyncMock

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

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="basic_mcp",
        name="basic_mcp",
        server_name="basic_mcp",
        alias="basic_mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="basic-client",
        client_secret="basic-secret",
        authorization_url="https://idp.example.com/authorize",
        token_url="https://idp.example.com/oauth2/token",
        token_endpoint_auth_method="client_secret_basic",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "at",
        "token_type": "Bearer",
        "expires_in": 3599,
    }
    mock_response.raise_for_status = MagicMock()

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        mock_async_client = MagicMock()
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_async_client

        await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="http://localhost/callback",
            client_id="basic-client",
            mcp_server_name="basic_mcp",
            client_secret="basic-secret",
            code_verifier="verifier",
        )

    call_args = mock_async_client.post.call_args
    expected = "Basic " + base64.b64encode(b"basic-client:basic-secret").decode()
    assert call_args[1]["headers"]["Authorization"] == expected
    assert "client_secret" not in call_args[1]["data"]
    assert "client_id" not in call_args[1]["data"]
    assert call_args[1]["data"]["grant_type"] == "authorization_code"
    assert call_args[1]["data"]["code"] == "auth-code"


@pytest.mark.asyncio
async def test_token_endpoint_client_secret_basic_without_secret_returns_400():
    """A server configured client_secret_basic but missing its secret is a misconfiguration; the
    inbound /token endpoint surfaces it as a 400 rather than silently posting a downgraded request."""
    from fastapi import HTTPException, Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        token_endpoint,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    global_mcp_server_manager.registry.clear()
    oauth2_server = MCPServer(
        server_id="basic_no_secret",
        name="basic_no_secret",
        server_name="basic_no_secret",
        alias="basic_no_secret",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="basic-client",
        client_secret=None,
        authorization_url="https://idp.example.com/authorize",
        token_url="https://idp.example.com/oauth2/token",
        token_endpoint_auth_method="client_secret_basic",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm-proxy.example.com/"
    mock_request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        await token_endpoint(
            request=mock_request,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="http://localhost/callback",
            client_id="basic-client",
            mcp_server_name="basic_no_secret",
            client_secret=None,
            code_verifier="verifier",
        )
    assert exc_info.value.status_code == 400


# -------------------------------------------------------------------
# Non-oauth2 (auth_type=none, access-group gated) servers must not be
# driven through the gateway OAuth authorize/token/register/discovery
# flow, and must not be advertised as OAuth-protected in discovery docs.
# -------------------------------------------------------------------


def _access_group_none_server(server_name="access_group_server"):
    """A non-oauth2, access-group gated MCP server: no client_id, no OAuth."""
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id=server_name,
        name=server_name,
        server_name=server_name,
        alias=server_name,
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        access_groups=["eng"],
    )


@pytest.mark.asyncio
async def test_authorize_endpoint_rejects_non_oauth2_server():
    """authorize() against a none-auth server returns an accurate 'does not use OAuth' 400,
    not the misleading 'client_id is required' that fired before the auth_type was checked."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server = _access_group_none_server()
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            await authorize(
                request=mock_request,
                client_id=None,
                mcp_server_name="access_group_server",
                redirect_uri="http://127.0.0.1:60108/callback",
                state="test_state",
            )
        assert exc_info.value.status_code == 400
        detail_text = str(exc_info.value.detail)
        assert "does not use OAuth" in detail_text
        assert "client_id is required" not in detail_text
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_token_endpoint_rejects_non_oauth2_server():
    """token_endpoint() against a none-auth server returns 'does not use OAuth' 400 instead
    of the misleading 'token url is not set'."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server = _access_group_none_server()
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            await token_endpoint(
                request=mock_request,
                grant_type="authorization_code",
                code="auth-code",
                redirect_uri="http://localhost/callback",
                client_id="some-client",
                mcp_server_name="access_group_server",
                client_secret=None,
                code_verifier="verifier",
            )
        assert exc_info.value.status_code == 400
        detail_text = str(exc_info.value.detail)
        assert "does not use OAuth" in detail_text
        assert "token url is not set" not in detail_text
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_register_client_rejects_non_oauth2_server():
    """register_client() against a named none-auth server returns 'does not use OAuth' 400
    instead of the misleading 'authorization url is not set'."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server = _access_group_none_server()
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            with patch(
                "litellm.proxy._experimental.mcp_server.discoverable_endpoints._read_request_body",
                new=AsyncMock(return_value={}),
            ):
                await register_client(request=mock_request, mcp_server_name="access_group_server")
        assert exc_info.value.status_code == 400
        detail_text = str(exc_info.value.detail)
        assert "does not use OAuth" in detail_text
        assert "authorization url is not set" not in detail_text
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_protected_resource_404_for_non_oauth2_server():
    """Discovery must not advertise a none-auth server as an OAuth-protected resource."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_protected_resource_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server = _access_group_none_server()
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            await _build_oauth_protected_resource_response(
                request=mock_request,
                mcp_server_name="access_group_server",
                use_standard_pattern=False,
            )
        assert exc_info.value.status_code == 404
        assert "not an OAuth-protected resource" in str(exc_info.value.detail)
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_authorization_server_404_for_non_oauth2_server():
    """Discovery must not advertise a none-auth server as an OAuth authorization server."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_authorization_server_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    server = _access_group_none_server()
    global_mcp_server_manager.registry[server.server_id] = server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with pytest.raises(HTTPException) as exc_info:
            _build_oauth_authorization_server_response(
                request=mock_request,
                mcp_server_name="access_group_server",
            )
        assert exc_info.value.status_code == 404
        assert "not an OAuth authorization server" in str(exc_info.value.detail)
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_protected_resource_passthrough_none_auth_not_404():
    """Regression guard for the protected-resource auth_type gate placement: a none-auth
    server that opted into OAuth pass-through must still proxy upstream metadata, it must
    NOT be 404'd. The gate has to sit after the pass-through branch."""
    try:
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
    passthrough_server = MCPServer(
        server_id="passthrough_server",
        name="passthrough_server",
        server_name="passthrough_server",
        alias="passthrough_server",
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        oauth_passthrough=True,
        extra_headers=["Authorization"],
    )
    global_mcp_server_manager.registry[passthrough_server.server_id] = passthrough_server

    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.fetch_upstream_oauth_protected_resource",
            new=AsyncMock(return_value={"authorization_servers": ["https://upstream-idp.example.com"]}),
        ):
            response = await _build_oauth_protected_resource_response(
                request=mock_request,
                mcp_server_name="passthrough_server",
                use_standard_pattern=False,
            )
        assert response["authorization_servers"] == ["https://upstream-idp.example.com"]
        assert response["resource"].endswith("/passthrough_server/mcp")
    finally:
        global_mcp_server_manager.registry.clear()


@pytest.mark.asyncio
async def test_oauth_protected_resource_404_for_unknown_server_name():
    """A discovery request for an unknown server name returns the same 404 as a non-oauth2
    server (not a 200 metadata doc with broken URLs), so the well-known paths cannot be used
    to enumerate non-OAuth server names."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_protected_resource_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        await _build_oauth_protected_resource_response(
            request=mock_request,
            mcp_server_name="does_not_exist",
            use_standard_pattern=True,
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_oauth_authorization_server_404_for_unknown_server_name():
    """A named authorization-server discovery request for an unknown server returns 404, not a
    200 metadata document pointing at non-existent /{name}/authorize and /{name}/token."""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            _build_oauth_authorization_server_response,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()
    mock_request = MagicMock()
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    with pytest.raises(HTTPException) as exc_info:
        _build_oauth_authorization_server_response(
            request=mock_request,
            mcp_server_name="does_not_exist",
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_store_per_user_token_server_side_invalidates_v2_token_cache():
    """A token stored by the OAuth callback (code exchange or refresh) drops the v2 per-user
    token cache entry, so egress stops serving the replaced token immediately instead of
    until its TTL."""
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as manager_module
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _store_per_user_token_server_side,
    )
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv-cb-1",
        name="cb_server",
        url="https://upstream.example/mcp",
        transport="http",
        auth_type=MCPAuth.oauth2,
    )
    invalidate_mock = AsyncMock(return_value=None)
    cache_set_mock = AsyncMock(return_value=None)

    with (
        patch(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.store_user_oauth_credential",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.mcp_per_user_token_cache.set",
            new=cache_set_mock,
        ),
        patch.object(
            manager_module.global_mcp_server_manager,
            "invalidate_user_oauth_token_cache",
            new=invalidate_mock,
        ),
    ):
        await _store_per_user_token_server_side(
            server=server,
            user_id="user-cb-1",
            token_response={"access_token": "fresh-tok", "expires_in": 3600},
        )

    invalidate_mock.assert_awaited_once_with("user-cb-1", "srv-cb-1")
    cache_set_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_per_user_token_server_side_skips_invalidate_when_db_write_fails():
    """A failed DB write neither warms the v1 cache nor drops the v2 cache entry; the
    previously stored token is still the truth."""
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as manager_module
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        _store_per_user_token_server_side,
    )
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv-cb-2",
        name="cb_server_2",
        url="https://upstream.example/mcp",
        transport="http",
        auth_type=MCPAuth.oauth2,
    )
    invalidate_mock = AsyncMock(return_value=None)
    cache_set_mock = AsyncMock(return_value=None)

    with (
        patch(
            "litellm.proxy.utils.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.store_user_oauth_credential",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.oauth2_token_cache.mcp_per_user_token_cache.set",
            new=cache_set_mock,
        ),
        patch.object(
            manager_module.global_mcp_server_manager,
            "invalidate_user_oauth_token_cache",
            new=invalidate_mock,
        ),
    ):
        await _store_per_user_token_server_side(
            server=server,
            user_id="user-cb-2",
            token_response={"access_token": "fresh-tok", "expires_in": 3600},
        )

    invalidate_mock.assert_not_awaited()
    cache_set_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_token_exchange_pairs_client_secret_with_server_client_id():
    """Re-auth regression: the register short-circuit hands the browser a placeholder
    ``client_secret: "dummy"``, which the browser echoes back to /token. The server-side
    persisted client_id wins the resolution, so the secret must come from the same (server)
    source; pairing the persisted public PKCE client (no stored secret) with the caller's
    placeholder makes the IdP reject the exchange with 401 on every re-auth."""
    from fastapi import Request

    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        exchange_token_with_server,
    )
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    server = MCPServer(
        server_id="srv-1",
        name="srv-1",
        server_name="srv-1",
        alias="srv-1",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="persisted-client",
        client_secret=None,
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
    )

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://litellm.example.com/"
    mock_request.headers = {}

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"access_token": "at", "token_type": "Bearer"}
    mock_async_client = MagicMock()
    mock_async_client.post = AsyncMock(return_value=mock_response)

    with patch(
        "litellm.proxy._experimental.mcp_server.discoverable_endpoints.get_async_httpx_client",
        return_value=mock_async_client,
    ):
        await exchange_token_with_server(
            request=mock_request,
            mcp_server=server,
            grant_type="authorization_code",
            code="auth-code",
            redirect_uri="https://litellm.example.com/ui/mcp/oauth/callback",
            client_id="srv-1",
            client_secret="dummy",
            code_verifier="verifier",
        )

    sent = mock_async_client.post.call_args.kwargs["data"]
    assert sent["client_id"] == "persisted-client"
    assert "client_secret" not in sent
