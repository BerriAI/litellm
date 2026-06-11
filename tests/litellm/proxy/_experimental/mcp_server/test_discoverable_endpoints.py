"""Tests for MCP OAuth discoverable endpoints"""

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch

TRUSTED_PROXY_IP = "10.0.0.5"
TRUSTED_PROXY_RANGES = ["10.0.0.0/8"]


def set_request_from_trusted_proxy(mock_request):
    mock_request.client = MagicMock()
    mock_request.client.host = TRUSTED_PROXY_IP


@pytest.fixture
def trusted_proxy_origin_headers():
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.discoverable_endpoints.IPAddressUtils.is_request_from_trusted_proxy",
            return_value=True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.oauth_utils.IPAddressUtils.is_request_from_trusted_proxy",
            return_value=True,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_authorize_endpoint_includes_response_type():
    """Test that authorize endpoint includes response_type=code parameter (fixes #15684)"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
async def test_authorize_endpoint_forwards_pkce_parameters():
    """Test that authorize endpoint forwards PKCE parameters (code_challenge and code_challenge_method)"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

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
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
async def test_authorize_endpoint_respects_x_forwarded_proto(
    trusted_proxy_origin_headers,
):
    """Test that authorize endpoint uses X-Forwarded-Proto header to construct correct redirect_uri"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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
async def test_token_endpoint_respects_x_forwarded_proto(
    trusted_proxy_origin_headers,
):
    """Test that token endpoint uses X-Forwarded-Proto header for redirect_uri"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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

        # Call token endpoint
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
async def test_oauth_protected_resource_standard_pattern():
    """Test that oauth_protected_resource_mcp_standard returns standard MCP URL pattern (/mcp/{server_name})"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_protected_resource_mcp_standard,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_server",
        name="test_server",
        server_name="test_server",
        alias="test_server",
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

    # Call the standard pattern endpoint
    response = await oauth_protected_resource_mcp_standard(
        request=mock_request,
        mcp_server_name="test_server",
    )

    # Verify response uses standard MCP pattern: /mcp/{server_name}
    assert response["resource"] == "https://litellm.example.com/mcp/test_server"
    assert (
        response["authorization_servers"][0]
        == "https://litellm.example.com/test_server"
    )
    assert response["scopes_supported"] == oauth2_server.scopes


@pytest.mark.asyncio
async def test_oauth_protected_resource_legacy_pattern():
    """Test that oauth_protected_resource_mcp returns legacy URL pattern (/{server_name}/mcp)"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_protected_resource_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Clear registry
    global_mcp_server_manager.registry.clear()

    # Create mock OAuth2 server
    oauth2_server = MCPServer(
        server_id="test_server",
        name="test_server",
        server_name="test_server",
        alias="test_server",
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

    # Call the legacy pattern endpoint
    response = await oauth_protected_resource_mcp(
        request=mock_request,
        mcp_server_name="test_server",
    )

    # Verify response uses legacy pattern: /{server_name}/mcp
    assert response["resource"] == "https://litellm.example.com/test_server/mcp"
    assert (
        response["authorization_servers"][0]
        == "https://litellm.example.com/test_server"
    )
    assert response["scopes_supported"] == oauth2_server.scopes


@pytest.mark.asyncio
async def test_oauth_protected_resource_respects_x_forwarded_proto(
    trusted_proxy_origin_headers,
):
    """Test that oauth_protected_resource_mcp uses X-Forwarded-Proto for URLs"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_protected_resource_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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
async def test_oauth_authorization_server_respects_x_forwarded_proto(
    trusted_proxy_origin_headers,
):
    """Test that oauth_authorization_server_mcp uses X-Forwarded-Proto for URLs"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            oauth_authorization_server_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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
async def test_register_client_respects_x_forwarded_proto(
    trusted_proxy_origin_headers,
):
    """Test that register_client uses X-Forwarded-Proto for redirect_uris"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            register_client,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    global_mcp_server_manager.registry.clear()

    # Mock request with http base_url but X-Forwarded-Proto: https
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://proxy.litellm.example/"  # HTTP
    mock_request.headers = {"X-Forwarded-Proto": "https"}  # Behind HTTPS proxy
    set_request_from_trusted_proxy(mock_request)

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
async def test_authorize_endpoint_respects_x_forwarded_host(
    trusted_proxy_origin_headers,
):
    """Test that authorize endpoint uses X-Forwarded-Host and X-Forwarded-Proto to construct correct redirect_uri"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            authorize,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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
async def test_token_endpoint_respects_x_forwarded_host(
    trusted_proxy_origin_headers,
):
    """Test that token endpoint uses X-Forwarded-Host and X-Forwarded-Proto for redirect_uri"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            token_endpoint,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
        from fastapi import Request
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
    set_request_from_trusted_proxy(mock_request)

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

        # Call token endpoint
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
    base_url,
    x_forwarded_proto,
    x_forwarded_host,
    x_forwarded_port,
    expected_url,
    trusted_proxy_origin_headers,
):
    """Comprehensive test for get_request_base_url with various header combinations"""
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            get_request_base_url,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.base_url = base_url
    set_request_from_trusted_proxy(mock_request)

    # Build headers dict
    headers = {}
    if x_forwarded_proto:
        headers["X-Forwarded-Proto"] = x_forwarded_proto
    if x_forwarded_host:
        headers["X-Forwarded-Host"] = x_forwarded_host
    if x_forwarded_port:
        headers["X-Forwarded-Port"] = x_forwarded_port

    # Mock headers.get() to return our test values
    def mock_get(header_name, default=None):
        return headers.get(header_name, default)

    mock_request.headers.get = mock_get

    # Test the function
    result = get_request_base_url(mock_request)

    # Verify result
    assert result == expected_url, (
        f"Expected '{expected_url}' but got '{result}'\n"
        f"Input: base_url={base_url}, "
        f"X-Forwarded-Proto={x_forwarded_proto}, "
        f"X-Forwarded-Host={x_forwarded_host}, "
        f"X-Forwarded-Port={x_forwarded_port}"
    )


def test_get_request_base_url_ignores_forwarded_headers_from_untrusted_client():
    try:
        from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
            get_request_base_url,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP discoverable endpoints not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://gateway.example.com/mcp"
    mock_request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "attacker.example.com",
        "X-Forwarded-Port": "443",
    }
    mock_request.client = MagicMock()
    mock_request.client.host = "203.0.113.10"

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "use_x_forwarded_for": True,
            "mcp_trusted_proxy_ranges": TRUSTED_PROXY_RANGES,
        },
        create=True,
    ):
        assert get_request_base_url(mock_request) == "https://gateway.example.com/mcp"


def test_validate_trusted_redirect_uri_rejects_spoofed_forwarded_host():
    try:
        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            validate_trusted_redirect_uri,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP OAuth utilities not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "https://gateway.example.com/"
    mock_request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "attacker.example.com",
    }
    mock_request.client = MagicMock()
    mock_request.client.host = "203.0.113.10"

    with (
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {
                "use_x_forwarded_for": True,
                "mcp_trusted_proxy_ranges": TRUSTED_PROXY_RANGES,
            },
            create=True,
        ),
        pytest.raises(HTTPException),
    ):
        validate_trusted_redirect_uri(
            mock_request,
            "https://attacker.example.com/callback",
        )


def test_validate_trusted_redirect_uri_allows_forwarded_origin_from_trusted_proxy(
    trusted_proxy_origin_headers,
):
    try:
        from litellm.proxy._experimental.mcp_server.oauth_utils import (
            validate_trusted_redirect_uri,
        )
        from fastapi import Request
    except ImportError:
        pytest.skip("MCP OAuth utilities not available")

    mock_request = MagicMock(spec=Request)
    mock_request.base_url = "http://localhost:4000/"
    mock_request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "proxy.example.com",
    }
    set_request_from_trusted_proxy(mock_request)

    validate_trusted_redirect_uri(
        mock_request,
        "https://proxy.example.com/callback",
    )
