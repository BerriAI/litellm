"""
Tests for the Unified Authentication System

This module tests the unified auth approach that consolidates authentication
across all LiteLLM proxy endpoints, including MCP.

Following the guidance from https://fastapi.tiangolo.com/tutorial/security/
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Request
from starlette.datastructures import Headers

from litellm.proxy._types import SpecialHeaders, UserAPIKeyAuth


class TestLiteLLMSecurityScheme:
    """Tests for the LiteLLMSecurityScheme class."""

    @pytest.fixture
    def security_scheme(self):
        from litellm.proxy.auth.unified_auth import LiteLLMSecurityScheme

        return LiteLLMSecurityScheme()

    def test_extract_api_key_from_custom_header(self, security_scheme):
        """Test extraction from x-litellm-api-key header."""
        headers = Headers({SpecialHeaders.custom_litellm_api_key.value: "sk-test-key"})
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result == "sk-test-key"

    def test_extract_api_key_from_authorization_header(self, security_scheme):
        """Test extraction from Authorization header."""
        headers = Headers(
            {SpecialHeaders.openai_authorization.value: "Bearer sk-test-key"}
        )
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result == "Bearer sk-test-key"

    def test_extract_api_key_from_azure_header(self, security_scheme):
        """Test extraction from api-key header (Azure style)."""
        headers = Headers({SpecialHeaders.azure_authorization.value: "azure-api-key"})
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result == "azure-api-key"

    def test_extract_api_key_from_anthropic_header(self, security_scheme):
        """Test extraction from x-api-key header (Anthropic style)."""
        headers = Headers(
            {SpecialHeaders.anthropic_authorization.value: "anthropic-api-key"}
        )
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result == "anthropic-api-key"

    def test_extract_api_key_priority_custom_over_auth(self, security_scheme):
        """Test that custom header takes priority over Authorization."""
        headers = Headers(
            {
                SpecialHeaders.custom_litellm_api_key.value: "custom-key",
                SpecialHeaders.openai_authorization.value: "Bearer auth-key",
            }
        )
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result == "custom-key"

    def test_extract_api_key_no_headers(self, security_scheme):
        """Test extraction when no auth headers present."""
        headers = Headers({})
        result = security_scheme._extract_api_key_from_headers(headers)
        assert result is None


class TestUnifiedAuth:
    """Tests for the UnifiedAuth class."""

    @pytest.fixture
    def unified_auth(self):
        from litellm.proxy.auth.unified_auth import UnifiedAuth

        return UnifiedAuth()

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        request.headers = Headers(
            {SpecialHeaders.openai_authorization.value: "Bearer sk-test-key"}
        )
        return request

    @pytest.mark.asyncio
    async def test_unified_auth_calls_user_api_key_auth(
        self, unified_auth, mock_request
    ):
        """Test that UnifiedAuth calls the standard user_api_key_auth function."""
        expected_auth = UserAPIKeyAuth(user_id="test-user")

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=expected_auth),
        ) as mock_auth:
            result = await unified_auth(mock_request)

            mock_auth.assert_called_once()
            assert result.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_unified_auth_handles_missing_key(self, unified_auth):
        """Test that UnifiedAuth handles missing API key."""
        request = MagicMock(spec=Request)
        request.headers = Headers({})

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth()),
        ) as mock_auth:
            _result = await unified_auth(request)

            mock_auth.assert_called_once()
            call_kwargs = mock_auth.call_args
            assert call_kwargs[1].get("api_key") == "" or call_kwargs[0][0] == ""

    @pytest.mark.asyncio
    async def test_unified_auth_raises_on_auth_failure(
        self, unified_auth, mock_request
    ):
        """Test that UnifiedAuth raises HTTPException on auth failure."""
        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Invalid key")
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await unified_auth(mock_request)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_unified_auth_no_auto_error(self, mock_request):
        """Test UnifiedAuth with auto_error=False returns empty auth on failure."""
        from litellm.proxy.auth.unified_auth import UnifiedAuth

        unified_auth = UnifiedAuth(auto_error=False)

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Invalid key")
            ),
        ):
            result = await unified_auth(mock_request)

            assert isinstance(result, UserAPIKeyAuth)


class TestUnifiedAuthForMCP:
    """Tests for the UnifiedAuthForMCP class."""

    @pytest.fixture
    def mcp_auth(self):
        from litellm.proxy.auth.unified_auth import UnifiedAuthForMCP

        return UnifiedAuthForMCP()

    @pytest.fixture
    def mock_request_with_mcp_servers(self):
        request = MagicMock(spec=Request)
        request.headers = Headers(
            {
                SpecialHeaders.openai_authorization.value: "Bearer sk-test-key",
                SpecialHeaders.mcp_servers.value: "server1,server2",
            }
        )
        return request

    @pytest.mark.asyncio
    async def test_mcp_auth_uses_standard_auth(
        self, mcp_auth, mock_request_with_mcp_servers
    ):
        """Test that MCP auth uses the standard user_api_key_auth."""
        expected_auth = UserAPIKeyAuth(user_id="test-user")

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=expected_auth),
        ):
            result = await mcp_auth(mock_request_with_mcp_servers)
            assert result.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_mcp_auth_oauth2_passthrough_disabled(self):
        """Test MCP auth with OAuth2 passthrough disabled."""
        from litellm.proxy.auth.unified_auth import UnifiedAuthForMCP

        mcp_auth = UnifiedAuthForMCP(allow_oauth2_passthrough=False)

        request = MagicMock(spec=Request)
        request.headers = Headers({})

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=UserAPIKeyAuth()),
        ) as mock_auth:
            await mcp_auth(request)
            mock_auth.assert_called_once()


class TestMCPRequestHandlerUnifiedAuth:
    """Tests for MCPRequestHandler.authenticate_request - the unified MCP auth method."""

    @pytest.fixture
    def mock_request(self):
        request = MagicMock(spec=Request)
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.headers.__iter__ = MagicMock(return_value=iter([]))
        request.scope = {"path": "/mcp/test"}
        return request

    @pytest.mark.asyncio
    async def test_authenticate_request_with_api_key(self, mock_request):
        """Test authenticate_request with valid API key."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        mock_request.headers.get.side_effect = lambda k, d=None: {
            SpecialHeaders.custom_litellm_api_key.value: "sk-test-key",
        }.get(k, d)

        expected_auth = UserAPIKeyAuth(user_id="test-user")

        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            new=AsyncMock(return_value=expected_auth),
        ):
            with patch(
                "litellm.proxy.auth.auth_utils.get_request_route",
                return_value="/mcp/test",
            ):
                result = await MCPRequestHandler.authenticate_request(mock_request)
                assert result.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_authenticate_request_well_known_public(self, mock_request):
        """Test that .well-known routes are public."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        mock_request.headers.get.return_value = None

        with patch(
            "litellm.proxy.auth.auth_utils.get_request_route",
            return_value="/.well-known/oauth-authorization-server",
        ):
            result = await MCPRequestHandler.authenticate_request(mock_request)
            assert isinstance(result, UserAPIKeyAuth)


class TestUnifiedAuthMiddleware:
    """Tests for the UnifiedAuthMiddleware class."""

    @pytest.fixture
    def mock_app(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_middleware_passes_non_http(self, mock_app):
        """Test that non-HTTP requests pass through unchanged."""
        from litellm.proxy.middleware.unified_auth_middleware import (
            UnifiedAuthMiddleware,
        )

        middleware = UnifiedAuthMiddleware(mock_app)
        scope = {"type": "websocket", "path": "/ws"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        mock_app.assert_called_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_middleware_allows_public_routes(self, mock_app):
        """Test that public routes pass through without auth."""
        from litellm.proxy.middleware.unified_auth_middleware import (
            UnifiedAuthMiddleware,
        )

        middleware = UnifiedAuthMiddleware(mock_app)
        scope = {"type": "http", "path": "/health"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        mock_app.assert_called_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_middleware_allows_well_known(self, mock_app):
        """Test that .well-known routes pass through."""
        from litellm.proxy.middleware.unified_auth_middleware import (
            UnifiedAuthMiddleware,
        )

        middleware = UnifiedAuthMiddleware(mock_app)
        scope = {"type": "http", "path": "/.well-known/openid-configuration"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        mock_app.assert_called_once_with(scope, receive, send)


class TestConsistencyBetweenMCPAndStandardAuth:
    """
    Tests to verify that MCP auth and standard auth behave consistently.

    This addresses the issue mentioned in the Slack thread where MCP uses
    a completely different auth middleware from other endpoints.
    """

    @pytest.mark.asyncio
    async def test_same_key_same_result(self):
        """Test that the same API key produces the same auth result in both systems."""
        from litellm.proxy.auth.unified_auth import UnifiedAuth, UnifiedAuthForMCP

        standard_auth = UnifiedAuth()
        mcp_auth = UnifiedAuthForMCP()

        expected_auth = UserAPIKeyAuth(user_id="test-user", team_id="test-team")

        request = MagicMock(spec=Request)
        request.headers = Headers(
            {SpecialHeaders.openai_authorization.value: "Bearer sk-same-key"}
        )

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(return_value=expected_auth),
        ):
            standard_result = await standard_auth(request)
            mcp_result = await mcp_auth(request)

            assert standard_result.user_id == mcp_result.user_id
            assert standard_result.team_id == mcp_result.team_id

    @pytest.mark.asyncio
    async def test_auth_failure_same_behavior(self):
        """Test that auth failures are handled the same way."""
        from litellm.proxy.auth.unified_auth import UnifiedAuth, UnifiedAuthForMCP

        standard_auth = UnifiedAuth()
        mcp_auth = UnifiedAuthForMCP()

        request = MagicMock(spec=Request)
        request.headers = Headers(
            {SpecialHeaders.openai_authorization.value: "Bearer invalid-key"}
        )

        with patch(
            "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
            new=AsyncMock(
                side_effect=HTTPException(status_code=401, detail="Invalid key")
            ),
        ):
            with pytest.raises(HTTPException) as standard_exc:
                await standard_auth(request)

            with pytest.raises(HTTPException) as mcp_exc:
                await mcp_auth(request)

            assert standard_exc.value.status_code == mcp_exc.value.status_code
