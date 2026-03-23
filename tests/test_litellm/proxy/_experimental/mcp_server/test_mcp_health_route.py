"""
Tests for the MCP health passthrough endpoint (Issue #24450).

Verifies that /{mcp_server_name}/health is forwarded to the upstream
MCP server's /health endpoint and the response is returned verbatim.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

_MCP_MANAGER_PATH = (
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"
)
_IP_UTILS_PATH = "litellm.proxy.auth.ip_address_utils.IPAddressUtils"
_HTTP_CLIENT_PATH = "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"


def _make_request():
    request = MagicMock()
    request.scope = {"type": "http"}
    return request


def _make_server(url="http://mcp.example.com", transport=None):
    try:
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
        from litellm.proxy._types import MCPTransport
    except ImportError:
        return None

    return MCPServer(
        server_id="test-server-id",
        name="my_mcp",
        server_name="my_mcp",
        url=url,
        transport=transport or MCPTransport.http,
    )


class TestDynamicMcpHealthRoute:
    """Unit tests for dynamic_mcp_health_route in proxy_server.py."""

    @pytest.mark.asyncio
    async def test_health_passthrough_success(self):
        """Upstream /health response is forwarded back to the caller."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
        except ImportError:
            pytest.skip("proxy_server not available")

        mcp_server = _make_server()
        if mcp_server is None:
            pytest.skip("MCP types not available")

        upstream_response = MagicMock()
        upstream_response.content = b'{"status": "ok"}'
        upstream_response.status_code = 200
        upstream_response.headers = {"content-type": "application/json"}

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=upstream_response)

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
            patch(_HTTP_CLIENT_PATH, return_value=mock_http_client),
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = mcp_server

            response = await dynamic_mcp_health_route("my_mcp", _make_request())

        assert response.status_code == 200
        assert response.body == b'{"status": "ok"}'
        mock_http_client.get.assert_awaited_once_with("http://mcp.example.com/health")

    @pytest.mark.asyncio
    async def test_health_route_strips_trailing_slash_from_url(self):
        """Trailing slash on the server URL must not produce a double slash."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
        except ImportError:
            pytest.skip("proxy_server not available")

        server_with_slash = _make_server(url="http://mcp.example.com/")
        if server_with_slash is None:
            pytest.skip("MCP types not available")

        upstream_response = MagicMock()
        upstream_response.content = b'{"status": "ok"}'
        upstream_response.status_code = 200
        upstream_response.headers = {"content-type": "application/json"}

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=upstream_response)

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
            patch(_HTTP_CLIENT_PATH, return_value=mock_http_client),
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = server_with_slash

            await dynamic_mcp_health_route("slash_mcp", _make_request())

        # Must not produce double slash
        mock_http_client.get.assert_awaited_once_with("http://mcp.example.com/health")

    @pytest.mark.asyncio
    async def test_health_route_unknown_server_returns_404(self):
        """Returns 404 when the MCP server name is not registered."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
        except ImportError:
            pytest.skip("proxy_server not available")

        from fastapi import HTTPException

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await dynamic_mcp_health_route("unknown_mcp", _make_request())

        assert exc_info.value.status_code == 404
        assert "unknown_mcp" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_health_route_no_url_returns_400(self):
        """Returns 400 when the MCP server has no URL (e.g. stdio-only servers)."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
            from litellm.proxy._types import MCPTransport
        except ImportError:
            pytest.skip("proxy_server not available")

        from fastapi import HTTPException

        server_no_url = _make_server(url=None, transport=MCPTransport.stdio)
        if server_no_url is None:
            pytest.skip("MCP types not available")

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = server_no_url

            with pytest.raises(HTTPException) as exc_info:
                await dynamic_mcp_health_route("stdio_mcp", _make_request())

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_health_route_upstream_error_returns_502(self):
        """Returns 502 when the upstream health request raises an exception."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
        except ImportError:
            pytest.skip("proxy_server not available")

        from fastapi import HTTPException

        mcp_server = _make_server()
        if mcp_server is None:
            pytest.skip("MCP types not available")

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
            patch(_HTTP_CLIENT_PATH, return_value=mock_http_client),
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = mcp_server

            with pytest.raises(HTTPException) as exc_info:
                await dynamic_mcp_health_route("my_mcp", _make_request())

        assert exc_info.value.status_code == 502
        assert "connection refused" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_health_route_forwards_upstream_status_code(self):
        """Non-200 status codes from the upstream server are forwarded as-is."""
        try:
            from litellm.proxy.proxy_server import dynamic_mcp_health_route
        except ImportError:
            pytest.skip("proxy_server not available")

        mcp_server = _make_server()
        if mcp_server is None:
            pytest.skip("MCP types not available")

        upstream_response = MagicMock()
        upstream_response.content = b'{"status": "degraded"}'
        upstream_response.status_code = 503
        upstream_response.headers = {"content-type": "application/json"}

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=upstream_response)

        with (
            patch(_MCP_MANAGER_PATH) as mock_manager,
            patch(_IP_UTILS_PATH) as mock_ip_utils,
            patch(_HTTP_CLIENT_PATH, return_value=mock_http_client),
        ):
            mock_ip_utils.get_mcp_client_ip.return_value = "127.0.0.1"
            mock_manager.get_mcp_server_by_name.return_value = mcp_server

            response = await dynamic_mcp_health_route("my_mcp", _make_request())

        assert response.status_code == 503
        assert response.body == b'{"status": "degraded"}'
