"""
Tests for MCPDebug — MCP OAuth2 debug response headers.
"""

import asyncio
from unittest.mock import MagicMock

from litellm.proxy._experimental.mcp_server.mcp_debug import (
    MCP_DEBUG_REQUEST_HEADER,
    MCPDebug,
)


class TestIsDebugEnabled:
    def test_enabled_true(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "true"}) is True

    def test_enabled_yes(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "yes"}) is True

    def test_enabled_one(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "1"}) is True

    def test_disabled_false(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "false"}) is False

    def test_disabled_missing(self):
        assert MCPDebug.is_debug_enabled({"other-header": "value"}) is False

    def test_case_insensitive_header_name(self):
        assert MCPDebug.is_debug_enabled({"X-LiteLLM-MCP-Debug": "true"}) is True

    def test_case_insensitive_value(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "TRUE"}) is True


class TestMask:
    def test_none_returns_none_label(self):
        assert MCPDebug._mask(None) == "(none)"

    def test_empty_returns_none_label(self):
        assert MCPDebug._mask("") == "(none)"

    def test_short_value_unchanged(self):
        # visible_prefix=6 + visible_suffix=4 = 10, so <= 10 chars unchanged
        assert MCPDebug._mask("sk-1234") == "sk-1234"

    def test_long_value_masked(self):
        result = MCPDebug._mask("Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert result.startswith("Bearer")
        assert result.endswith("VCJ9")
        assert "****" in result or "**" in result

    def test_litellm_key_masked(self):
        result = MCPDebug._mask("Bearer sk-1234567890abcdef")
        assert result.startswith("Bearer")
        assert "sk-1234567890abcdef" not in result


class TestBuildDebugHeaders:
    def test_basic_no_auth(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={"host": "localhost"},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert headers["x-mcp-debug-inbound-auth"] == "(none)"
        assert headers["x-mcp-debug-oauth2-token"] == "(none)"
        assert headers["x-mcp-debug-auth-resolution"] == "no-auth"
        assert headers["x-mcp-debug-outbound-url"] == "https://mcp.example.com"
        assert headers["x-mcp-debug-server-auth-type"] == "oauth2"

    def test_litellm_key_in_dedicated_header(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "x-litellm-api-key": "Bearer sk-1234567890abcdef",
                "host": "localhost",
            },
            oauth2_headers=None,
            litellm_api_key="Bearer sk-1234567890abcdef",
            auth_resolution="no-auth",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert "x-litellm-api-key=" in headers["x-mcp-debug-inbound-auth"]
        assert headers["x-mcp-debug-oauth2-token"] == "(none)"

    def test_same_key_flagged(self):
        """When Authorization and x-litellm-api-key carry the same token."""
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "authorization": "Bearer sk-1234567890abcdef",
            },
            oauth2_headers={"Authorization": "Bearer sk-1234567890abcdef"},
            litellm_api_key="Bearer sk-1234567890abcdef",
            auth_resolution="oauth2-passthrough",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert "SAME_AS_LITELLM_KEY" in headers["x-mcp-debug-oauth2-token"]

    def test_different_tokens_not_flagged(self):
        """When OAuth2 token is different from LiteLLM key."""
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "x-litellm-api-key": "Bearer sk-litellm-key-here",
                "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.atlassian",
            },
            oauth2_headers={
                "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.atlassian"
            },
            litellm_api_key="Bearer sk-litellm-key-here",
            auth_resolution="oauth2-passthrough",
            server_url="https://mcp.atlassian.com/v1/mcp",
            server_auth_type="oauth2",
        )
        assert "SAME_AS_LITELLM_KEY" not in headers["x-mcp-debug-oauth2-token"]
        assert headers["x-mcp-debug-auth-resolution"] == "oauth2-passthrough"

    def test_m2m_resolution(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={"x-litellm-api-key": "Bearer sk-key"},
            oauth2_headers=None,
            litellm_api_key="Bearer sk-key",
            auth_resolution="m2m-client-credentials",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert headers["x-mcp-debug-auth-resolution"] == "m2m-client-credentials"

    def test_missing_server_url(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url=None,
            server_auth_type=None,
        )
        assert headers["x-mcp-debug-outbound-url"] == "(unknown)"
        assert headers["x-mcp-debug-server-auth-type"] == "(none)"

    def test_all_five_headers_present(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url=None,
            server_auth_type=None,
        )
        expected_keys = {
            "x-mcp-debug-inbound-auth",
            "x-mcp-debug-oauth2-token",
            "x-mcp-debug-auth-resolution",
            "x-mcp-debug-outbound-url",
            "x-mcp-debug-server-auth-type",
        }
        assert set(headers.keys()) == expected_keys


class TestResolveAuthResolution:
    def _make_server(self, **kwargs):
        server = MagicMock()
        server.alias = kwargs.get("alias", "test")
        server.server_name = kwargs.get("server_name", "test")
        server.has_client_credentials = kwargs.get("has_client_credentials", False)
        server.authentication_token = kwargs.get("authentication_token", None)
        server.auth_type = kwargs.get("auth_type", None)
        return server

    def test_per_request_header(self):
        server = self._make_server()
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header="Bearer xxx", mcp_server_auth_headers=None, oauth2_headers=None
        )
        assert result == "per-request-header"

    def test_server_specific_header(self):
        server = self._make_server(alias="atlas")
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header=None,
            mcp_server_auth_headers={"atlas": {"Authorization": "Bearer xxx"}},
            oauth2_headers=None,
        )
        assert result == "per-request-header"

    def test_m2m(self):
        server = self._make_server(has_client_credentials=True)
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header=None, mcp_server_auth_headers=None, oauth2_headers=None
        )
        assert result == "m2m-client-credentials"

    def test_static_token(self):
        server = self._make_server(authentication_token="static-tok")
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header=None, mcp_server_auth_headers=None, oauth2_headers=None
        )
        assert result == "static-token"

    def test_oauth2_passthrough(self):
        server = self._make_server(auth_type="oauth2")
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header=None, mcp_server_auth_headers=None,
            oauth2_headers={"Authorization": "Bearer eyJ..."},
        )
        assert result == "oauth2-passthrough"

    def test_no_auth(self):
        server = self._make_server()
        result = MCPDebug.resolve_auth_resolution(
            server, mcp_auth_header=None, mcp_server_auth_headers=None, oauth2_headers=None
        )
        assert result == "no-auth"


class TestWrapSendWithDebugHeaders:
    def test_injects_headers(self):
        captured = []

        async def mock_send(message):
            captured.append(message)

        wrapped = MCPDebug.wrap_send_with_debug_headers(
            mock_send, {"x-mcp-debug-test": "value123"}
        )

        message = {"type": "http.response.start", "status": 200, "headers": []}
        asyncio.get_event_loop().run_until_complete(wrapped(message))

        assert len(captured) == 1
        headers = dict(captured[0]["headers"])
        assert headers[b"x-mcp-debug-test"] == b"value123"

    def test_body_messages_unchanged(self):
        captured = []

        async def mock_send(message):
            captured.append(message)

        wrapped = MCPDebug.wrap_send_with_debug_headers(
            mock_send, {"x-mcp-debug-test": "value"}
        )

        body_msg = {"type": "http.response.body", "body": b"hello"}
        asyncio.get_event_loop().run_until_complete(wrapped(body_msg))

        assert captured[0] == body_msg
