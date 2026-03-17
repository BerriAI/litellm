"""
Tests for pre_mcp_call guardrail hook header mutation support.

Validates that:
1. _convert_mcp_hook_response_to_kwargs extracts extra_headers from hook response
2. pre_call_tool_check returns hook-provided extra_headers
3. call_tool flows hook headers into _call_regular_mcp_tool
4. Hook-provided headers take highest priority (merge after static_headers)
5. Backward compatibility: hooks without extra_headers continue to work
"""

import asyncio
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestConvertMcpHookResponseToKwargs:
    """Tests for ProxyLogging._convert_mcp_hook_response_to_kwargs"""

    def setup_method(self):
        self.proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

    def test_returns_original_kwargs_when_response_is_none(self):
        original = {"arguments": {"key": "val"}, "name": "tool"}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            None, original
        )
        assert result == original

    def test_returns_original_kwargs_when_response_is_empty_dict(self):
        original = {"arguments": {"key": "val"}}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs({}, original)
        assert result == original

    def test_extracts_modified_arguments(self):
        original = {"arguments": {"old": "value"}}
        response = {"modified_arguments": {"new": "value"}}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            response, original
        )
        assert result["arguments"] == {"new": "value"}

    def test_extracts_extra_headers(self):
        original = {"arguments": {"key": "val"}}
        response = {"extra_headers": {"Authorization": "Bearer signed-jwt"}}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            response, original
        )
        assert result["extra_headers"] == {"Authorization": "Bearer signed-jwt"}

    def test_extracts_both_arguments_and_headers(self):
        original = {"arguments": {"old": "value"}}
        response = {
            "modified_arguments": {"new": "value"},
            "extra_headers": {"X-Custom": "header-val"},
        }
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            response, original
        )
        assert result["arguments"] == {"new": "value"}
        assert result["extra_headers"] == {"X-Custom": "header-val"}

    def test_no_extra_headers_key_preserves_original(self):
        """Backward compat: hooks that only return modified_arguments still work."""
        original = {"arguments": {"key": "val"}}
        response = {"modified_arguments": {"key": "new_val"}}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            response, original
        )
        assert "extra_headers" not in result
        assert result["arguments"] == {"key": "new_val"}

    def test_empty_extra_headers_not_set(self):
        """Empty dict for extra_headers is falsy and should not be set."""
        original = {"arguments": {"key": "val"}}
        response = {"extra_headers": {}}
        result = self.proxy_logging._convert_mcp_hook_response_to_kwargs(
            response, original
        )
        assert "extra_headers" not in result


class TestPreCallToolCheckReturnsHeaders:
    """Tests that pre_call_tool_check returns hook-provided headers."""

    def _make_server(self, name="test_server"):
        return MCPServer(
            server_id="test-id",
            name=name,
            server_name=name,
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
        )

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_hook_has_no_headers(self):
        manager = MCPServerManager()
        server = self._make_server()

        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
        )
        proxy_logging._convert_mcp_to_llm_format = MagicMock(
            return_value={"model": "fake"}
        )
        proxy_logging.pre_call_hook = AsyncMock(
            return_value={"modified_arguments": {"key": "val"}}
        )
        proxy_logging._convert_mcp_hook_response_to_kwargs = MagicMock(
            return_value={"arguments": {"key": "val"}}
        )

        with patch.object(manager, "check_allowed_or_banned_tools", return_value=True):
            with patch.object(
                manager,
                "check_tool_permission_for_key_team",
                new_callable=AsyncMock,
            ):
                with patch.object(manager, "validate_allowed_params"):
                    result = await manager.pre_call_tool_check(
                        name="test_tool",
                        arguments={"key": "val"},
                        server_name="test_server",
                        user_api_key_auth=None,
                        proxy_logging_obj=proxy_logging,
                        server=server,
                    )

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_extra_headers_from_hook(self):
        manager = MCPServerManager()
        server = self._make_server()

        hook_headers = {"Authorization": "Bearer signed-jwt", "X-Trace-Id": "abc123"}

        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
        )
        proxy_logging._convert_mcp_to_llm_format = MagicMock(
            return_value={"model": "fake"}
        )
        proxy_logging.pre_call_hook = AsyncMock(
            return_value={"extra_headers": hook_headers}
        )
        proxy_logging._convert_mcp_hook_response_to_kwargs = MagicMock(
            return_value={"arguments": {"key": "val"}, "extra_headers": hook_headers}
        )

        with patch.object(manager, "check_allowed_or_banned_tools", return_value=True):
            with patch.object(
                manager,
                "check_tool_permission_for_key_team",
                new_callable=AsyncMock,
            ):
                with patch.object(manager, "validate_allowed_params"):
                    result = await manager.pre_call_tool_check(
                        name="test_tool",
                        arguments={"key": "val"},
                        server_name="test_server",
                        user_api_key_auth=None,
                        proxy_logging_obj=proxy_logging,
                        server=server,
                    )

        assert result["extra_headers"] == hook_headers

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_hook_returns_none(self):
        manager = MCPServerManager()
        server = self._make_server()

        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
        )
        proxy_logging._convert_mcp_to_llm_format = MagicMock(
            return_value={"model": "fake"}
        )
        proxy_logging.pre_call_hook = AsyncMock(return_value=None)

        with patch.object(manager, "check_allowed_or_banned_tools", return_value=True):
            with patch.object(
                manager,
                "check_tool_permission_for_key_team",
                new_callable=AsyncMock,
            ):
                with patch.object(manager, "validate_allowed_params"):
                    result = await manager.pre_call_tool_check(
                        name="test_tool",
                        arguments={"key": "val"},
                        server_name="test_server",
                        user_api_key_auth=None,
                        proxy_logging_obj=proxy_logging,
                        server=server,
                    )

        assert result == {}


class TestCallToolFlowsHookHeaders:
    """Tests that call_tool passes hook_extra_headers to _call_regular_mcp_tool."""

    def _make_server(self, name="test_server"):
        return MCPServer(
            server_id="test-id",
            name=name,
            server_name=name,
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
        )

    @pytest.mark.asyncio
    async def test_hook_headers_passed_to_call_regular_mcp_tool(self):
        """Verify that hook_extra_headers kwarg is forwarded."""
        manager = MCPServerManager()
        server = self._make_server()

        hook_headers = {"Authorization": "Bearer signed-jwt"}

        with patch.object(
            manager,
            "_get_mcp_server_from_tool_name",
            return_value=server,
        ):
            with patch.object(
                manager,
                "pre_call_tool_check",
                new_callable=AsyncMock,
                return_value={"extra_headers": hook_headers},
            ):
                with patch.object(
                    manager,
                    "_create_during_hook_task",
                    return_value=asyncio.create_task(asyncio.sleep(0)),
                ):
                    with patch.object(
                        manager,
                        "_call_regular_mcp_tool",
                        new_callable=AsyncMock,
                        return_value=MagicMock(),
                    ) as mock_call:
                        proxy_logging = MagicMock(spec=ProxyLogging)

                        await manager.call_tool(
                            server_name="test_server",
                            name="test_tool",
                            arguments={"key": "val"},
                            proxy_logging_obj=proxy_logging,
                        )

                        mock_call.assert_called_once()
                        call_kwargs = mock_call.call_args
                        assert call_kwargs.kwargs.get("hook_extra_headers") == hook_headers

    @pytest.mark.asyncio
    async def test_no_hook_headers_when_no_proxy_logging(self):
        """Without proxy_logging_obj, no pre_call_tool_check runs."""
        manager = MCPServerManager()
        server = self._make_server()

        with patch.object(
            manager,
            "_get_mcp_server_from_tool_name",
            return_value=server,
        ):
            with patch.object(
                manager,
                "_call_regular_mcp_tool",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_call:
                await manager.call_tool(
                    server_name="test_server",
                    name="test_tool",
                    arguments={"key": "val"},
                    proxy_logging_obj=None,
                )

                mock_call.assert_called_once()
                call_kwargs = mock_call.call_args
                assert call_kwargs.kwargs.get("hook_extra_headers") is None


class TestHookHeaderMergePriority:
    """Tests that hook-provided headers have highest priority in _call_regular_mcp_tool."""

    def _make_server(
        self,
        static_headers: Optional[Dict[str, str]] = None,
        extra_headers_config: Optional[list] = None,
    ):
        return MCPServer(
            server_id="test-id",
            name="Test Server",
            server_name="test_server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            static_headers=static_headers,
            extra_headers=extra_headers_config,
        )

    @pytest.mark.asyncio
    async def test_hook_headers_override_static_headers(self):
        """Hook headers should take precedence over static_headers."""
        manager = MCPServerManager()
        server = self._make_server(
            static_headers={"Authorization": "Bearer static-token", "X-Static": "yes"}
        )

        hook_headers = {"Authorization": "Bearer hook-signed-jwt"}

        captured_extra_headers: Dict[str, Any] = {}

        async def fake_create_mcp_client(
            server, mcp_auth_header=None, extra_headers=None, stdio_env=None
        ):
            captured_extra_headers["value"] = extra_headers
            mock_client = MagicMock()
            mock_client.call_tool = AsyncMock(return_value=MagicMock())
            return mock_client

        with patch.object(
            manager, "_create_mcp_client", side_effect=fake_create_mcp_client
        ):
            with patch.object(manager, "_build_stdio_env", return_value=None):
                try:
                    await manager._call_regular_mcp_tool(
                        mcp_server=server,
                        original_tool_name="test_tool",
                        arguments={"key": "val"},
                        tasks=[],
                        mcp_auth_header=None,
                        mcp_server_auth_headers=None,
                        oauth2_headers=None,
                        raw_headers=None,
                        proxy_logging_obj=None,
                        hook_extra_headers=hook_headers,
                    )
                except Exception:
                    pass

        headers = captured_extra_headers.get("value", {})
        assert headers["Authorization"] == "Bearer hook-signed-jwt"
        assert headers["X-Static"] == "yes"

    @pytest.mark.asyncio
    async def test_no_hook_headers_preserves_existing_behavior(self):
        """When hook_extra_headers is None, existing header logic is unchanged."""
        manager = MCPServerManager()
        server = self._make_server(
            static_headers={"X-Static": "static-value"}
        )

        captured_extra_headers: Dict[str, Any] = {}

        async def fake_create_mcp_client(
            server, mcp_auth_header=None, extra_headers=None, stdio_env=None
        ):
            captured_extra_headers["value"] = extra_headers
            mock_client = MagicMock()
            mock_client.call_tool = AsyncMock(return_value=MagicMock())
            return mock_client

        with patch.object(
            manager, "_create_mcp_client", side_effect=fake_create_mcp_client
        ):
            with patch.object(manager, "_build_stdio_env", return_value=None):
                try:
                    await manager._call_regular_mcp_tool(
                        mcp_server=server,
                        original_tool_name="test_tool",
                        arguments={"key": "val"},
                        tasks=[],
                        mcp_auth_header=None,
                        mcp_server_auth_headers=None,
                        oauth2_headers=None,
                        raw_headers=None,
                        proxy_logging_obj=None,
                        hook_extra_headers=None,
                    )
                except Exception:
                    pass

        headers = captured_extra_headers.get("value", {})
        assert headers == {"X-Static": "static-value"}

    @pytest.mark.asyncio
    async def test_hook_headers_merge_with_oauth2(self):
        """Hook headers merge on top of OAuth2 headers."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="test-id",
            name="Test Server",
            server_name="test_server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
        )

        captured_extra_headers: Dict[str, Any] = {}

        async def fake_create_mcp_client(
            server, mcp_auth_header=None, extra_headers=None, stdio_env=None
        ):
            captured_extra_headers["value"] = extra_headers
            mock_client = MagicMock()
            mock_client.call_tool = AsyncMock(return_value=MagicMock())
            return mock_client

        with patch.object(
            manager, "_create_mcp_client", side_effect=fake_create_mcp_client
        ):
            with patch.object(manager, "_build_stdio_env", return_value=None):
                try:
                    await manager._call_regular_mcp_tool(
                        mcp_server=server,
                        original_tool_name="test_tool",
                        arguments={"key": "val"},
                        tasks=[],
                        mcp_auth_header=None,
                        mcp_server_auth_headers=None,
                        oauth2_headers={
                            "Authorization": "Bearer oauth2-token",
                            "X-OAuth": "yes",
                        },
                        raw_headers=None,
                        proxy_logging_obj=None,
                        hook_extra_headers={
                            "Authorization": "Bearer hook-jwt",
                            "X-Trace-Id": "trace-123",
                        },
                    )
                except Exception:
                    pass

        headers = captured_extra_headers.get("value", {})
        assert headers["Authorization"] == "Bearer hook-jwt"
        assert headers["X-OAuth"] == "yes"
        assert headers["X-Trace-Id"] == "trace-123"


class TestUserAPIKeyAuthJwtClaims:
    """Tests that UserAPIKeyAuth correctly carries jwt_claims."""

    def test_jwt_claims_field_defaults_to_none(self):
        auth = UserAPIKeyAuth(api_key="test-key")
        assert auth.jwt_claims is None

    def test_jwt_claims_field_accepts_dict(self):
        claims = {"sub": "user-123", "iss": "litellm", "exp": 9999999999}
        auth = UserAPIKeyAuth(api_key="test-key", jwt_claims=claims)
        assert auth.jwt_claims == claims
        assert auth.jwt_claims["sub"] == "user-123"

    def test_jwt_claims_backward_compatible_without_field(self):
        """Existing code that doesn't pass jwt_claims should still work."""
        auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="user-1",
            team_id="team-1",
        )
        assert auth.jwt_claims is None
        assert auth.user_id == "user-1"
