import importlib
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../../../")


import httpx
from mcp import ReadResourceResult, Resource
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ResourceTemplate,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _deserialize_json_dict,
    _deserialize_json_list,
    _normalize_mcp_server_cost_info,
    _should_strip_caller_authorization,
    _without_authorization,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPApprovalStatus,
    MCPEnvVar,
    MCPEnvVarScope,
    MCPTransport,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPOAuthMetadata, MCPServer


def _reload_mcp_manager_module():
    utils_module = sys.modules["litellm.proxy._experimental.mcp_server.utils"]
    manager_module = sys.modules["litellm.proxy._experimental.mcp_server.mcp_server_manager"]
    importlib.reload(utils_module)
    reloaded = importlib.reload(manager_module)
    # After reload, server.py still holds a stale reference to the old
    # global_mcp_server_manager. Update it so tests that exercise server.py
    # functions (e.g. _get_tools_from_mcp_servers) use the fresh instance.
    server_module = sys.modules.get("litellm.proxy._experimental.mcp_server.server")
    if server_module is not None and hasattr(server_module, "global_mcp_server_manager"):
        server_module.global_mcp_server_manager = reloaded.global_mcp_server_manager
    return reloaded


class TestMCPServerManager:
    """Test MCP Server Manager stdio functionality"""

    def test_deserialize_json_dict(self):
        """Test environment dictionary deserialization"""
        # Test JSON string
        env_json = '{"PATH": "/usr/bin", "DEBUG": "1"}'
        result = _deserialize_json_dict(env_json)
        assert result == {"PATH": "/usr/bin", "DEBUG": "1"}

        # Test already dict
        env_dict = {"PATH": "/usr/bin", "DEBUG": "1"}
        result = _deserialize_json_dict(env_dict)
        assert result == {"PATH": "/usr/bin", "DEBUG": "1"}

        # Test invalid JSON
        invalid_json = '{"PATH": "/usr/bin", "DEBUG": 1'
        result = _deserialize_json_dict(invalid_json)
        assert result is None

    async def test_add_update_server_stdio(self):
        """Test adding stdio MCP server"""
        manager = MCPServerManager()

        stdio_server = LiteLLM_MCPServerTable(
            server_id="stdio-server-1",
            alias="test_stdio_server",
            description="Test stdio server",
            url=None,
            transport=MCPTransport.stdio,
            command="python",
            args=["-m", "server"],
            env={"DEBUG": "1", "TEST": "1"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await manager.add_server(stdio_server)

        # Verify server was added
        assert "stdio-server-1" in manager.registry
        added_server = manager.registry["stdio-server-1"]

        assert added_server.server_id == "stdio-server-1"
        assert added_server.name == "test_stdio_server"
        assert added_server.transport == MCPTransport.stdio
        assert added_server.command == "python"
        assert added_server.args == ["-m", "server"]
        assert added_server.env == {"DEBUG": "1", "TEST": "1"}

    async def test_create_mcp_client_stdio(self):
        """Test creating MCP client for stdio transport"""
        manager = MCPServerManager()

        stdio_server = MCPServer(
            server_id="stdio-server-2",
            name="test_stdio_server",
            url=None,
            transport=MCPTransport.stdio,
            command="node",
            args=["server.js"],
            env={"NODE_ENV": "test"},
        )

        client = await manager._create_mcp_client(stdio_server)

        assert client.transport_type == MCPTransport.stdio
        assert client.stdio_config is not None
        assert client.stdio_config["command"] == "node"
        assert client.stdio_config["args"] == ["server.js"]
        # NPM_CONFIG_CACHE is injected automatically for container compatibility
        from litellm.constants import MCP_NPM_CACHE_DIR

        assert client.stdio_config["env"]["NODE_ENV"] == "test"
        assert client.stdio_config["env"]["NPM_CONFIG_CACHE"] == MCP_NPM_CACHE_DIR

    @pytest.mark.asyncio
    async def test_caller_auth_header_cannot_bypass_v2_for_authorization_code(self):
        """A caller-supplied per-request override must not substitute the stored authorization_code
        token: _create_mcp_client keeps the v2 spec and resolves through the injected provider
        rather than deferring to the v1 caller-override path."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
            Ok,
        )
        from litellm.types.mcp import MCPAuth

        calls = []

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                calls.append((subject.subject_id, server.server_id))
                return Ok(StaticHeaderAuth("stored-token"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = MCPServer(
            server_id="authz-srv",
            name="authz",
            url="https://upstream.example/mcp",
            transport=MCPTransport.sse,
            auth_type=MCPAuth.oauth2,  # oauth2 + no client creds + not delegate -> authorization_code
        )

        client = await manager._create_mcp_client(server, mcp_auth_header="Bearer caller-supplied-token")

        # the v2 resolver ran (the caller override did NOT defer to v1); the stored token wins
        assert calls == [("", "authz-srv")]
        assert client is not None

    async def test_create_mcp_client_stdio_injects_npm_config_cache(self):
        """Test that _create_mcp_client injects NPM_CONFIG_CACHE when not already set,
        and preserves user-provided NPM_CONFIG_CACHE when present."""
        from litellm.constants import MCP_NPM_CACHE_DIR

        manager = MCPServerManager()

        # Case 1: NPM_CONFIG_CACHE not set -> should be injected
        server_no_cache = MCPServer(
            server_id="stdio-npm-1",
            name="test_npm_server",
            url=None,
            transport=MCPTransport.stdio,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            env={},
        )
        client = await manager._create_mcp_client(server_no_cache)
        assert client.stdio_config["env"]["NPM_CONFIG_CACHE"] == MCP_NPM_CACHE_DIR

        # Case 2: NPM_CONFIG_CACHE already set -> should NOT be overwritten
        server_with_cache = MCPServer(
            server_id="stdio-npm-2",
            name="test_npm_server_custom",
            url=None,
            transport=MCPTransport.stdio,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            env={"NPM_CONFIG_CACHE": "/custom/cache"},
        )
        client2 = await manager._create_mcp_client(server_with_cache)
        assert client2.stdio_config["env"]["NPM_CONFIG_CACHE"] == "/custom/cache"

    def test_build_stdio_env_only_accepts_x_prefixed_placeholders(self):
        """Ensure only ${X-*} placeholders are substituted from headers."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="stdio-server-env",
            name="stdio_env",
            transport=MCPTransport.stdio,
            command="node",
            args=["server.js"],
            env={
                "PASSTHROUGH": "${X-Test-Header}",
                "STATIC": "value",
                "IGNORED": "${Not-Allowed}",
            },
        )

        env = manager._build_stdio_env(
            server,
            raw_headers={
                "x-test-header": "resolved-value",
                "x-not-used": "other",
            },
        )

        assert env == {
            "PASSTHROUGH": "resolved-value",
            "STATIC": "value",
            "IGNORED": "${Not-Allowed}",
        }

    def test_build_stdio_env_missing_header_skips_entry(self):
        """Ensure missing headers drop the placeholder from the resolved env."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="stdio-server-env-miss",
            name="stdio_env_miss",
            transport=MCPTransport.stdio,
            command="node",
            args=["server.js"],
            env={"EXPECTED": "${X-Missing}"},
        )

        env = manager._build_stdio_env(server, raw_headers={})

        # When the header isn't provided, the key is omitted entirely
        assert env == {}

    @pytest.mark.asyncio
    async def test_load_servers_from_config_warns_on_invalid_alias(self, caplog):
        """Invalid aliases from config should emit warnings during load."""

        manager = MCPServerManager()
        config = {
            "validserver": {
                "alias": "bad/name",
                "url": "https://example.com",
                "transport": MCPTransport.http,
            }
        }

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            await manager.load_servers_from_config(config)

        assert any("invalid alias 'bad/name'" in message for message in caplog.messages)

    @pytest.mark.asyncio
    async def test_load_servers_from_config_accepts_valid_alias(self, caplog):
        """Valid aliases should be accepted and populate the registry."""

        manager = MCPServerManager()
        config = {
            "validserver": {
                "alias": "friendly_alias",
                "url": "https://example.com",
                "transport": MCPTransport.http,
            }
        }

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            await manager.load_servers_from_config(config)

        # No warnings logged for the valid alias
        assert all("invalid alias" not in message for message in caplog.messages)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.alias == "friendly_alias"
        assert server.server_name == "validserver"

    def _oauth2_config(self, **overrides):
        base = {
            "url": "https://example.com/mcp",
            "transport": MCPTransport.http,
            "auth_type": MCPAuth.oauth2,
            "token_url": "https://idp.example.com/token",
            "client_id": "cid",
            "client_secret": "csec",
        }
        base.update(overrides)
        return {"m2mserver": base}

    @pytest.mark.asyncio
    async def test_load_servers_from_config_requires_oauth2_flow(self):
        """auth_type oauth2 without an explicit oauth2_flow is a config error: the
        credential shape is ambiguous (a DCR interactive server looks identical to M2M),
        so the config must assert the flow instead of the proxy guessing it."""

        manager = MCPServerManager()

        with (
            patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)),
            pytest.raises(ValueError) as exc_info,
        ):
            await manager.load_servers_from_config(self._oauth2_config())

        assert "oauth2_flow: client_credentials" in str(exc_info.value)
        assert "oauth2_flow: authorization_code" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_servers_from_config_rejects_unknown_oauth2_flow(self):
        manager = MCPServerManager()

        with (
            patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)),
            pytest.raises(ValueError) as exc_info,
        ):
            await manager.load_servers_from_config(self._oauth2_config(oauth2_flow="m2m"))

        assert "got 'm2m'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_servers_from_config_accepts_explicit_client_credentials(self):
        manager = MCPServerManager()

        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            await manager.load_servers_from_config(self._oauth2_config(oauth2_flow="client_credentials"))

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.oauth2_flow == "client_credentials"
        assert server.has_client_credentials is True

    @pytest.mark.asyncio
    async def test_load_servers_from_config_accepts_explicit_authorization_code(self):
        manager = MCPServerManager()

        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            await manager.load_servers_from_config(self._oauth2_config(oauth2_flow="authorization_code"))

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.oauth2_flow == "authorization_code"
        assert server.needs_user_oauth_token is True

    @pytest.mark.asyncio
    async def test_load_servers_from_config_non_oauth2_needs_no_flow(self):
        manager = MCPServerManager()
        config = {
            "apiserver": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.api_key,
                "auth_value": "sk-upstream",
            }
        }

        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.oauth2_flow is None

    def _client_forwarded_config(self, auth_type, **overrides):
        base = {
            "url": "https://example.com/mcp",
            "transport": MCPTransport.http,
            "auth_type": auth_type,
        }
        base.update(overrides)
        return {"bridgeserver": base}

    @pytest.mark.asyncio
    async def test_load_servers_from_config_rejects_dcr_bridge_on_gateway_managed_auth_type(self):
        manager = MCPServerManager()

        with (
            patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)),
            pytest.raises(ValueError) as exc_info,
        ):
            await manager.load_servers_from_config(
                self._oauth2_config(oauth2_flow="authorization_code", dcr_bridge=True)
            )

        assert "dcr_bridge is only supported" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_servers_from_config_rejects_non_boolean_dcr_bridge(self):
        manager = MCPServerManager()

        with (
            patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)),
            pytest.raises(ValueError) as exc_info,
        ):
            await manager.load_servers_from_config(
                self._client_forwarded_config(MCPAuth.true_passthrough, dcr_bridge="yes")
            )

        assert "must be a boolean" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
    async def test_load_servers_from_config_accepts_dcr_bridge_on_client_forwarded_modes(self, auth_type):
        manager = MCPServerManager()

        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            await manager.load_servers_from_config(self._client_forwarded_config(auth_type, dcr_bridge=True))

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.dcr_bridge is True
        assert server.is_dcr_bridge is True

    @pytest.mark.asyncio
    async def test_load_servers_from_config_dcr_bridge_defaults_off(self):
        manager = MCPServerManager()

        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            await manager.load_servers_from_config(self._client_forwarded_config(MCPAuth.true_passthrough))

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.dcr_bridge is None
        assert server.is_dcr_bridge is False

    @pytest.mark.asyncio
    async def test_load_servers_from_config_coerces_cost_string_to_float(self):
        """YAML 1.1 parses `7e-05` as a string; ingest must coerce it to float."""
        manager = MCPServerManager()
        config = {
            "google_maps": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "mcp_info": {
                    "mcp_server_cost_info": {
                        "default_cost_per_query": "7e-05",
                        "tool_name_to_cost_per_query": {"geocode": "1e-3"},
                    }
                },
            }
        }

        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        cost_info = server.mcp_info["mcp_server_cost_info"]
        assert cost_info["default_cost_per_query"] == 7e-05
        assert isinstance(cost_info["default_cost_per_query"], float)
        assert cost_info["tool_name_to_cost_per_query"]["geocode"] == 1e-3
        assert isinstance(cost_info["tool_name_to_cost_per_query"]["geocode"], float)

    @pytest.mark.asyncio
    async def test_load_servers_from_config_sets_token_endpoint_auth_method(self):
        """token_endpoint_auth_method from config is carried onto the MCPServer (LIT-4091)."""
        manager = MCPServerManager()
        config = {
            "basic_provider": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "token_endpoint_auth_method": "client_secret_basic",
            },
            "default_provider": {
                "url": "https://example.com/mcp2",
                "transport": MCPTransport.http,
            },
        }

        await manager.load_servers_from_config(config)

        by_name = {s.server_name: s for s in manager.config_mcp_servers.values()}
        assert by_name["basic_provider"].token_endpoint_auth_method == "client_secret_basic"
        assert by_name["default_provider"].token_endpoint_auth_method is None

    def test_normalize_mcp_server_cost_info_preserves_float_values(self):
        mcp_info = {
            "server_name": "maps",
            "mcp_server_cost_info": {
                "default_cost_per_query": 0.01,
                "tool_name_to_cost_per_query": {"search": 0.05},
            },
        }

        _normalize_mcp_server_cost_info(mcp_info)

        cost_info = mcp_info["mcp_server_cost_info"]
        assert cost_info["default_cost_per_query"] == 0.01
        assert cost_info["tool_name_to_cost_per_query"] == {"search": 0.05}

    def test_normalize_mcp_server_cost_info_drops_non_numeric_values(self):
        mcp_info = {
            "server_name": "maps",
            "mcp_server_cost_info": {
                "default_cost_per_query": "not-a-number",
                "tool_name_to_cost_per_query": {"search": "free", "geocode": "2e-4"},
            },
        }

        _normalize_mcp_server_cost_info(mcp_info)

        cost_info = mcp_info["mcp_server_cost_info"]
        assert "default_cost_per_query" not in cost_info
        assert cost_info["tool_name_to_cost_per_query"] == {"geocode": 2e-4}

    def test_normalize_mcp_server_cost_info_leaves_missing_cost_info_alone(self):
        mcp_info = {"server_name": "maps"}

        _normalize_mcp_server_cost_info(mcp_info)

        assert "mcp_server_cost_info" not in mcp_info

    def test_warns_when_custom_separator_invalid(self, monkeypatch, caplog):
        """Invalid MCP_TOOL_PREFIX_SEPARATOR values should log a warning."""

        original_value = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR")
        monkeypatch.setenv("MCP_TOOL_PREFIX_SEPARATOR", "/")

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            _reload_mcp_manager_module()

        assert any("violates SEP-986" in message for message in caplog.messages)

        # Restore original setting and ensure warning disappears
        if original_value is None:
            monkeypatch.delenv("MCP_TOOL_PREFIX_SEPARATOR", raising=False)
        else:
            monkeypatch.setenv("MCP_TOOL_PREFIX_SEPARATOR", original_value)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            _reload_mcp_manager_module()

        assert all("violates SEP-986" not in message for message in caplog.messages)

    def test_accepts_valid_custom_separator(self, monkeypatch, caplog):
        """Valid separators should not emit warnings during module import."""

        original_value = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR")
        monkeypatch.setenv("MCP_TOOL_PREFIX_SEPARATOR", "_")

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            _reload_mcp_manager_module()

        assert all("violates SEP-986" not in message for message in caplog.messages)

        if original_value is None:
            monkeypatch.delenv("MCP_TOOL_PREFIX_SEPARATOR", raising=False)
        else:
            monkeypatch.setenv("MCP_TOOL_PREFIX_SEPARATOR", original_value)

        _reload_mcp_manager_module()

    @pytest.mark.asyncio
    async def test_list_tools_with_server_specific_auth_headers(self):
        """Test list_tools method with server-specific auth headers"""
        manager = MCPServerManager()

        # Mock servers
        server1 = MagicMock()
        server1.name = "github"
        server1.alias = "github"
        server1.server_name = "github"

        server2 = MagicMock()
        server2.name = "zapier"
        server2.alias = "zapier"
        server2.server_name = "zapier"

        # Mock get_allowed_mcp_servers to return our test servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github", "zapier"])
        manager.get_mcp_server_by_id = MagicMock(side_effect=lambda x: server1 if x == "github" else server2)

        # Mock _get_tools_from_server to return different results
        async def mock_get_tools_from_server(
            server,
            mcp_auth_header=None,
            **kwargs,
        ):
            if server.name == "github":
                tool1 = MagicMock()
                tool1.name = "github_tool_1"
                tool2 = MagicMock()
                tool2.name = "github_tool_2"
                return [tool1, tool2]
            else:
                tool1 = MagicMock()
                tool1.name = "zapier_tool_1"
                return [tool1]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Test with server-specific auth headers
        mcp_server_auth_headers = {
            "github": "Bearer github-token",
            "zapier": "zapier-api-key",
        }

        result = await manager.list_tools(mcp_server_auth_headers=mcp_server_auth_headers)

        # Verify that both servers were called with their specific auth headers
        assert len(result) == 3  # 2 from github + 1 from zapier

        # Verify the tools have the expected names
        tool_names = [tool.name for tool in result]
        assert "github_tool_1" in tool_names
        assert "github_tool_2" in tool_names
        assert "zapier_tool_1" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_fallback_to_legacy_auth_header(self):
        """Test that list_tools falls back to legacy auth header when server-specific not available"""
        manager = MCPServerManager()

        # Mock server
        server = MagicMock()
        server.name = "github"
        server.alias = "github"
        server.server_name = "github"

        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(
            server,
            mcp_auth_header=None,
            **kwargs,
        ):
            assert mcp_auth_header == "legacy-token"  # Should use legacy header
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Test with only legacy auth header (no server-specific headers)
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={},  # Empty server-specific headers
        )

        assert len(result) == 1
        assert result[0].name == "github_tool_1"

    @pytest.mark.asyncio
    async def test_list_tools_prioritizes_server_specific_over_legacy(self):
        """Test that server-specific auth headers take priority over legacy header"""
        manager = MCPServerManager()

        # Mock server
        server = MagicMock()
        server.name = "github"
        server.alias = "github"
        server.server_name = "github"

        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(
            server,
            mcp_auth_header=None,
            **kwargs,
        ):
            assert mcp_auth_header == "server-specific-token"  # Should use server-specific header
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Test with both legacy and server-specific headers
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={"github": "server-specific-token"},
        )

        assert len(result) == 1
        assert result[0].name == "github_tool_1"

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_case_insensitive_extra_headers(self):
        """_call_regular_mcp_tool should forward headers regardless of original casing."""

        manager = MCPServerManager()
        server = MCPServer(
            server_id="server-case-call",
            name="case-call-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.authorization,
            extra_headers=["Authorization"],
        )

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured_extra_headers = None

        async def capture_create_mcp_client(
            server, mcp_auth_header, extra_headers, stdio_env, **kwargs
        ):  # pragma: no cover - helper
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        result = await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={"authorization": "Bearer token"},
            proxy_logging_obj=None,
        )

        assert captured_extra_headers == {"Authorization": "Bearer token"}
        assert isinstance(result, CallToolResult)

    async def _capture_list_subject_token(self, server, oauth2_headers, raw_headers=None):
        """Run _get_tools_from_server and return the subject_token it threaded to _create_mcp_client."""
        manager = MCPServerManager()
        captured = {}

        async def capture_create_mcp_client(
            server, mcp_auth_header, extra_headers, stdio_env, subject_token=None, **kwargs
        ):  # pragma: no cover - helper
            captured["subject_token"] = subject_token
            return AsyncMock()

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)
        manager._fetch_tools_with_timeout = AsyncMock(return_value=[])
        await manager._get_tools_from_server(server=server, oauth2_headers=oauth2_headers, raw_headers=raw_headers)
        return captured["subject_token"]

    @pytest.mark.asyncio
    async def test_list_threads_subject_token_for_token_exchange(self):
        """tools/list discovery must hand the caller's bearer to the resolver for OBO servers."""
        server = MCPServer(
            server_id="te-list",
            name="te-list-server",
            url="https://up.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
        subject_token = await self._capture_list_subject_token(
            server, oauth2_headers={"Authorization": "Bearer subj-jwt"}
        )
        assert subject_token == "subj-jwt"

    @pytest.mark.asyncio
    async def test_list_does_not_thread_subject_token_for_non_token_exchange(self):
        """A non-OBO server must not get the caller's bearer threaded (no leak across modes)."""
        server = MCPServer(
            server_id="none-list",
            name="none-list-server",
            url="https://up.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
        )
        subject_token = await self._capture_list_subject_token(
            server, oauth2_headers={"Authorization": "Bearer subj-jwt"}
        )
        assert subject_token is None

    @pytest.mark.asyncio
    async def test_list_subject_token_none_without_oauth2_headers(self):
        """Background/registry refresh (no oauth2 headers) lists with no subject token, as before."""
        server = MCPServer(
            server_id="te-list-bg",
            name="te-list-bg-server",
            url="https://up.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
        subject_token = await self._capture_list_subject_token(server, oauth2_headers=None)
        assert subject_token is None

    @pytest.mark.asyncio
    async def test_list_surfaces_resolver_401_as_upstream_auth_error(self):
        """A v2 resolver auth challenge (HTTPException 401) raised while building the client must
        surface as MCPUpstreamAuthError with its WWW-Authenticate preserved, so single-server routes
        challenge the client instead of the old behavior of masking it to an empty tool list."""
        server = MCPServer(
            server_id="te-401",
            name="te-401-server",
            url="https://up.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
        manager = MCPServerManager()
        challenge = (
            'Bearer resource_metadata="/.well-known/oauth-protected-resource/mcp/te-401-server", error="invalid_token"'
        )
        manager._create_mcp_client = AsyncMock(
            side_effect=HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": challenge})
        )
        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await manager._get_tools_from_server(server=server, oauth2_headers={"Authorization": "Bearer subj-jwt"})
        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate == challenge

    @pytest.mark.asyncio
    async def test_list_absorbs_non_auth_httpexception(self):
        """A non-auth HTTP error (e.g. 412 no endpoint, 503 IdP down) must stay absorbed to [] so one
        misconfigured/unavailable server does not blank the whole aggregate listing."""
        server = MCPServer(
            server_id="te-412",
            name="te-412-server",
            url="https://up.example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
        manager = MCPServerManager()
        manager._create_mcp_client = AsyncMock(
            side_effect=HTTPException(status_code=412, detail="token exchange endpoint is not configured")
        )
        result = await manager._get_tools_from_server(
            server=server, oauth2_headers={"Authorization": "Bearer subj-jwt"}
        )
        assert result == []

    def _upstream_status_error(self, status_code: int, www_authenticate: Optional[str] = None) -> httpx.HTTPStatusError:
        """Build an httpx.HTTPStatusError shaped like the one the MCP SDK surfaces for an upstream
        HTTP failure, so _extract_upstream_auth_failure can read status_code and WWW-Authenticate."""
        request = httpx.Request("POST", "https://up.example.com/mcp")
        headers = {"www-authenticate": www_authenticate} if www_authenticate else {}
        response = httpx.Response(status_code, headers=headers, request=request)
        return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)

    def _passthrough_call_server(self, auth_type, server_id: str = "pt-call") -> "MCPServer":
        return MCPServer(
            server_id=server_id,
            name=f"{server_id}-server",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=auth_type,
        )

    async def _run_call_regular(self, manager, server):
        return await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={"authorization": "Bearer caller-upstream-token"},
            proxy_logging_obj=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
    async def test_call_relays_upstream_401_for_client_forwarded_modes(self, auth_type):
        """A client-forwarded pass-through/delegate call must relay an upstream 401 (expired/invalid
        token) as MCPUpstreamAuthError with the upstream WWW-Authenticate preserved, so single-server
        REST routes challenge the caller instead of masking it as a generic isError. Only 401 is a
        re-auth signal; the relay opts into raise_on_error so the transport failure surfaces."""
        server = self._passthrough_call_server(auth_type)
        challenge = f'Bearer resource_metadata="/.well-known/oauth-protected-resource/mcp/{server.name}"'
        manager = MCPServerManager()
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(side_effect=self._upstream_status_error(401, challenge))
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await self._run_call_regular(manager, server)

        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate == challenge
        assert mock_client.call_tool.call_args.kwargs.get("raise_on_error") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_error", [False, True])
    async def test_call_passthrough_returns_tool_result_unchanged(self, is_error):
        """The relay only re-raises transport failures. A tool that RETURNS a result (a success, or a
        tool-level isError, neither of which raises) on a pass-through call must be returned verbatim,
        never wrapped as MCPUpstreamAuthError or replaced by error_tool_result."""
        server = self._passthrough_call_server(MCPAuth.true_passthrough, server_id=f"pt-ok-{is_error}")
        manager = MCPServerManager()
        expected = CallToolResult(content=[], isError=is_error)
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=expected)
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        result = await self._run_call_regular(manager, server)

        assert result is expected
        assert mock_client.call_tool.call_args.kwargs.get("raise_on_error") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [403, 503])
    async def test_call_passthrough_non_reauth_failure_stays_iserror(self, status_code):
        """Only an upstream 401 is a re-auth signal. A 403 (authenticated but forbidden; re-auth
        won't help) and a genuine non-auth failure (e.g. 503) both keep the default isError
        degradation and stay a visible warning, mirroring the list path, rather than being relayed as
        a re-auth challenge."""
        from litellm.experimental_mcp_client.client import MCPClient

        server = self._passthrough_call_server(MCPAuth.true_passthrough, server_id=f"pt-{status_code}")
        manager = MCPServerManager()
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(side_effect=self._upstream_status_error(status_code))
        mock_client.error_tool_result = MCPClient.error_tool_result
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        import litellm.proxy._experimental.mcp_server.mcp_server_manager as _mgr_mod

        with patch.object(_mgr_mod, "verbose_logger") as mock_log:
            result = await self._run_call_regular(manager, server)

        assert result.isError is True
        # A genuine non-auth failure keeps operator visibility at warning level, since call_tool's
        # raise_on_error demoted the client-layer error log to debug.
        assert mock_log.warning.called

    @pytest.mark.asyncio
    async def test_call_non_passthrough_does_not_opt_into_raise_on_error(self):
        """Non-client-forwarded auth types keep the default call_tool masking (raise_on_error stays
        off), so this relay is scoped to the pass-through modes and cannot regress api_key/OBO calls."""
        server = MCPServer(
            server_id="ak-call",
            name="ak-call-server",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.api_key,
            authentication_token="static-key",
        )
        manager = MCPServerManager()
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        result = await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
        )

        assert result.isError is False
        assert mock_client.call_tool.call_args.kwargs.get("raise_on_error") is not True

    def _token_exchange_server(self, server_id: str) -> "MCPServer":
        return MCPServer(
            server_id=server_id,
            name=f"{server_id}-server",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )

    @pytest.mark.asyncio
    async def test_entra_obo_profile_survives_db_credentials_round_trip(self):
        # A credentials blob from the management API / DB carries token_exchange_profile; the DB build
        # must reconstruct it onto the MCPServer, and the v2 adapter must map it onto the resolver
        # config. Without threading it through, an entra_obo server persisted via the API silently
        # falls back to rfc8693 and posts the wrong grant to the IdP.
        from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import to_server_spec
        from litellm.proxy._experimental.mcp_server.outbound_credentials.types import TokenExchangeConfig

        manager = MCPServerManager()
        row = LiteLLM_MCPServerTable(
            server_id="entra-db-1",
            alias="entra_db",
            description="entra obo from db",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={
                "client_id": "cid",
                "client_secret": "csec",
                "token_exchange_endpoint": "https://login.microsoftonline.com/tid/oauth2/v2.0/token",
                "scopes": ["api://target/.default"],
                "token_exchange_profile": "entra_obo",
            },
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        built = await manager.build_mcp_server_from_table(row, credentials_are_encrypted=False)
        assert built.token_exchange_profile == "entra_obo"

        spec = to_server_spec(built)
        assert spec is not None and isinstance(spec.config, TokenExchangeConfig)
        assert spec.config.profile == "entra_obo"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("auth_type", [MCPAuth.true_passthrough, MCPAuth.oauth_delegate])
    async def test_build_from_table_discovers_upstream_oauth_for_client_forwarded_modes(self, auth_type):
        """The gateway's relayed authorize flow (used by the browser-only Authorize) needs the
        upstream's authorization_url on the registry entry, and these rows never persist one, so
        the DB build must discover it the same way oauth2 rows do."""
        from types import SimpleNamespace

        manager = MCPServerManager()
        row = LiteLLM_MCPServerTable(
            server_id="cf-db-1",
            alias="cf_db",
            description="client-forwarded from db",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=auth_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        metadata = SimpleNamespace(
            authorization_url="https://idp.example.com/authorize",
            token_url="https://idp.example.com/token",
            registration_url="https://idp.example.com/register",
            scopes=None,
        )
        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=metadata)) as mock_discovery:
            built = await manager.build_mcp_server_from_table(row, credentials_are_encrypted=False)

        mock_discovery.assert_awaited_once()
        assert built.authorization_url == "https://idp.example.com/authorize"
        assert built.token_url == "https://idp.example.com/token"

    async def _capture_subject_token(self, call) -> Optional[str]:
        """Run a manager method (via ``call(manager)``) and return the subject_token it threaded
        into ``_create_mcp_client``."""
        manager = MCPServerManager()
        captured: Dict[str, Any] = {}

        async def capture_create_mcp_client(
            server, mcp_auth_header, extra_headers, stdio_env, subject_token=None, **kwargs
        ):  # pragma: no cover - helper
            captured["subject_token"] = subject_token
            return AsyncMock()

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)
        await call(manager)
        return captured.get("subject_token")

    @pytest.mark.asyncio
    async def test_prompts_thread_subject_token_for_token_exchange(self):
        """prompts/list on an OBO server must exchange the caller's bearer, not connect with none."""
        server = self._token_exchange_server("te-prompts")
        st = await self._capture_subject_token(
            lambda m: m.get_prompts_from_server(server=server, raw_headers={"authorization": "Bearer subj-jwt"})
        )
        assert st == "subj-jwt"

    @pytest.mark.asyncio
    async def test_resources_thread_subject_token_for_token_exchange(self):
        """resources/list on an OBO server must exchange the caller's bearer."""
        server = self._token_exchange_server("te-resources")
        st = await self._capture_subject_token(
            lambda m: m.get_resources_from_server(server=server, raw_headers={"authorization": "Bearer subj-jwt"})
        )
        assert st == "subj-jwt"

    @pytest.mark.asyncio
    async def test_read_resource_threads_subject_token_for_token_exchange(self):
        """resources/read on an OBO server must exchange the caller's bearer."""
        server = self._token_exchange_server("te-read")
        st = await self._capture_subject_token(
            lambda m: m.read_resource_from_server(
                server=server,
                url="https://up.example.com/r",
                raw_headers={"authorization": "Bearer subj-jwt"},
            )
        )
        assert st == "subj-jwt"

    @pytest.mark.asyncio
    async def test_prompts_no_subject_token_for_non_token_exchange(self):
        """A non-OBO server must not get the caller's bearer threaded (no cross-mode leak)."""
        server = MCPServer(
            server_id="none-prompts",
            name="none-prompts-server",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
        )
        st = await self._capture_subject_token(
            lambda m: m.get_prompts_from_server(server=server, raw_headers={"authorization": "Bearer subj-jwt"})
        )
        assert st is None

    @pytest.mark.asyncio
    async def test_caller_header_cannot_bypass_v2_for_token_exchange(self):
        """A caller-supplied per-server header (x-mcp-*) must NOT disable the OBO exchange:
        _create_mcp_client keeps the v2 spec and runs the resolver (which exchanges the subject),
        rather than deferring to v1 and forwarding the caller's header verbatim upstream."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok

        exchanged_subjects = []

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                exchanged_subjects.append(subject.inbound_token.get_secret_value() if subject.inbound_token else None)
                return Ok(StaticHeaderAuth("Bearer MINTED", header_name="Authorization"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = self._token_exchange_server("te-bypass")

        client = await manager._create_mcp_client(
            server,
            mcp_auth_header="Bearer x-mcp-caller-header",
            subject_token="subj-jwt",
        )

        # The resolver ran and exchanged the subject despite the per-server header; not bypassed.
        assert exchanged_subjects == ["subj-jwt"]
        assert client is not None

    @pytest.mark.asyncio
    async def test_injected_authorization_does_not_shadow_obo_minted_token(self):
        """A guardrail/static Authorization (e.g. MCPJWTSigner) must NOT shadow the exchanged OBO
        token. The resolver-owned credential is authoritative: the conflicting header is dropped and
        the minted token is what reaches the upstream, not the injected JWT."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                return Ok(StaticHeaderAuth("Bearer MINTED", header_name="Authorization"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = self._token_exchange_server("te-shadow")

        client = await manager._create_mcp_client(
            server,
            extra_headers={"Authorization": "Bearer signer-jwt"},  # simulate the JWT signer
            subject_token="subj-jwt",
        )

        # minted token wins (resolved_auth kept), signer's header dropped from extra_headers
        assert client._resolved_auth is not None
        assert "authorization" not in {k.lower() for k in (client.extra_headers or {})}

    @pytest.mark.asyncio
    async def test_preflight_token_exchange_challenges_on_rejected_subject(self):
        """A subject the IdP rejects must raise the RFC 9728 401 challenge from the preflight, so a
        single-server route fails observably at the transport edge instead of the old behavior of
        the session opening and list_tools masking the failed exchange as an empty tool list."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
        from litellm.proxy._experimental.mcp_server.outbound_credentials.types import CredError

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                return Error(CredError.of_unauthorized("subject token rejected by the IdP"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = self._token_exchange_server("te-preflight-401")

        with pytest.raises(HTTPException) as exc_info:
            await manager.preflight_token_exchange(
                server=server,
                oauth2_headers={"Authorization": "Bearer rejected-subject"},
                user_api_key_auth=None,
            )
        assert exc_info.value.status_code == 401
        headers = exc_info.value.headers or {}
        www_authenticate = headers.get("WWW-Authenticate") or headers.get("www-authenticate") or ""
        assert "resource_metadata" in www_authenticate

    @pytest.mark.asyncio
    async def test_preflight_token_exchange_maps_gateway_fault_to_public_status(self):
        """A gateway-fault CredError (e.g. invalid_client) must surface its public status (500)
        from the preflight, not the OBO 401 challenge and not an empty-success session."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
        from litellm.proxy._experimental.mcp_server.outbound_credentials.types import CredError

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                return Error(CredError.of_misconfigured("token exchange configuration error"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = self._token_exchange_server("te-preflight-500")

        with pytest.raises(HTTPException) as exc_info:
            await manager.preflight_token_exchange(
                server=server,
                oauth2_headers={"Authorization": "Bearer subj"},
                user_api_key_auth=None,
            )
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_preflight_token_exchange_noop_on_success_and_without_subject(self):
        """A successful exchange returns without raising, and a request with no bearer never
        reaches the resolver (the no-subject case is the existing preemptive challenge's job)."""
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok

        resolved = []

        class _FakeProvider:
            async def resolve_credentials(self, subject, server):
                resolved.append(subject.inbound_token.get_secret_value() if subject.inbound_token else None)
                return Ok(StaticHeaderAuth("Bearer MINTED", header_name="Authorization"))

        manager = MCPServerManager(cred_provider=_FakeProvider())
        server = self._token_exchange_server("te-preflight-ok")

        assert (
            await manager.preflight_token_exchange(
                server=server,
                oauth2_headers={"Authorization": "Bearer good-subject"},
                user_api_key_auth=None,
            )
            is None
        )
        assert resolved == ["good-subject"]

        await manager.preflight_token_exchange(server=server, oauth2_headers=None, user_api_key_auth=None)
        assert resolved == ["good-subject"]

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_passthrough_strips_authorization_when_admission_consumed_litellm_key(
        self,
    ):
        """OAuth pass-through must not forward the caller's Authorization to upstream
        when LiteLLM admission consumed the bearer as its API key — otherwise the
        LiteLLM key the caller used for admission would leak upstream."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="server-passthrough-call",
            name="passthrough-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            extra_headers=["Authorization", "x-request-id"],
            oauth_passthrough=True,
        )

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured_extra_headers = None

        async def capture_create_mcp_client(
            server,
            mcp_auth_header,
            extra_headers,
            stdio_env,
            subject_token=None,
            **kwargs,
        ):  # pragma: no cover - helper
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={
                "authorization": "Bearer sk-litellm-key",
                "x-request-id": "req-123",
            },
            proxy_logging_obj=None,
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        )

        assert captured_extra_headers == {"x-request-id": "req-123"}

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_v2_authz_code_drops_caller_authorization(
        self,
    ):
        """A v2-migrated per-user OAuth (authorization_code) server must NOT seed a
        caller-forwarded Authorization into extra_headers — the resolver injects the
        stored per-user token, and apply-if-absent would otherwise let the caller's
        header override another user's stored credential (matches v1's overwrite)."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        # oauth2, not M2M, not delegate => to_server_spec maps it to AuthorizationCodeConfig
        server = MCPServer(
            server_id="server-authz-code-call",
            name="authz-code-server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
        )
        # Migrated authorization_code => the centralized strip decision says drop the
        # caller's Authorization (the v2 resolver injects the stored token).
        assert _should_strip_caller_authorization(mcp_server=server, raw_headers=None, user_api_key_auth=None) is True

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured_extra_headers = "unset"

        async def capture_create_mcp_client(
            server,
            mcp_auth_header,
            extra_headers,
            stdio_env,
            subject_token=None,
            **kwargs,
        ):  # pragma: no cover - helper
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers={"Authorization": "Bearer caller-supplied-token"},
            raw_headers={"authorization": "Bearer caller-supplied-token"},
            proxy_logging_obj=None,
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        )

        # The caller's Authorization must not reach extra_headers; the v2 resolver is
        # the sole Authorization source for this server.
        assert captured_extra_headers != "unset"
        if captured_extra_headers:
            assert "authorization" not in {k.lower() for k in captured_extra_headers}

    def test_without_authorization_drops_only_the_credential(self):
        # None / empty -> None
        assert _without_authorization(None) is None
        assert _without_authorization({}) is None
        # Only Authorization present -> nothing left -> None (case-insensitive)
        assert _without_authorization({"authorization": "Bearer x"}) is None
        # Authorization dropped, other headers kept
        assert _without_authorization({"Authorization": "Bearer x", "X-Trace-Id": "t"}) == {"X-Trace-Id": "t"}

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_passthrough_forwards_authorization_with_admission_header(
        self,
    ):
        """OAuth pass-through forwards Authorization upstream when x-litellm-api-key
        provides admission — in that case Authorization carries the upstream OAuth
        bearer, not the LiteLLM key."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="server-passthrough-call-admission",
            name="passthrough-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            extra_headers=["Authorization"],
            oauth_passthrough=True,
        )

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured_extra_headers = None

        async def capture_create_mcp_client(
            server,
            mcp_auth_header,
            extra_headers,
            stdio_env,
            subject_token=None,
            **kwargs,
        ):  # pragma: no cover - helper
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={
                "x-litellm-api-key": "Bearer sk-litellm-key",
                "authorization": "Bearer upstream-oauth-bearer",
            },
            proxy_logging_obj=None,
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        )

        assert captured_extra_headers == {"Authorization": "Bearer upstream-oauth-bearer"}

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_passthrough_forwards_authorization_for_anonymous_admission(
        self,
    ):
        """OAuth pass-through cold-start return (RFC 9728): the caller's only
        credential is the upstream bearer in Authorization, and LiteLLM admission
        is anonymous (no api_key on user_api_key_auth). Authorization must be
        forwarded so the delegated flow can complete."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="server-passthrough-call-anon",
            name="passthrough-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            extra_headers=["Authorization"],
            oauth_passthrough=True,
        )

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured_extra_headers = None

        async def capture_create_mcp_client(
            server,
            mcp_auth_header,
            extra_headers,
            stdio_env,
            subject_token=None,
            **kwargs,
        ):  # pragma: no cover - helper
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={"authorization": "Bearer upstream-oauth-bearer"},
            proxy_logging_obj=None,
            user_api_key_auth=UserAPIKeyAuth(api_key=None),
        )

        assert captured_extra_headers == {"Authorization": "Bearer upstream-oauth-bearer"}

    async def _capture_call_extra_headers(self, server, oauth2_headers, raw_headers, user_api_key_auth):
        manager = MCPServerManager()
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))
        captured = {"extra_headers": "unset"}

        async def capture_create_mcp_client(
            server, mcp_auth_header, extra_headers, stdio_env, subject_token=None, **kwargs
        ):  # pragma: no cover - helper
            captured["extra_headers"] = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            proxy_logging_obj=None,
            user_api_key_auth=user_api_key_auth,
        )
        return captured["extra_headers"]

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_true_passthrough_forwards_authorization(self):
        from litellm.proxy._types import UserAPIKeyAuth

        server = MCPServer(
            server_id="server-true-passthrough",
            name="tp-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
        )
        extra_headers = await self._capture_call_extra_headers(
            server,
            oauth2_headers={"Authorization": "Bearer upstream-token"},
            raw_headers={"authorization": "Bearer upstream-token"},
            user_api_key_auth=UserAPIKeyAuth(api_key=None),
        )
        assert extra_headers == {"Authorization": "Bearer upstream-token"}

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_oauth_delegate_forwards_separate_authorization(self):
        from litellm.proxy._types import UserAPIKeyAuth

        server = MCPServer(
            server_id="server-oauth-delegate",
            name="od-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        extra_headers = await self._capture_call_extra_headers(
            server,
            oauth2_headers={"Authorization": "Bearer upstream-token"},
            raw_headers={
                "x-litellm-api-key": "Bearer sk-litellm-key",
                "authorization": "Bearer upstream-token",
            },
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        )
        assert extra_headers == {"Authorization": "Bearer upstream-token"}

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_oauth_delegate_never_forwards_admission_key(self):
        from litellm.proxy._types import UserAPIKeyAuth

        server = MCPServer(
            server_id="server-oauth-delegate-leak",
            name="od-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        extra_headers = await self._capture_call_extra_headers(
            server,
            oauth2_headers={"Authorization": "Bearer sk-litellm-key"},
            raw_headers={"authorization": "Bearer sk-litellm-key"},
            user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
        )
        assert not extra_headers or "authorization" not in {k.lower() for k in extra_headers}

    def test_should_strip_caller_authorization_new_modes(self):
        from litellm.proxy._types import UserAPIKeyAuth

        true_passthrough = MCPServer(
            server_id="tp",
            name="tp",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
        )
        assert (
            _should_strip_caller_authorization(
                mcp_server=true_passthrough,
                raw_headers={"authorization": "Bearer upstream"},
                user_api_key_auth=UserAPIKeyAuth(api_key=None),
            )
            is False
        )

        oauth_delegate = MCPServer(
            server_id="od",
            name="od",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        assert (
            _should_strip_caller_authorization(
                mcp_server=oauth_delegate,
                raw_headers={
                    "x-litellm-api-key": "Bearer sk-litellm-key",
                    "authorization": "Bearer upstream",
                },
                user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
            )
            is False
        )
        assert (
            _should_strip_caller_authorization(
                mcp_server=oauth_delegate,
                raw_headers={"authorization": "Bearer sk-litellm-key"},
                user_api_key_auth=UserAPIKeyAuth(api_key="sk-litellm-key"),
            )
            is True
        )

    def test_should_strip_authorization_for_oauth_delegate_admitted_via_jwt_without_api_key(self):
        """JWT / SSO / OIDC / session admission yields a UserAPIKeyAuth with a user_id but
        api_key=None; the caller's Authorization was that credential and must be stripped for
        oauth_delegate when no separate x-litellm-api-key carried admission (LIT-3794-class leak)."""
        from litellm.proxy._types import UserAPIKeyAuth

        oauth_delegate = MCPServer(
            server_id="od-jwt",
            name="od",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        assert (
            _should_strip_caller_authorization(
                mcp_server=oauth_delegate,
                raw_headers={"authorization": "Bearer eyJ-idp-jwt"},
                user_api_key_auth=UserAPIKeyAuth(user_id="alice", api_key=None),
            )
            is True
        )
        assert (
            _should_strip_caller_authorization(
                mcp_server=oauth_delegate,
                raw_headers={
                    "x-litellm-api-key": "Bearer sk-1234",
                    "authorization": "Bearer upstream",
                },
                user_api_key_auth=UserAPIKeyAuth(user_id="alice", api_key=None),
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_oauth_delegate_never_forwards_jwt_admission(self):
        from litellm.proxy._types import UserAPIKeyAuth

        server = MCPServer(
            server_id="od-jwt-e2e",
            name="od",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        extra_headers = await self._capture_call_extra_headers(
            server,
            oauth2_headers={"Authorization": "Bearer eyJ-idp-jwt"},
            raw_headers={"authorization": "Bearer eyJ-idp-jwt"},
            user_api_key_auth=UserAPIKeyAuth(user_id="alice", api_key=None),
        )
        assert not extra_headers or "authorization" not in {k.lower() for k in extra_headers}

    def test_new_passthrough_modes_require_per_user_auth(self):
        for auth_type in (MCPAuth.true_passthrough, MCPAuth.oauth_delegate):
            server = MCPServer(
                server_id="s",
                name="s",
                url="https://example.com",
                transport=MCPTransport.http,
                auth_type=auth_type,
            )
            assert server.requires_per_user_auth is True

    @pytest.mark.asyncio
    async def test_create_mcp_client_forwarded_modes_use_the_passthrough_arm(self):
        manager = MCPServerManager()
        server = MCPServer(
            server_id="tp-egress",
            name="tp",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
        )
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.resolve_mcp_auth",
                new_callable=AsyncMock,
            ) as mock_resolve,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient") as mock_client_cls,
        ):
            await manager._create_mcp_client(server=server, extra_headers={"Authorization": "Bearer upstream-token"})
        mock_resolve.assert_not_awaited()
        kwargs = mock_client_cls.call_args.kwargs
        emitted = httpx.Request("GET", "https://example.com/mcp")
        flow = kwargs["resolved_auth"].auth_flow(emitted)
        next(flow)
        flow.close()
        assert emitted.headers["Authorization"] == "Bearer upstream-token"
        assert not kwargs["extra_headers"] or "authorization" not in {k.lower() for k in kwargs["extra_headers"]}

    @staticmethod
    def _emitted_authorization(mock_client_cls) -> str:
        kwargs = mock_client_cls.call_args.kwargs
        emitted = httpx.Request("GET", "https://example.com/mcp")
        flow = kwargs["resolved_auth"].auth_flow(emitted)
        next(flow)
        flow.close()
        return emitted.headers["Authorization"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "per_server_header",
        ["Bearer per-server-token", {"Authorization": "Bearer per-server-token"}],
    )
    async def test_create_mcp_client_passthrough_prefers_per_server_token(self, per_server_header):
        """A per-server x-mcp-{alias}-authorization value is the explicit one-token-one-server
        binding, so it must win over the request-wide Authorization and reach the upstream
        verbatim through the passthrough arm."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="tp-per-server",
            name="tp",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
        )
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.resolve_mcp_auth",
                new_callable=AsyncMock,
            ) as mock_resolve,
            patch("litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPClient") as mock_client_cls,
        ):
            await manager._create_mcp_client(
                server=server,
                mcp_auth_header=per_server_header,
                extra_headers={"Authorization": "Bearer global-token"},
            )
        mock_resolve.assert_not_awaited()
        assert self._emitted_authorization(mock_client_cls) == "Bearer per-server-token"
        kwargs = mock_client_cls.call_args.kwargs
        assert not kwargs["extra_headers"] or "authorization" not in {k.lower() for k in kwargs["extra_headers"]}

    def test_consumes_caller_authorization_per_mode(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _consumes_caller_authorization,
        )

        def build(**kwargs) -> MCPServer:
            return MCPServer(
                server_id="s",
                name="s",
                url="https://example.com",
                transport=MCPTransport.http,
                **kwargs,
            )

        assert _consumes_caller_authorization(build(auth_type=MCPAuth.true_passthrough)) is True
        assert _consumes_caller_authorization(build(auth_type=MCPAuth.oauth_delegate)) is True
        assert (
            _consumes_caller_authorization(
                build(auth_type=MCPAuth.none, extra_headers=["Authorization"], oauth_passthrough=True)
            )
            is True
        )
        assert _consumes_caller_authorization(build(auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True)) is True
        assert _consumes_caller_authorization(build(auth_type=MCPAuth.api_key, authentication_token="x")) is False
        assert (
            _consumes_caller_authorization(
                build(
                    auth_type=MCPAuth.oauth2,
                    delegate_auth_to_upstream=True,
                    oauth2_flow="client_credentials",
                    token_url="https://idp/token",
                )
            )
            is False
        )

    def test_caller_authorization_fans_out_only_with_second_consumer(self):
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _caller_authorization_fans_out,
        )

        delegate = MCPServer(
            server_id="od",
            name="od",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
        )
        second = MCPServer(
            server_id="tp",
            name="tp",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
        )
        static_server = MCPServer(
            server_id="static",
            name="static",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.api_key,
            authentication_token="x",
        )

        assert _caller_authorization_fans_out(delegate, None) is False
        assert _caller_authorization_fans_out(delegate, [delegate]) is False
        assert _caller_authorization_fans_out(delegate, [delegate, static_server]) is False
        assert _caller_authorization_fans_out(delegate, [delegate, second]) is True

    @pytest.mark.asyncio
    async def test_get_prompts_from_server_success(self):
        """Ensure prompts are fetched and prefixed when requested."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="server-1",
            name="alias-server",
            alias="alias-server",
            server_name="alias-server",
            url="https://example.com",
            transport=MCPTransport.http,
        )

        mock_prompt = Prompt(name="hello", description="Say hi")
        mock_client = AsyncMock()
        mock_client.list_prompts = AsyncMock(return_value=[mock_prompt])

        with patch.object(
            manager,
            "_create_mcp_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            prompts = await manager.get_prompts_from_server(server, add_prefix=True)

        mock_client.list_prompts.assert_awaited_once()
        assert len(prompts) == 1
        assert prompts[0].name == "alias-server-hello"

    @pytest.mark.asyncio
    async def test_get_prompt_from_server_success(self):
        """Ensure a single prompt definition is requested via the MCP client."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="server-1",
            name="alias-server",
            alias="alias-server",
            server_name="alias-server",
            url="https://example.com",
            transport=MCPTransport.http,
        )

        mock_result = GetPromptResult(
            description="Hello world prompt",
            messages=[],
        )
        mock_client = AsyncMock()
        mock_client.get_prompt = AsyncMock(return_value=mock_result)

        with patch.object(
            manager,
            "_create_mcp_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await manager.get_prompt_from_server(
                server=server,
                prompt_name="hello",
                arguments={"tone": "casual"},
            )

        mock_client.get_prompt.assert_awaited_once()
        awaited_call = mock_client.get_prompt.await_args
        called_params = awaited_call.args[0]
        assert called_params.name == "hello"
        assert called_params.arguments == {"tone": "casual"}
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_get_resources_from_server_success(self):
        manager = MCPServerManager()

        server = MCPServer(
            server_id="server-1",
            name="alias-server",
            alias="alias-server",
            server_name="alias-server",
            url="https://example.com",
            transport=MCPTransport.http,
            static_headers={"X-Static": "static"},
        )

        mock_client = AsyncMock()
        mock_resources = [Resource(name="file", uri="https://example.com/file")]
        mock_client.list_resources = AsyncMock(return_value=mock_resources)
        prefixed_resources = [Resource(name="alias-server-file", uri="https://example.com/file")]

        with (
            patch.object(
                manager,
                "_create_mcp_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ) as mock_create_client,
            patch.object(
                manager,
                "_create_prefixed_resources",
                return_value=prefixed_resources,
            ) as mock_prefix,
        ):
            result = await manager.get_resources_from_server(
                server=server,
                mcp_auth_header="auth",
                extra_headers={"X-Test": "1"},
                add_prefix=True,
            )

        mock_create_client.assert_called_once()
        called_kwargs = mock_create_client.call_args.kwargs
        assert called_kwargs["server"] is server
        assert called_kwargs["mcp_auth_header"] == "auth"
        assert called_kwargs["extra_headers"] == {"X-Test": "1", "X-Static": "static"}
        mock_client.list_resources.assert_awaited_once()
        mock_prefix.assert_called_once_with(mock_resources, server, add_prefix=True)
        assert result == prefixed_resources

    @pytest.mark.asyncio
    async def test_get_resource_templates_from_server_success(self):
        manager = MCPServerManager()

        server = MCPServer(
            server_id="server-1",
            name="alias-server",
            alias="alias-server",
            server_name="alias-server",
            url="https://example.com",
            transport=MCPTransport.http,
        )

        mock_client = AsyncMock()
        mock_templates = [
            ResourceTemplate(
                name="template",
                uriTemplate="https://example.com/{id}",
            )
        ]
        mock_client.list_resource_templates = AsyncMock(return_value=mock_templates)
        prefixed_templates = [
            ResourceTemplate(
                name="alias-server-template",
                uriTemplate="https://example.com/{id}",
            )
        ]

        with (
            patch.object(
                manager,
                "_create_mcp_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ) as mock_create_client,
            patch.object(
                manager,
                "_create_prefixed_resource_templates",
                return_value=prefixed_templates,
            ) as mock_prefix,
        ):
            result = await manager.get_resource_templates_from_server(
                server=server,
                mcp_auth_header="auth",
                extra_headers=None,
                add_prefix=False,
            )

        mock_create_client.assert_called_once_with(
            server=server,
            mcp_auth_header="auth",
            extra_headers=None,
            stdio_env=None,
            subject_token=None,
        )
        mock_client.list_resource_templates.assert_awaited_once()
        mock_prefix.assert_called_once_with(mock_templates, server, add_prefix=False)
        assert result == prefixed_templates

    @pytest.mark.asyncio
    async def test_read_resource_from_server_success(self):
        manager = MCPServerManager()

        server = MCPServer(
            server_id="server-1",
            name="alias-server",
            alias="alias-server",
            server_name="alias-server",
            url="https://example.com",
            transport=MCPTransport.http,
            static_headers={"X-Static": "1"},
        )

        mock_client = AsyncMock()
        read_result = ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="https://example.com/resource",
                    text="hello",
                    mimeType="text/plain",
                )
            ]
        )
        mock_client.read_resource = AsyncMock(return_value=read_result)

        with patch.object(
            manager,
            "_create_mcp_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ) as mock_create_client:
            result = await manager.read_resource_from_server(
                server=server,
                url="https://example.com/resource",
                mcp_auth_header="auth",
                extra_headers={"X-Test": "1"},
            )

        mock_create_client.assert_called_once()
        called_kwargs = mock_create_client.call_args.kwargs
        assert called_kwargs["extra_headers"] == {"X-Test": "1", "X-Static": "1"}
        mock_client.read_resource.assert_awaited_once_with("https://example.com/resource")
        assert result is read_result

    @pytest.mark.asyncio
    async def test_fetch_oauth_metadata_from_resource_returns_servers_and_scopes(self):
        manager = MCPServerManager()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "authorization_servers": [
                "https://auth1.example.com",
                "https://auth2.example.com",
            ],
            "scopes_supported": ["read", "write"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://protected.example.com/.well-known/oauth",
                "https://protected.example.com/mcp",
            )

        assert servers == [
            "https://auth1.example.com",
            "https://auth2.example.com",
        ]
        assert scopes == ["read", "write"]

    @pytest.mark.asyncio
    async def test_descovery_metadata_probes_well_known_when_server_does_not_challenge(
        self,
    ):
        manager = MCPServerManager()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_metadata = MCPOAuthMetadata(
            scopes=None,
            authorization_url="https://login.microsoftonline.com/tenant/oauth2/v2.0/authorize",
            token_url="https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
            registration_url=None,
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
                return_value=mock_client,
            ),
            patch.object(
                manager,
                "_attempt_well_known_discovery",
                AsyncMock(
                    return_value=(
                        ["https://login.microsoftonline.com/test-tenant-id/v2.0"],
                        ["api://some-scope/.default"],
                    )
                ),
            ) as mock_well_known,
            patch.object(
                manager,
                "_fetch_authorization_server_metadata",
                AsyncMock(return_value=mock_metadata),
            ) as mock_fetch_auth,
        ):
            result = await manager._descovery_metadata("http://localhost:8001/mcp")

        mock_well_known.assert_awaited_once_with("http://localhost:8001/mcp")
        mock_fetch_auth.assert_awaited_once_with(
            ["https://login.microsoftonline.com/test-tenant-id/v2.0"],
            "http://localhost:8001/mcp",
        )
        assert result is mock_metadata
        assert result.scopes == ["api://some-scope/.default"]

    @pytest.mark.asyncio
    async def test_fetch_single_authorization_server_metadata_supports_azure_issuer_path(
        self,
    ):
        manager = MCPServerManager()
        issuer = "https://login.microsoftonline.com/test-tenant-id/v2.0"

        def build_response(url: str, **kwargs):
            mock_response = MagicMock()
            if url == f"{issuer}/.well-known/openid-configuration":
                mock_response.json.return_value = {
                    "authorization_endpoint": "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/authorize",
                    "token_endpoint": "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/token",
                    "scopes_supported": ["api://some-scope/.default"],
                }
                mock_response.raise_for_status = MagicMock()
            else:
                request = httpx.Request("GET", url)
                response_obj = httpx.Response(status_code=404, request=request)
                mock_response.raise_for_status = MagicMock(
                    side_effect=httpx.HTTPStatusError("not found", request=request, response=response_obj)
                )
            return mock_response

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=build_response)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            # The Azure issuer is cross-origin against the server_url — use
            # the issuer itself as server_url so the test exercises the
            # well-known fetch logic without needing real DNS.
            result = await manager._fetch_single_authorization_server_metadata(issuer, issuer)

        assert result is not None
        assert result.authorization_url == "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/authorize"
        assert result.token_url == "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/token"
        assert result.scopes == ["api://some-scope/.default"]

    @pytest.mark.asyncio
    async def test_fetch_single_authorization_server_metadata_derives_azure_metadata(
        self,
    ):
        manager = MCPServerManager()
        issuer = "https://login.microsoftonline.com/test-tenant-id/v2.0"

        request = httpx.Request("GET", issuer)
        response_obj = httpx.Response(status_code=404, request=request)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("not found", request=request, response=response_obj)
        )

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await manager._fetch_single_authorization_server_metadata(issuer, issuer)

        assert result is not None
        assert result.authorization_url == "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/authorize"
        assert result.token_url == "https://login.microsoftonline.com/test-tenant-id/oauth2/v2.0/token"

    @pytest.mark.asyncio
    async def test_descovery_metadata_falls_back_to_origin_when_no_auth_servers(self):
        manager = MCPServerManager()
        server_url = "https://example.com/public/mcp"

        request = httpx.Request("GET", server_url)
        response_obj = httpx.Response(
            status_code=401,
            request=request,
            headers={"WWW-Authenticate": 'Bearer scope="read"'},
        )

        def raise_http_error():
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response_obj)

        response_obj.raise_for_status = MagicMock(side_effect=raise_http_error)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=response_obj)

        mock_metadata = MCPOAuthMetadata(
            scopes=None,
            authorization_url="https://example.com/auth",
            token_url="https://example.com/token",
            registration_url=None,
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
                return_value=mock_client,
            ),
            patch.object(
                manager,
                "_fetch_oauth_metadata_from_resource",
                AsyncMock(return_value=([], None)),
            ),
            patch.object(
                manager,
                "_attempt_well_known_discovery",
                AsyncMock(return_value=([], None)),
            ),
            patch.object(
                manager,
                "_fetch_authorization_server_metadata",
                AsyncMock(return_value=mock_metadata),
            ) as mock_fetch_auth,
        ):
            result = await manager._descovery_metadata(server_url)

        mock_fetch_auth.assert_awaited_once_with(["https://example.com"], server_url)
        assert result is mock_metadata
        assert result.scopes == ["read"]

    @pytest.mark.asyncio
    async def test_load_servers_from_config_overrides_discovery_metadata(self):
        manager = MCPServerManager()

        discovered_metadata = MCPOAuthMetadata(
            scopes=["discovered"],
            authorization_url="https://discovered.example.com/auth",
            token_url="https://discovered.example.com/token",
            registration_url="https://discovered.example.com/register",
        )

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            assert server_url == "https://example.com/mcp"
            # oauth2 (browser flow) keeps the origin fallback; only OBO disables it.
            assert allow_origin_fallback is True
            return discovered_metadata

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        config = {
            "example": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2,
                "oauth2_flow": "authorization_code",
                "scopes": ["config"],
                "authorization_url": "https://config.example.com/auth",
            }
        }

        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.scopes == ["config"]  # config overrides discovery
        assert server.authorization_url == "https://config.example.com/auth"
        assert server.token_url == "https://discovered.example.com/token"
        assert server.registration_url == "https://discovered.example.com/register"

    @pytest.mark.asyncio
    async def test_load_servers_from_config_filters_blank_scopes(self):
        """A YAML ``scopes: [""]`` must normalize to None (matching the DB path), so a blank-only
        list never becomes a ``("",)`` tuple that skips the entra_obo fail-closed scope check."""
        manager = MCPServerManager()
        config = {
            "entra": {
                "url": "https://up.example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2_token_exchange,
                "token_exchange_profile": "entra_obo",
                "token_exchange_endpoint": "https://login.microsoftonline.com/t/oauth2/v2.0/token",
                "client_id": "cid",
                "client_secret": "csec",
                "scopes": ["", "  "],
            }
        }

        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.scopes is None

        # And the exchange precondition now fails closed before any IdP call.
        from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import to_server_spec
        from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
        from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
            OboTokenExchanger,
        )

        spec = to_server_spec(server)
        assert spec is not None
        assert spec.config.scopes == ()

        async def _must_not_post(url, form, headers):
            raise AssertionError("entra_obo with blank scopes must fail closed before POSTing to the IdP")

        result = await OboTokenExchanger(_must_not_post).exchange("subj", spec, spec.config)
        assert isinstance(result, Error)
        assert result.error.tag == "misconfigured"

    @pytest.mark.asyncio
    async def test_load_servers_from_config_reads_all_token_exchange_fields(self):
        """Every token-exchange setting is configurable through config.yaml as a top-level
        key (the config counterpart of the REST/UI columns) and reaches the resolver spec;
        omitted keys resolve to their documented defaults. token_exchange servers need no
        oauth2_flow (that requirement is oauth2-only)."""
        from litellm.types.mcp import DEFAULT_SUBJECT_TOKEN_TYPE

        manager = MCPServerManager()
        config = {
            "te_full": {
                "url": "https://up.example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2_token_exchange,
                "token_exchange_endpoint": "https://idp.example.com/oauth2/token",
                "audience": "api://upstream",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_exchange_profile": "entra_obo",
                "client_id": "cid",
                "client_secret": "csec",
                "scopes": ["api://upstream/.default"],
            },
            "te_minimal": {
                "url": "https://up2.example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2_token_exchange,
                "token_exchange_endpoint": "https://idp2.example.com/oauth2/token",
                "client_id": "cid2",
                "client_secret": "csec2",
            },
        }

        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            await manager.load_servers_from_config(config)

        by_name = {s.server_name: s for s in manager.config_mcp_servers.values()}

        full = by_name["te_full"]
        assert full.token_exchange_endpoint == "https://idp.example.com/oauth2/token"
        assert full.audience == "api://upstream"
        assert full.subject_token_type == "urn:ietf:params:oauth:token-type:jwt"
        assert full.token_exchange_profile == "entra_obo"

        minimal = by_name["te_minimal"]
        assert minimal.audience is None
        assert minimal.subject_token_type == DEFAULT_SUBJECT_TOKEN_TYPE
        assert minimal.token_exchange_profile == "rfc8693"

        from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import to_server_spec

        spec = to_server_spec(full)
        assert spec is not None
        assert spec.config.token_exchange_endpoint == "https://idp.example.com/oauth2/token"
        assert spec.config.profile == "entra_obo"

        minimal_spec = to_server_spec(minimal)
        assert minimal_spec is not None
        assert minimal_spec.config.subject_token_type == DEFAULT_SUBJECT_TOKEN_TYPE
        assert minimal_spec.config.profile == "rfc8693"

    @pytest.mark.asyncio
    async def test_config_oauth_initialize_tool_name_to_mcp_server_name_mapping(self):
        manager = MCPServerManager()

        config = {
            "example": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2,
                "oauth2_flow": "authorization_code",
                "scopes": ["config"],
                "authorization_url": "https://config.example.com/auth",
            }
        }

        await manager.load_servers_from_config(config)

        # Initialize the tool mapping
        await manager._initialize_tool_name_to_mcp_server_name_mapping()
        assert manager.tool_name_to_mcp_server_name_mapping == {}

    @pytest.mark.asyncio
    async def test_list_tools_handles_missing_server_alias(self):
        """Test that list_tools handles servers without alias gracefully"""
        manager = MCPServerManager()

        # Mock server without alias
        server = MagicMock()
        server.name = "github"
        server.alias = None  # No alias
        server.server_name = "github"

        # Mock get_allowed_mcp_servers
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock _get_tools_from_server
        async def mock_get_tools_from_server(
            server,
            mcp_auth_header=None,
            **kwargs,
        ):
            assert mcp_auth_header == "server-specific-token"  # Should use server-specific header via server_name
            tool = MagicMock()
            tool.name = "github_tool_1"
            return [tool]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Test with server-specific headers that match server_name (even without alias)
        result = await manager.list_tools(
            mcp_auth_header="legacy-token",
            mcp_server_auth_headers={"github": "server-specific-token"},
        )

        assert len(result) == 1
        assert result[0].name == "github_tool_1"

    @pytest.mark.asyncio
    async def test_health_check_server_healthy(self):
        """Test health check for a healthy server"""
        manager = MCPServerManager()

        # Mock server
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            auth_type=None,
            authentication_token="test-token",
            url="http://test-server.com",
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock successful client.run_with_session
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(return_value="ok")
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        # Perform health check
        result = await manager.health_check_server("test-server")

        # Verify results - result is now LiteLLM_MCPServerTable
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "test-server"
        assert result.status == "healthy"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_server_unhealthy(self):
        """Test health check for an unhealthy server"""
        manager = MCPServerManager()

        # Mock server
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            auth_type=None,
            authentication_token="test-token",
            url="http://test-server.com",
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock failed client.run_with_session
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(side_effect=Exception("Connection timeout"))
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        # Perform health check
        result = await manager.health_check_server("test-server")

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "test-server"
        assert result.status == "unhealthy"
        assert result.health_check_error == "Connection timeout"
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_server_not_found(self):
        """Test health check for a server that doesn't exist"""
        manager = MCPServerManager()

        # Mock server not found
        manager.get_mcp_server_by_id = MagicMock(return_value=None)

        # Perform health check
        result = await manager.health_check_server("non-existent-server")

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "non-existent-server"
        assert result.server_name is None
        assert result.status == "unknown"
        assert result.health_check_error == "Server not found"
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_server_oauth2_skips_check(self):
        """Test that health check is skipped for OAuth2 servers and returns unknown status"""
        manager = MCPServerManager()

        # Mock OAuth2 server
        server = MCPServer(
            server_id="oauth2-server",
            name="oauth2-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            url="http://oauth2-server.com",
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # _create_mcp_client should not be called for OAuth2 servers
        manager._create_mcp_client = AsyncMock()

        # Perform health check
        result = await manager.health_check_server("oauth2-server")

        # Verify that client was not created (health check was skipped)
        manager._create_mcp_client.assert_not_called()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "oauth2-server"
        assert result.status == "unknown"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_server_no_token_skips_check(self):
        """Test that health check is skipped when auth_type is set but authentication_token is missing"""
        manager = MCPServerManager()

        # Mock server with auth_type but no authentication_token
        server = MCPServer(
            server_id="no-token-server",
            name="no-token-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.bearer_token,
            authentication_token=None,  # No token
            url="http://no-token-server.com",
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # _create_mcp_client should not be called
        manager._create_mcp_client = AsyncMock()

        # Perform health check
        result = await manager.health_check_server("no-token-server")

        # Verify that client was not created (health check was skipped)
        manager._create_mcp_client.assert_not_called()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "no-token-server"
        assert result.status == "unknown"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_server_with_static_headers(self):
        """Test health check with static headers configured"""
        manager = MCPServerManager()

        # Mock server with static_headers
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            auth_type=None,
            authentication_token="test-token",
            url="http://test-server.com",
            static_headers={"X-Custom-Header": "custom-value"},
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock successful client
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(return_value="ok")

        # Capture the extra_headers passed to _create_mcp_client
        captured_extra_headers = None

        async def capture_create_mcp_client(server, mcp_auth_header, extra_headers, stdio_env):
            nonlocal captured_extra_headers
            captured_extra_headers = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=capture_create_mcp_client)

        # Perform health check
        result = await manager.health_check_server("test-server")

        # Verify static headers were passed
        assert captured_extra_headers == {"X-Custom-Header": "custom-value"}

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "test-server"
        assert result.status == "healthy"
        assert result.health_check_error is None

    @pytest.mark.asyncio
    async def test_health_check_skips_passthrough_auth_with_authorization_header(self):
        """Test that health check is skipped for servers with passthrough Authorization header"""
        manager = MCPServerManager()

        # Mock server with auth_type=none and Authorization in extra_headers (passthrough auth)
        server = MCPServer(
            server_id="github-server",
            name="github-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            authentication_token=None,
            url="http://github-server.com",
            extra_headers=["Authorization"],  # Passthrough auth configured
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # _create_mcp_client should not be called (health check should be skipped)
        manager._create_mcp_client = AsyncMock()

        # Perform health check
        result = await manager.health_check_server("github-server")

        # Verify that client was not created (health check was skipped)
        manager._create_mcp_client.assert_not_called()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "github-server"
        assert result.status == "unknown"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_skips_passthrough_auth_with_api_key_header(self):
        """Test that health check is skipped for servers with passthrough x-api-key header"""
        manager = MCPServerManager()

        # Mock server with auth_type=none and x-api-key in extra_headers
        server = MCPServer(
            server_id="sourcegraph-server",
            name="sourcegraph-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            authentication_token=None,
            url="http://sourcegraph-server.com",
            extra_headers=["x-api-key"],  # Passthrough auth configured
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # _create_mcp_client should not be called
        manager._create_mcp_client = AsyncMock()

        # Perform health check
        result = await manager.health_check_server("sourcegraph-server")

        # Verify that client was not created (health check was skipped)
        manager._create_mcp_client.assert_not_called()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "sourcegraph-server"
        assert result.status == "unknown"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_runs_when_no_passthrough_auth(self):
        """Test that health check runs normally for servers with auth_type=none but no passthrough headers"""
        manager = MCPServerManager()

        # Mock server with auth_type=none but no extra_headers (no passthrough auth)
        server = MCPServer(
            server_id="public-server",
            name="public-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            authentication_token=None,
            url="http://public-server.com",
            extra_headers=None,  # No passthrough auth
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock successful client
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(return_value="ok")
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        # Perform health check
        result = await manager.health_check_server("public-server")

        # Verify that client WAS created (health check should run)
        manager._create_mcp_client.assert_called_once()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "public-server"
        assert result.status == "healthy"
        assert result.health_check_error is None
        assert result.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_runs_when_extra_headers_no_auth(self):
        """Test that health check runs when extra_headers exist but don't include auth headers"""
        manager = MCPServerManager()

        # Mock server with extra_headers but no auth-related headers
        server = MCPServer(
            server_id="custom-server",
            name="custom-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            authentication_token=None,
            url="http://custom-server.com",
            extra_headers=["X-Custom-Header", "X-Request-ID"],  # Non-auth headers
        )

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock successful client
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(return_value="ok")
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        # Perform health check
        result = await manager.health_check_server("custom-server")

        # Verify that client WAS created (health check should run)
        manager._create_mcp_client.assert_called_once()

        # Verify results
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "custom-server"
        assert result.status == "healthy"
        assert result.health_check_error is None

    @pytest.mark.asyncio
    async def test_requires_per_user_auth_property_oauth2(self):
        """Test that requires_per_user_auth returns True for OAuth2 without client credentials"""
        # OAuth2 without client credentials
        server = MCPServer(
            server_id="oauth-server",
            name="oauth-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            url="http://oauth-server.com",
            client_id=None,
            client_secret=None,
            token_url=None,
        )
        assert server.requires_per_user_auth is True
        assert server.needs_user_oauth_token is True

    @pytest.mark.asyncio
    async def test_requires_per_user_auth_property_oauth2_with_client_creds(self):
        """Test that requires_per_user_auth returns False for OAuth2 with client credentials"""
        # M2M must be opted in explicitly with oauth2_flow="client_credentials"
        server = MCPServer(
            server_id="oauth-server",
            name="oauth-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            url="http://oauth-server.com",
            client_id="client-id",
            client_secret="client-secret",
            token_url="http://oauth-server.com/token",
            oauth2_flow="client_credentials",
        )
        assert server.requires_per_user_auth is False
        assert server.has_client_credentials is True

    @pytest.mark.asyncio
    async def test_requires_per_user_auth_property_passthrough_auth(self):
        """Test that requires_per_user_auth returns True for passthrough auth (auth_type=none + Authorization header)"""
        # Passthrough auth with Authorization header
        server = MCPServer(
            server_id="github-server",
            name="github-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            url="http://github-server.com",
            extra_headers=["Authorization"],
        )
        assert server.requires_per_user_auth is True

        # Passthrough auth with x-api-key header
        server2 = MCPServer(
            server_id="sourcegraph-server",
            name="sourcegraph-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            url="http://sourcegraph-server.com",
            extra_headers=["x-api-key"],
        )
        assert server2.requires_per_user_auth is True

        # Passthrough auth with api-key header (case insensitive)
        server3 = MCPServer(
            server_id="api-server",
            name="api-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            url="http://api-server.com",
            extra_headers=["API-Key"],
        )
        assert server3.requires_per_user_auth is True

    @pytest.mark.asyncio
    async def test_requires_per_user_auth_property_no_passthrough(self):
        """Test that requires_per_user_auth returns False when no passthrough auth is configured"""
        # auth_type=none but no extra_headers
        server = MCPServer(
            server_id="public-server",
            name="public-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            url="http://public-server.com",
            extra_headers=None,
        )
        assert server.requires_per_user_auth is False

        # auth_type=none with non-auth extra_headers
        server2 = MCPServer(
            server_id="custom-server",
            name="custom-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            url="http://custom-server.com",
            extra_headers=["X-Custom-Header", "X-Request-ID"],
        )
        assert server2.requires_per_user_auth is False

    @pytest.mark.asyncio
    async def test_register_openapi_tools_includes_static_headers(self, tmp_path):
        """Ensure OpenAPI-to-MCP tool calls include server.static_headers (Issue #19341)."""
        manager = MCPServerManager()

        spec_path = tmp_path / "openapi.json"
        spec_path.write_text(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "info": {"title": "Demo", "version": "1.0.0"},
                    "paths": {
                        "/health": {
                            "get": {
                                "operationId": "health_check",
                                "summary": "health",
                            }
                        }
                    },
                }
            )
        )

        server = MCPServer(
            server_id="openapi-server",
            name="openapi-server",
            server_name="openapi-server",
            url="https://example.com",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            static_headers={"Authorization": "STATIC token"},
        )

        captured: dict = {}

        def fake_create_tool_function(path, method, operation, base_url, headers=None):
            captured["headers"] = headers

            async def tool_func(**kwargs):
                return "ok"

            return tool_func

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.create_tool_function",
                side_effect=fake_create_tool_function,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator.build_input_schema",
                return_value={"type": "object", "properties": {}, "required": []},
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.tool_registry.global_mcp_tool_registry.register_tool",
                return_value=None,
            ),
        ):
            await manager._register_openapi_tools(
                spec_path=str(spec_path),
                server=server,
                base_url="https://example.com",
            )

        assert captured["headers"] is not None
        assert captured["headers"]["Authorization"] == "STATIC token"

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_allowed_tools_list_allows_tool(self):
        """Test pre_call_tool_check allows tool when it's in allowed_tools list"""
        manager = MCPServerManager()

        # Create server with allowed_tools list
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=["allowed_tool", "another_allowed_tool"],
            disallowed_tools=None,
        )

        # Mock dependencies - set object_permission and object_permission_id to None
        # so permission checks return None (no restrictions)
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None
        proxy_logging_obj = MagicMock()

        # Mock the async methods that pre_call_tool_check calls
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})

        # This should not raise an exception
        await manager.pre_call_tool_check(
            name="allowed_tool",
            arguments={"param": "value"},
            server_name="test-server",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_allowed_tools_list_blocks_tool(self):
        """Test pre_call_tool_check blocks tool when it's not in allowed_tools list"""
        manager = MCPServerManager()

        # Create server with allowed_tools list
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=["allowed_tool", "another_allowed_tool"],
            disallowed_tools=None,
        )

        # Mock dependencies
        user_api_key_auth = MagicMock()
        proxy_logging_obj = MagicMock()

        # This should raise an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await manager.pre_call_tool_check(
                name="blocked_tool",
                arguments={"param": "value"},
                server_name="test-server",
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=server,
            )

        assert exc_info.value.status_code == 403
        assert "Tool blocked_tool is not allowed for server test-server" in exc_info.value.detail["error"]
        assert "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_disallowed_tools_list_allows_tool(self):
        """Test pre_call_tool_check allows tool when it's not in disallowed_tools list"""
        manager = MCPServerManager()

        # Create server with disallowed_tools list
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=None,
            disallowed_tools=["banned_tool", "another_banned_tool"],
        )

        # Mock dependencies - set object_permission and object_permission_id to None
        # so permission checks return None (no restrictions)
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None
        proxy_logging_obj = MagicMock()

        # Mock the async methods that pre_call_tool_check calls
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})

        # This should not raise an exception
        await manager.pre_call_tool_check(
            name="allowed_tool",
            arguments={"param": "value"},
            server_name="test-server",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_disallowed_tools_list_blocks_tool(self):
        """Test pre_call_tool_check blocks tool when it's in disallowed_tools list"""
        manager = MCPServerManager()

        # Create server with disallowed_tools list
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=None,
            disallowed_tools=["banned_tool", "another_banned_tool"],
        )

        # Mock dependencies
        user_api_key_auth = MagicMock()
        proxy_logging_obj = MagicMock()

        # This should raise an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await manager.pre_call_tool_check(
                name="banned_tool",
                arguments={"param": "value"},
                server_name="test-server",
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=server,
            )

        assert exc_info.value.status_code == 403
        assert "Tool banned_tool is not allowed for server test-server" in exc_info.value.detail["error"]
        assert "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_no_restrictions_allows_any_tool(self):
        """Test pre_call_tool_check allows any tool when no restrictions are set"""
        manager = MCPServerManager()

        # Create server with no tool restrictions
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=None,
            disallowed_tools=None,
        )

        # Mock dependencies - set object_permission and object_permission_id to None
        # so permission checks return None (no restrictions)
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None
        proxy_logging_obj = MagicMock()

        # Mock the async methods that pre_call_tool_check calls
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})

        # This should not raise an exception
        await manager.pre_call_tool_check(
            name="any_tool",
            arguments={"param": "value"},
            server_name="test-server",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

    @pytest.mark.asyncio
    async def test_pre_call_tool_check_allowed_tools_takes_precedence(self):
        """Test that allowed_tools list takes precedence over disallowed_tools list"""
        manager = MCPServerManager()

        # Create server with both allowed_tools and disallowed_tools
        # Note: The logic in check_allowed_or_banned_tools prioritizes allowed_tools
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.stdio,
            allowed_tools=["tool1", "tool2"],
            disallowed_tools=["tool2", "tool3"],  # tool2 is in both lists
        )

        # Mock dependencies - set object_permission and object_permission_id to None
        # so permission checks return None (no restrictions)
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None
        proxy_logging_obj = MagicMock()

        # Mock the async methods that pre_call_tool_check calls
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})

        # tool2 should be allowed since it's in allowed_tools (takes precedence)
        await manager.pre_call_tool_check(
            name="tool2",
            arguments={"param": "value"},
            server_name="test-server",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

        # tool3 should be blocked since it's not in allowed_tools
        with pytest.raises(HTTPException) as exc_info:
            await manager.pre_call_tool_check(
                name="tool3",
                arguments={"param": "value"},
                server_name="test-server",
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=server,
            )

        assert exc_info.value.status_code == 403
        assert "Tool tool3 is not allowed for server test-server" in exc_info.value.detail["error"]

    async def test_get_tools_from_server_add_prefix(self):
        """Verify _get_tools_from_server respects add_prefix True/False."""
        manager = MCPServerManager()

        # Create a minimal server with alias used as prefix
        server = MCPServer(
            server_id="zapier",
            name="zapier",
            transport=MCPTransport.http,
        )

        # Mock client creation and fetching tools
        manager._create_mcp_client = AsyncMock(return_value=object())

        # Tools returned upstream (unprefixed from provider)
        upstream_tool = MCPTool(
            name="send_email",
            description="Send an email",
            inputSchema={},
        )

        manager._fetch_tools_with_timeout = AsyncMock(return_value=[upstream_tool])

        # Case 1: add_prefix=True (default for multi-server) -> expect prefixed
        tools_prefixed = await manager._get_tools_from_server(server, add_prefix=True)
        assert len(tools_prefixed) == 1
        assert tools_prefixed[0].name == "zapier-send_email"

        # Case 2: add_prefix=False (single-server) -> expect unprefixed
        tools_unprefixed = await manager._get_tools_from_server(server, add_prefix=False)
        assert len(tools_unprefixed) == 1
        assert tools_unprefixed[0].name == "send_email"

    @pytest.mark.asyncio
    async def test_get_tools_from_server_jwt_skipped_when_mcp_auth_header_set(self):
        """When a per-user mcp_auth_header is resolved, JWT injection must be skipped.

        MCPClient._get_auth_headers() applies extra_headers AFTER writing
        Authorization from auth_value, so an injected JWT would clobber the
        user's per-server OAuth token. Regression test for that interaction.
        """
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="zapier",
            name="zapier",
            transport=MCPTransport.http,
        )

        manager._create_mcp_client = AsyncMock(return_value=object())
        manager._fetch_tools_with_timeout = AsyncMock(return_value=[])

        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")

        with (
            patch(
                "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer.get_mcp_jwt_signer",
                return_value=MagicMock(),
            ),
            patch(
                "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer.inject_mcp_jwt_headers_for_upstream",
                new=AsyncMock(return_value={"Authorization": "Bearer signed-jwt"}),
            ) as mock_inject,
        ):
            # Case A: mcp_auth_header present -> JWT must NOT be injected
            await manager._get_tools_from_server(
                server,
                mcp_auth_header="oauth-user-token",
                user_api_key_auth=user_auth,
            )
            mock_inject.assert_not_called()

            # Case B: no mcp_auth_header -> JWT injection runs as before
            await manager._get_tools_from_server(
                server,
                user_api_key_auth=user_auth,
            )
            mock_inject.assert_awaited_once()

    def test_resolve_mcp_server_for_tool_call_via_prefixed_name(self):
        """Resolution succeeds when the prefixed tool name is in the mapping."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="jira",
            name="jira",
            transport=MCPTransport.http,
        )
        manager.registry = {"jira": server}
        manager.tool_name_to_mcp_server_name_mapping["jira-search_issues"] = "jira"
        manager.tool_name_to_mcp_server_name_mapping["search_issues"] = "jira"

        resolved = manager._resolve_mcp_server_for_tool_call("jira", "search_issues")
        assert resolved is server

    def test_resolve_mcp_server_for_tool_call_via_alias(self):
        """Resolution falls back to alias/server_name match in the registry."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="srv-uuid-123",
            name="zapier",
            alias="zapier-alias",
            transport=MCPTransport.http,
        )
        manager.registry = {"srv-uuid-123": server}
        manager.tool_name_to_mcp_server_name_mapping["create_zap"] = "zapier"

        resolved = manager._resolve_mcp_server_for_tool_call("zapier-alias", "create_zap")
        assert resolved is server

    def test_resolve_mcp_server_for_tool_call_unknown_tool_with_empty_mapping(self):
        """Server-name match alone must not let unknown tools through when the
        mapping has no entries for that server (e.g. listing has not completed
        or the server is OAuth2 and the user has not yet listed tools).
        """
        manager = MCPServerManager()
        server = MCPServer(
            server_id="srv-uuid-123",
            name="zapier",
            alias="zapier-alias",
            transport=MCPTransport.http,
        )
        manager.registry = {"srv-uuid-123": server}

        with pytest.raises(ValueError, match="Tool create_zap not found"):
            manager._resolve_mcp_server_for_tool_call("zapier-alias", "create_zap")

    def test_resolve_mcp_server_for_tool_call_fallback_to_unprefixed_lookup(self):
        """Fallback to unprefixed _get_mcp_server_from_tool_name when other paths fail."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="linear",
            name="linear",
            transport=MCPTransport.http,
        )
        manager.registry = {"linear": server}
        manager.tool_name_to_mcp_server_name_mapping["create_issue"] = "linear"

        # server_name is empty so the fallback unprefixed lookup runs and matches.
        resolved = manager._resolve_mcp_server_for_tool_call("", "create_issue")
        assert resolved is server

    def test_resolve_mcp_server_for_tool_call_raises_when_not_found(self):
        """ValueError is raised when no resolution path finds the tool."""
        manager = MCPServerManager()
        with pytest.raises(ValueError, match="Tool .* not found"):
            manager._resolve_mcp_server_for_tool_call("nonexistent", "ghost_tool")

    def test_resolve_mcp_server_for_tool_call_unknown_tool_with_known_server(self):
        """Server-name match alone must not let unknown tools slip through.

        If the registry has tools for this server but neither the prefixed nor
        unprefixed tool name is in the mapping, raise rather than returning the
        server (would otherwise allow tool enumeration via name spoofing).
        """
        manager = MCPServerManager()
        server = MCPServer(
            server_id="github",
            name="github",
            transport=MCPTransport.http,
        )
        manager.registry = {"github": server}
        # Mapping has *some* tools for github but not "missing_tool".
        manager.tool_name_to_mcp_server_name_mapping["github-list_repos"] = "github"
        manager.tool_name_to_mcp_server_name_mapping["list_repos"] = "github"

        with pytest.raises(ValueError, match="Tool missing_tool not found"):
            manager._resolve_mcp_server_for_tool_call("github", "missing_tool")

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_skipped_when_not_user_oauth(self):
        """Returns input headers unchanged when server does not need user OAuth."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="plain",
            name="plain",
            transport=MCPTransport.http,
        )
        # needs_user_oauth_token defaults to False.
        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="bob")

        result = await manager._resolve_oauth2_headers_for_tool_call(
            server, oauth2_headers=None, user_api_key_auth=user_auth
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_returns_client_supplied_token(self):
        """Returns the client's oauth2_headers as-is when already set."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="oauth-srv",
            name="oauth-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
        )
        assert server.needs_user_oauth_token is True
        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")
        supplied = {"Authorization": "Bearer client-supplied"}

        result = await manager._resolve_oauth2_headers_for_tool_call(
            server, oauth2_headers=supplied, user_api_key_auth=user_auth
        )
        assert result is supplied

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_looks_up_stored_token(self):
        """Falls back to stored per-user OAuth headers for a non-migrated (delegate) oauth2 server."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="oauth-srv",
            name="oauth-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,  # stays on v1, so v1 still builds the header
        )
        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")
        stored = {"Authorization": "Bearer stored-user-token"}

        with patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new=AsyncMock(return_value=stored),
        ) as mock_lookup:
            result = await manager._resolve_oauth2_headers_for_tool_call(
                server, oauth2_headers=None, user_api_key_auth=user_auth
            )

        assert result == stored
        mock_lookup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_steps_aside_for_migrated_server(self):
        """A migrated authorization_code server is owned by the v2 resolver, so v1 must not also
        build the token into extra_headers (which the v2 graft would defer to and shadow).
        """
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="oauth-srv",
            name="oauth-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,  # per-user, not delegate -> migrated to v2
        )
        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")

        with patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new=AsyncMock(return_value={"Authorization": "Bearer should-not-be-used"}),
        ) as mock_lookup:
            result = await manager._resolve_oauth2_headers_for_tool_call(
                server, oauth2_headers=None, user_api_key_auth=user_auth
            )

        assert result is None  # stepped aside; the v2 resolver handles the token
        mock_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_swallows_lookup_exception(self):
        """Returns supplied headers (None) when the v1 stored-token lookup raises (delegate path)."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="oauth-srv",
            name="oauth-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,  # non-migrated, so it reaches the v1 lookup
        )
        user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")

        with patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new=AsyncMock(side_effect=RuntimeError("redis down")),
        ):
            result = await manager._resolve_oauth2_headers_for_tool_call(
                server, oauth2_headers=None, user_api_key_auth=user_auth
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_has_user_oauth_token_delegates_to_provider(self):
        """has_user_oauth_token maps the server and delegates the verdict to the v2 resolver."""
        from litellm.proxy._types import UserAPIKeyAuth

        for verdict in (True, False):

            class _Provider:
                async def has_user_token(self, subject, spec):
                    return verdict

            manager = MCPServerManager(cred_provider=_Provider())
            server = MCPServer(
                server_id="s",
                name="n",
                transport=MCPTransport.http,
                auth_type=MCPAuth.oauth2,
            )
            user_auth = UserAPIKeyAuth(api_key="sk", user_id="alice")
            assert await manager.has_user_oauth_token(server, user_auth) is verdict

    @pytest.mark.asyncio
    async def test_has_user_oauth_token_short_circuits_for_unmigrated_server(self):
        """A server the resolver does not own (None spec, e.g. delegate) is False without a call."""
        from litellm.proxy._types import UserAPIKeyAuth

        calls: list = []

        class _Provider:
            async def has_user_token(self, subject, spec):
                calls.append(spec)
                return True

        manager = MCPServerManager(cred_provider=_Provider())
        server = MCPServer(
            server_id="s",
            name="n",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
        )
        user_auth = UserAPIKeyAuth(api_key="sk", user_id="alice")
        assert await manager.has_user_oauth_token(server, user_auth) is False
        assert calls == []  # short-circuited on the None spec, never hit the resolver

    @pytest.mark.asyncio
    async def test_invalidate_user_oauth_token_cache_delegates_to_store(self):
        """The write side's cache drop reaches the same per-user store the resolver reads."""

        class _Store:
            def __init__(self) -> None:
                self.invalidations: list[tuple[str, str]] = []

            async def fetch(self, user_id: str, server_id: str):
                return None

            async def invalidate(self, user_id: str, server_id: str) -> None:
                self.invalidations.append((user_id, server_id))

        store = _Store()
        manager = MCPServerManager(per_user_oauth_token_store=store)
        await manager.invalidate_user_oauth_token_cache("alice", "srv-1")
        assert store.invalidations == [("alice", "srv-1")]

    @pytest.mark.asyncio
    async def test_invalidate_user_oauth_token_cache_drops_legacy_cache_too(self):
        """A per-user token can be served from the legacy per-user token cache as well as the v2
        store; the shared invalidation must evict both, or the path not evicted keeps serving a
        token minted for a replaced credential row until its TTL."""

        class _Store:
            async def fetch(self, user_id: str, server_id: str):
                return None

            async def invalidate(self, user_id: str, server_id: str) -> None:
                return None

        class _LegacyCache:
            def __init__(self) -> None:
                self.deletes: list[tuple[str, str]] = []

            async def delete(self, user_id: str, server_id: str) -> None:
                self.deletes.append((user_id, server_id))

        legacy_cache = _LegacyCache()
        manager = MCPServerManager(per_user_oauth_token_store=_Store(), per_user_token_cache=legacy_cache)
        await manager.invalidate_user_oauth_token_cache("alice", "srv-1")
        assert legacy_cache.deletes == [("alice", "srv-1")]

    @pytest.mark.asyncio
    async def test_invalidate_user_oauth_token_cache_swallows_store_errors(self):
        """A cache-drop failure must not fail the credential write that triggered it, and the
        legacy cache must still be evicted after the v2 store drop fails."""

        class _Store:
            async def fetch(self, user_id: str, server_id: str):
                return None

            async def invalidate(self, user_id: str, server_id: str) -> None:
                raise RuntimeError("redis down")

        class _LegacyCache:
            def __init__(self) -> None:
                self.deletes: list[tuple[str, str]] = []

            async def delete(self, user_id: str, server_id: str) -> None:
                self.deletes.append((user_id, server_id))

        legacy_cache = _LegacyCache()
        manager = MCPServerManager(per_user_oauth_token_store=_Store(), per_user_token_cache=legacy_cache)
        await manager.invalidate_user_oauth_token_cache("alice", "srv-1")
        assert legacy_cache.deletes == [("alice", "srv-1")]

    @pytest.mark.asyncio
    async def test_invalidate_user_oauth_token_cache_swallows_legacy_cache_errors(self):
        """The legacy cache drop is best-effort like the v2 drop: a failure must be logged, never
        raised into the credential write that triggered the invalidation."""

        class _Store:
            async def fetch(self, user_id: str, server_id: str):
                return None

            async def invalidate(self, user_id: str, server_id: str) -> None:
                return None

        class _RaisingLegacyCache:
            async def delete(self, user_id: str, server_id: str) -> None:
                raise RuntimeError("redis down")

        manager = MCPServerManager(per_user_oauth_token_store=_Store(), per_user_token_cache=_RaisingLegacyCache())
        await manager.invalidate_user_oauth_token_cache("alice", "srv-1")

    @pytest.mark.asyncio
    async def test_resolve_oauth2_headers_no_user_id(self):
        """Skip lookup entirely when user_api_key_auth has no user_id."""
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()
        server = MCPServer(
            server_id="oauth-srv",
            name="oauth-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
        )
        # user_id is None -> lookup must not happen
        user_auth = UserAPIKeyAuth(api_key="sk-test")

        with patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new=AsyncMock(return_value={"Authorization": "Bearer x"}),
        ) as mock_lookup:
            result = await manager._resolve_oauth2_headers_for_tool_call(
                server, oauth2_headers=None, user_api_key_auth=user_auth
            )
        assert result is None
        mock_lookup.assert_not_called()

    def test_create_prefixed_tools_updates_mapping_for_both_forms(self):
        """_create_prefixed_tools should populate mapping for prefixed and original names even when not adding prefix in output."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="jira",
            name="jira",
            transport=MCPTransport.http,
        )

        # Input tools as would come from upstream
        t1 = MCPTool(
            name="create_issue",
            description="",
            inputSchema={},
        )
        t2 = MCPTool(
            name="close_issue",
            description="",
            inputSchema={},
        )

        # Do not add prefix in returned objects
        out_tools = manager._create_prefixed_tools([t1, t2], server, add_prefix=False)

        # Returned names should be unprefixed
        names = sorted([t.name for t in out_tools])
        assert names == ["close_issue", "create_issue"]

        # Mapping should include both original and prefixed names -> resolves calls either way
        assert manager.tool_name_to_mcp_server_name_mapping["create_issue"] == "jira"
        assert manager.tool_name_to_mcp_server_name_mapping["jira-create_issue"] == "jira"
        assert manager.tool_name_to_mcp_server_name_mapping["close_issue"] == "jira"
        assert manager.tool_name_to_mcp_server_name_mapping["jira-close_issue"] == "jira"

    def test_get_mcp_server_from_tool_name_with_prefixed_and_unprefixed(self):
        """After mapping is populated, manager resolves both prefixed and unprefixed tool names to the same server."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="zapier",
            name="zapier",
            server_name="zapier",
            transport=MCPTransport.http,
        )

        # Register server so resolution can find it
        manager.registry = {server.server_id: server}

        # Populate mapping (add_prefix value doesn't matter for mapping population)
        base_tool = MCPTool(
            name="create_zap",
            description="",
            inputSchema={},
        )
        _ = manager._create_prefixed_tools([base_tool], server, add_prefix=False)

        # Unprefixed resolution
        resolved_server_unpref = manager._get_mcp_server_from_tool_name("create_zap")
        assert resolved_server_unpref is not None
        assert resolved_server_unpref.server_id == server.server_id

        # Prefixed resolution
        resolved_server_pref = manager._get_mcp_server_from_tool_name("zapier-create_zap")
        assert resolved_server_pref is not None
        assert resolved_server_pref.server_id == server.server_id

    @pytest.mark.asyncio
    async def test_rest_endpoint_filters_by_allowed_tools(self):
        """Test that REST endpoint _get_tools_for_single_server respects allowed_tools configuration"""
        from litellm.proxy._experimental.mcp_server.rest_endpoints import (
            _get_tools_for_single_server,
        )

        # Create server with allowed_tools configured
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            allowed_tools=["allowed_tool_1", "allowed_tool_2"],
        )
        server.mcp_info = {"server_name": "test-server"}

        # Mock tools returned from manager (3 tools, but only 2 are allowed)
        tool1 = MagicMock()
        tool1.name = "allowed_tool_1"
        tool1.description = "This tool is allowed"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "blocked_tool"
        tool2.description = "This tool is not allowed"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "allowed_tool_2"
        tool3.description = "This tool is also allowed"
        tool3.inputSchema = {}

        # Mock the global_mcp_server_manager._get_tools_from_server
        from litellm.proxy._experimental.mcp_server import rest_endpoints

        with patch.object(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            new=AsyncMock(return_value=[tool1, tool2, tool3]),
        ):
            # Call the REST endpoint helper
            filtered_response = await _get_tools_for_single_server(server, server_auth_header=None)

            # Verify only allowed tools are in the response
            assert len(filtered_response) == 2
            tool_names = [t.name for t in filtered_response]
            assert "allowed_tool_1" in tool_names
            assert "allowed_tool_2" in tool_names
            assert "blocked_tool" not in tool_names

    @pytest.mark.asyncio
    async def test_rest_endpoint_shows_all_when_allowed_tools_is_none(self):
        """Test that REST endpoint shows all tools when allowed_tools is None (backwards compatibility)"""
        from litellm.proxy._experimental.mcp_server.rest_endpoints import (
            _get_tools_for_single_server,
        )

        # Create server with allowed_tools as None
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            allowed_tools=None,  # No filtering
        )
        server.mcp_info = {"server_name": "test-server"}

        # Mock tools returned from manager
        tool1 = MagicMock()
        tool1.name = "tool_1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool_2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool_3"
        tool3.description = "Tool 3"
        tool3.inputSchema = {}

        # Mock the global_mcp_server_manager._get_tools_from_server
        from litellm.proxy._experimental.mcp_server import rest_endpoints

        with patch.object(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            new=AsyncMock(return_value=[tool1, tool2, tool3]),
        ):
            # Call the REST endpoint helper
            all_tools_response = await _get_tools_for_single_server(server, server_auth_header=None)

            # Verify all tools are returned (no filtering)
            assert len(all_tools_response) == 3
            tool_names = [t.name for t in all_tools_response]
            assert "tool_1" in tool_names
            assert "tool_2" in tool_names
            assert "tool_3" in tool_names

    @pytest.mark.asyncio
    async def test_rest_endpoint_shows_all_when_allowed_tools_is_empty_list(self):
        """Test that REST endpoint shows all tools when allowed_tools is empty list (backwards compatibility)"""
        from litellm.proxy._experimental.mcp_server.rest_endpoints import (
            _get_tools_for_single_server,
        )

        # Create server with allowed_tools as empty list
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            allowed_tools=[],  # Empty list means no filtering
        )
        server.mcp_info = {"server_name": "test-server"}

        # Mock tools returned from manager
        tool1 = MagicMock()
        tool1.name = "tool_1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool_2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        # Mock the global_mcp_server_manager._get_tools_from_server
        from litellm.proxy._experimental.mcp_server import rest_endpoints

        with patch.object(
            rest_endpoints.global_mcp_server_manager,
            "_get_tools_from_server",
            new=AsyncMock(return_value=[tool1, tool2]),
        ):
            # Call the REST endpoint helper
            all_tools_response = await _get_tools_for_single_server(server, server_auth_header=None)

            # Verify all tools are returned (no filtering)
            assert len(all_tools_response) == 2
            tool_names = [t.name for t in all_tools_response]
            assert "tool_1" in tool_names
            assert "tool_2" in tool_names

    async def test_add_db_mcp_server_to_registry(self):
        """Test that add_db_mcp_server_to_registry adds a MCP server to the registry"""
        manager = MCPServerManager()
        server = LiteLLM_MCPServerTable(
            **{
                "server_id": "4c679a81-acd9-4954-9f84-30b739362498",
                "server_name": "edc_mcp_server",
                "alias": "edc_mcp_server",
                "description": None,
                "url": "fake_mcp_url",
                "transport": "http",
                "auth_type": "none",
                "created_at": "2025-09-30T08:28:31.353000Z",
                "created_by": "a1248959",
                "updated_at": "2025-09-30T08:28:31.353000Z",
                "updated_by": "a1248959",
                "teams": [],
                "mcp_access_groups": [],
                "mcp_info": {
                    "server_name": "edc_mcp_server",
                    "mcp_server_cost_info": None,
                },
                "status": "unknown",
                "last_health_check": None,
                "health_check_error": None,
                "command": None,
                "args": [],
                "env": {},
            },
        )
        await manager.add_server(server)
        assert server.server_id in manager.get_registry()

    @pytest.mark.asyncio
    async def test_key_tool_permission_allows_permitted_tool(self):
        """
        Test that key can call tool when it's in mcp_tool_permissions allowed list.
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()

        server = MCPServer(
            server_id="test_server_123",
            name="Test Server",
            transport=MCPTransport.http,
            allowed_tools=None,
            disallowed_tools=None,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_123",
            mcp_tool_permissions={"test_server_123": ["read_wiki_structure"]},
        )

        user_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
        )

        proxy_logging = MagicMock()
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging.pre_call_hook = AsyncMock(return_value=None)

        # Should succeed
        await manager.pre_call_tool_check(
            server_name="Test Server",
            name="read_wiki_structure",
            arguments={"repoName": "facebook/react"},
            user_api_key_auth=user_auth,
            proxy_logging_obj=proxy_logging,
            server=server,
        )

    @pytest.mark.asyncio
    async def test_key_tool_permission_blocks_unpermitted_tool(self):
        """
        Test that key cannot call tool when it's NOT in mcp_tool_permissions allowed list.
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()

        server = MCPServer(
            server_id="test_server_123",
            name="Test Server",
            transport=MCPTransport.http,
            allowed_tools=None,
            disallowed_tools=None,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_123",
            mcp_tool_permissions={"test_server_123": ["read_wiki_structure"]},
        )

        user_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
        )

        proxy_logging = MagicMock()
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging.pre_call_hook = AsyncMock(return_value=None)

        # Should fail with 403
        with pytest.raises(HTTPException) as exc_info:
            await manager.pre_call_tool_check(
                server_name="Test Server",
                name="ask_question",
                arguments={"question": "test"},
                user_api_key_auth=user_auth,
                proxy_logging_obj=proxy_logging,
                server=server,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_check_tool_permission_for_key_team_allows_permitted_tool(self):
        """
        Test check_tool_permission_for_key_team directly - should allow permitted tool.
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()

        server = MCPServer(
            server_id="github_server",
            name="GitHub Server",
            transport=MCPTransport.http,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_456",
            mcp_tool_permissions={"github_server": ["read_repo", "list_issues"]},
        )

        user_auth = UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-456",
            object_permission=object_permission,
        )

        # Should not raise exception for allowed tool
        await manager.check_tool_permission_for_key_team(
            tool_name="read_repo",
            server=server,
            user_api_key_auth=user_auth,
        )

    @pytest.mark.asyncio
    async def test_check_tool_permission_for_key_team_blocks_unpermitted_tool(self):
        """
        Test check_tool_permission_for_key_team directly - should block unpermitted tool.
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()

        server = MCPServer(
            server_id="github_server",
            name="GitHub Server",
            transport=MCPTransport.http,
        )

        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_456",
            mcp_tool_permissions={"github_server": ["read_repo"]},
        )

        user_auth = UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-456",
            object_permission=object_permission,
        )

        # Should raise HTTPException for unpermitted tool
        with pytest.raises(HTTPException) as exc_info:
            await manager.check_tool_permission_for_key_team(
                tool_name="delete_repo",
                server=server,
                user_api_key_auth=user_auth,
            )

        assert exc_info.value.status_code == 403
        assert "delete_repo" in exc_info.value.detail["error"]
        assert "not allowed" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_check_tool_permission_for_key_team_allows_all_when_no_restrictions(
        self,
    ):
        """
        Test check_tool_permission_for_key_team - should allow all tools when no restrictions set.
        """
        from litellm.proxy._types import UserAPIKeyAuth

        manager = MCPServerManager()

        server = MCPServer(
            server_id="github_server",
            name="GitHub Server",
            transport=MCPTransport.http,
        )

        # No object_permission set on user_auth
        user_auth = UserAPIKeyAuth(
            api_key="sk-test-key",
            user_id="user-456",
            object_permission=None,
        )

        # Should allow any tool when no restrictions
        await manager.check_tool_permission_for_key_team(
            tool_name="any_tool",
            server=server,
            user_api_key_auth=user_auth,
        )

    @pytest.mark.asyncio
    async def test_allowed_tools_with_mixed_prefixed_and_unprefixed_names(self):
        """
        Test that allowed_tools works with both unprefixed and prefixed tool names.
        This tests the scenario where allowed_tools = ["getpetbyid", "my_api_mcp-findpetsbystatus"]
        Both getpetbyid (unprefixed) and findpetsbystatus (called unprefixed but allowed via prefix) should work.
        """
        manager = MCPServerManager()

        # Create server with mixed prefixed/unprefixed allowed_tools
        server = MCPServer(
            server_id="my_api_mcp",
            name="my_api_mcp",
            transport=MCPTransport.stdio,
            allowed_tools=["getpetbyid", "my_api_mcp-findpetsbystatus"],
            disallowed_tools=None,
        )

        # Mock dependencies - set object_permission and object_permission_id to None
        # so permission checks return None (no restrictions)
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None
        proxy_logging_obj = MagicMock()

        # Mock the async methods that pre_call_tool_check calls
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})

        # Test 1: Call getpetbyid (unprefixed in allowed_tools) - should succeed
        await manager.pre_call_tool_check(
            name="getpetbyid",
            arguments={"petId": "1"},
            server_name="my_api_mcp",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

        # Test 2: Call findpetsbystatus (prefixed in allowed_tools as "my_api_mcp-findpetsbystatus") - should succeed
        await manager.pre_call_tool_check(
            name="findpetsbystatus",
            arguments={"status": "available"},
            server_name="my_api_mcp",
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
            server=server,
        )

        # Test 3: Call a tool that's not in allowed_tools - should fail
        with pytest.raises(HTTPException) as exc_info:
            await manager.pre_call_tool_check(
                name="deletepet",
                arguments={"petId": "1"},
                server_name="my_api_mcp",
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging_obj,
                server=server,
            )

        assert exc_info.value.status_code == 403
        assert "Tool deletepet is not allowed for server my_api_mcp" in exc_info.value.detail["error"]
        assert "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_call_tool_without_broken_pipe_error(self):
        """
        Test that call_tool awaits the client call even without a persistent context manager.
        Ensures the gathered tasks still include the MCP client call result.
        """
        from unittest.mock import AsyncMock, MagicMock

        from mcp.types import CallToolResult

        manager = MCPServerManager()

        # Create a test server
        server = MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            url="http://test-server.com",
        )

        # Register the server and map a tool to it
        manager.registry = {"test-server": server}
        manager.tool_name_to_mcp_server_name_mapping["test_tool"] = "test-server"
        manager.tool_name_to_mcp_server_name_mapping["test-server-test_tool"] = "test-server"

        # Create mock client that tracks call_tool usage
        mock_client = AsyncMock()

        async def mock_call_tool(params, host_progress_callback=None):
            # Return a mock CallToolResult
            result = MagicMock(spec=CallToolResult)
            result.content = [{"type": "text", "text": "Tool executed successfully"}]
            result.isError = False
            return result

        mock_client.call_tool.side_effect = mock_call_tool

        # Mock _create_mcp_client to return our mock client
        manager._create_mcp_client = AsyncMock(return_value=mock_client)

        # Mock user auth with no restrictions
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None

        # Mock proxy logging
        proxy_logging_obj = MagicMock()
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(return_value={})
        proxy_logging_obj._convert_mcp_to_llm_format = MagicMock(return_value={})
        proxy_logging_obj.pre_call_hook = AsyncMock(return_value={})
        proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)

        # Call the tool
        result = await manager.call_tool(
            server_name="test-server",
            name="test_tool",
            arguments={"param": "value"},
            user_api_key_auth=user_api_key_auth,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Verify the result
        assert result is not None
        assert result.isError is False
        assert len(result.content) > 0

        # Verify the MCP client call was awaited exactly once
        assert mock_client.call_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_get_allowed_mcp_servers_with_user_api_key_auth(self):
        """
        Test that get_allowed_mcp_servers properly receives and uses user_api_key_auth
        when called. This verifies the fix where user_api_key_auth is passed through
        litellm_metadata from responses API.
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()

        # Create a mock user_api_key_auth with object_permission
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_123",
            mcp_servers=["test_server_1", "test_server_2"],
            mcp_access_groups=[],
        )

        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
            object_permission_id="perm_123",
        )

        # Mock MCPRequestHandler.get_allowed_mcp_servers to verify it receives user_api_key_auth
        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
        ) as mock_get_allowed:
            # Configure mock to return servers from object_permission
            mock_get_allowed.return_value = ["test_server_1", "test_server_2"]

            # Call get_allowed_mcp_servers with user_api_key_auth
            result = await manager.get_allowed_mcp_servers(user_api_key_auth)

            # Verify MCPRequestHandler.get_allowed_mcp_servers was called with user_api_key_auth
            mock_get_allowed.assert_called_once()
            call_args = mock_get_allowed.call_args
            assert call_args[0][0] is user_api_key_auth  # First positional arg should be user_api_key_auth
            assert call_args[0][0].user_id == "user-123"
            assert call_args[0][0].object_permission_id == "perm_123"
            assert call_args[0][0].object_permission is not None
            assert call_args[0][0].object_permission.mcp_servers == [
                "test_server_1",
                "test_server_2",
            ]

            # Verify result contains the expected servers
            assert "test_server_1" in result
            assert "test_server_2" in result

    @pytest.mark.asyncio
    async def test_no_mcp_servers_sentinel_blocks_allow_all_keys(self):
        """A key scoped to no-mcp-servers gets zero servers even when allow_all_keys
        servers exist, and the inner resolver is never consulted."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        manager = MCPServerManager()
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_no_mcp",
            mcp_servers=["no-mcp-servers"],
            mcp_access_groups=[],
        )
        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
            object_permission_id="perm_no_mcp",
        )

        with (
            patch.object(manager, "get_allow_all_keys_server_ids", return_value=["global-server"]),
            patch.object(
                MCPRequestHandler,
                "get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=["leaked-server"],
            ) as mock_inner,
        ):
            result = await manager.get_allowed_mcp_servers(user_api_key_auth)

        assert result == []
        mock_inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mcp_servers_sentinel_excludes_submitted_byom_servers(self):
        from litellm.proxy import proxy_server as proxy_server_module
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        class _Cache:
            async def async_get_cache(self, key: str):
                return ["submitted-server"]

        manager = MCPServerManager()
        manager.registry = {
            "submitted-server": MCPServer(
                server_id="submitted-server",
                name="submitted",
                transport=MCPTransport.http,
            )
        }
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_no_mcp",
            mcp_servers=["no-mcp-servers"],
            mcp_access_groups=[],
        )
        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
            object_permission_id="perm_no_mcp",
        )

        with (
            patch.object(proxy_server_module, "user_api_key_cache", _Cache()),
            patch.object(proxy_server_module, "prisma_client", None),
            patch.object(manager, "get_allow_all_keys_server_ids", return_value=["global-server"]),
            patch.object(
                MCPRequestHandler,
                "get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=["leaked-server"],
            ) as mock_inner,
        ):
            result = await manager.get_allowed_mcp_servers(user_api_key_auth)

        assert result == []
        mock_inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicitly_scoped_key_excludes_submitted_byom_servers(self):
        from litellm.proxy import proxy_server as proxy_server_module
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

        cache = MagicMock()
        cache.async_get_cache = AsyncMock(return_value=["submitted-server"])

        manager = MCPServerManager()
        manager.registry = {
            "submitted-server": MCPServer(
                server_id="submitted-server",
                name="submitted",
                transport=MCPTransport.http,
            ),
            "scoped-server": MCPServer(
                server_id="scoped-server",
                name="scoped",
                transport=MCPTransport.http,
            ),
        }
        object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_scoped",
            mcp_servers=["scoped-server"],
            mcp_access_groups=[],
        )
        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
            object_permission=object_permission,
            object_permission_id="perm_scoped",
        )

        with (
            patch.object(proxy_server_module, "user_api_key_cache", cache),
            patch.object(proxy_server_module, "prisma_client", None),
            patch.object(manager, "get_allow_all_keys_server_ids", return_value=[]),
            patch.object(
                MCPRequestHandler,
                "get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=["scoped-server"],
            ),
        ):
            result = await manager.get_allowed_mcp_servers(user_api_key_auth)

        assert result == ["scoped-server"]
        cache.async_get_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_toolset_scope_excludes_submitted_byom_servers(self):
        from litellm.proxy import proxy_server as proxy_server_module
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._experimental.mcp_server.mcp_context import (
            _mcp_active_toolset_id,
        )
        from litellm.proxy._types import UserAPIKeyAuth

        cache = MagicMock()
        cache.async_get_cache = AsyncMock(return_value=["submitted-server"])

        manager = MCPServerManager()
        manager.registry = {
            "submitted-server": MCPServer(
                server_id="submitted-server",
                name="submitted",
                transport=MCPTransport.http,
            ),
            "toolset-server": MCPServer(
                server_id="toolset-server",
                name="toolset",
                transport=MCPTransport.http,
            ),
        }
        user_api_key_auth = UserAPIKeyAuth(api_key="sk-test", user_id="user-123")

        token = _mcp_active_toolset_id.set("toolset-abc")
        try:
            with (
                patch.object(proxy_server_module, "user_api_key_cache", cache),
                patch.object(proxy_server_module, "prisma_client", None),
                patch.object(manager, "get_allow_all_keys_server_ids", return_value=["global-server"]),
                patch.object(
                    MCPRequestHandler,
                    "get_allowed_mcp_servers",
                    new_callable=AsyncMock,
                    return_value=["toolset-server"],
                ),
            ):
                result = await manager.get_allowed_mcp_servers(user_api_key_auth)
        finally:
            _mcp_active_toolset_id.reset(token)

        assert result == ["toolset-server"]

    @pytest.mark.asyncio
    async def test_invalidate_byom_submitted_servers_cache_deletes_key(self):
        from litellm.proxy import proxy_server as proxy_server_module

        cache = MagicMock()
        cache.async_delete_cache = AsyncMock()
        manager = MCPServerManager()

        with patch.object(proxy_server_module, "user_api_key_cache", cache):
            await manager.invalidate_byom_submitted_servers_cache("user-123")
            await manager.invalidate_byom_submitted_servers_cache(None)

        cache.async_delete_cache.assert_awaited_once_with(key="byom_submitted_servers:user-123")

    @pytest.mark.asyncio
    async def test_get_active_submitted_ids_cache_miss_queries_db_and_caches(self):
        from litellm.proxy import proxy_server as proxy_server_module
        from litellm.proxy._types import UserAPIKeyAuth

        cache = MagicMock()
        cache.async_get_cache = AsyncMock(return_value=None)
        cache.async_set_cache = AsyncMock()
        manager = MCPServerManager()
        manager.registry = {
            "submitted-server": MCPServer(
                server_id="submitted-server",
                name="submitted",
                transport=MCPTransport.http,
            )
        }
        user_api_key_auth = UserAPIKeyAuth(api_key="sk-test", user_id="user-123")

        with (
            patch.object(proxy_server_module, "user_api_key_cache", cache),
            patch.object(proxy_server_module, "prisma_client", MagicMock()),
            patch(
                "litellm.proxy._experimental.mcp_server.db.get_active_submitted_mcp_server_ids_for_user",
                AsyncMock(return_value=["submitted-server", "unknown-server"]),
            ),
        ):
            result = await manager._get_active_submitted_mcp_server_ids_for_user(user_api_key_auth)

        assert result == ["submitted-server"]
        cache.async_set_cache.assert_awaited_once_with(
            key="byom_submitted_servers:user-123",
            value=["submitted-server", "unknown-server"],
            ttl=60,
        )

    @pytest.mark.asyncio
    async def test_get_allowed_mcp_servers_fallback_keeps_submitted_byom_servers(self):
        from litellm.proxy import proxy_server as proxy_server_module
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._types import UserAPIKeyAuth

        class _Cache:
            async def async_get_cache(self, key: str):
                assert key == "byom_submitted_servers:user-123"
                return ["submitted-server"]

        manager = MCPServerManager()
        manager.registry = {
            "submitted-server": MCPServer(
                server_id="submitted-server",
                name="submitted",
                transport=MCPTransport.http,
            )
        }
        user_api_key_auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_id="user-123",
        )

        with (
            patch.object(proxy_server_module, "user_api_key_cache", _Cache()),
            patch.object(proxy_server_module, "prisma_client", None),
            patch.object(manager, "get_allow_all_keys_server_ids", return_value=["global-server"]),
            patch.object(
                MCPRequestHandler,
                "get_allowed_mcp_servers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("permission resolver failed"),
            ),
        ):
            result = await manager.get_allowed_mcp_servers(user_api_key_auth)

        assert set(result) == {"global-server", "submitted-server"}

    @pytest.mark.asyncio
    async def test_get_allowed_mcp_servers_anonymous_delegate_requires_oauth2(self):
        """Anonymous delegated auth listing should only include oauth2 servers."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        manager = MCPServerManager()
        oauth_delegate_server = MCPServer(
            server_id="oauth-delegate",
            name="oauth_delegate",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=True,
        )
        api_key_delegate_server = MCPServer(
            server_id="api-key-delegate",
            name="api_key_delegate",
            transport=MCPTransport.http,
            auth_type=MCPAuth.api_key,
            delegate_auth_to_upstream=True,
        )
        oauth_non_delegate_server = MCPServer(
            server_id="oauth-non-delegate",
            name="oauth_non_delegate",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            delegate_auth_to_upstream=False,
        )
        manager.registry = {
            oauth_delegate_server.server_id: oauth_delegate_server,
            api_key_delegate_server.server_id: api_key_delegate_server,
            oauth_non_delegate_server.server_id: oauth_non_delegate_server,
        }

        with patch.object(
            MCPRequestHandler,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await manager.get_allowed_mcp_servers(None)

        assert set(result) == {"oauth-delegate"}

    def test_get_mcp_server_from_tool_name_uses_server_name_not_name(self):
        """
        Test that _get_mcp_server_from_tool_name uses server.server_name instead of server.name
        when extracting server name from prefixed tool name (second case).
        This ensures the fix for using server_name instead of name works correctly.
        """
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
        )

        manager = MCPServerManager()

        # Create a server where server_name differs from name
        # This tests the scenario where server.name != server.server_name
        server = MCPServer(
            server_id="test-server-id",
            name="Test Server Name",  # Different from server_name
            server_name="test_server",  # This is what should be used
            alias="test_server",
            transport=MCPTransport.http,
        )

        # Register the server
        manager.registry = {server.server_id: server}

        # Create a tool with prefixed name
        tool_name = "test_tool"
        prefixed_tool_name = add_server_prefix_to_name(tool_name, "test_server")

        # Populate the mapping with the original tool name
        manager.tool_name_to_mcp_server_name_mapping[tool_name] = "test_server"
        manager.tool_name_to_mcp_server_name_mapping[prefixed_tool_name] = "test_server"

        # Test: _get_mcp_server_from_tool_name should find the server using server.server_name
        # even when server.name is different
        resolved_server = manager._get_mcp_server_from_tool_name(prefixed_tool_name)

        # Verify the server was found correctly
        assert resolved_server is not None
        assert resolved_server.server_id == server.server_id
        assert resolved_server.server_name == "test_server"
        # Verify it matched using server_name, not name
        assert resolved_server.name == "Test Server Name"  # name is different
        assert resolved_server.server_name == "test_server"  # server_name matches


class TestMCPServerTimestamps:
    """Regression tests: created_at/updated_at must be preserved, not overwritten with datetime.now()."""

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_preserves_timestamps(self):
        """build_mcp_server_from_table must carry created_at and updated_at into MCPServer."""
        manager = MCPServerManager()

        created = datetime(2024, 1, 15, 10, 0, 0)
        updated = datetime(2024, 6, 20, 12, 30, 0)

        table_record = LiteLLM_MCPServerTable(
            server_id="ts-server-1",
            server_name="ts_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            created_at=created,
            updated_at=updated,
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.created_at == created
        assert mcp_server.updated_at == updated

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_reads_token_endpoint_auth_method(self):
        """token_endpoint_auth_method stored in the credentials JSON is loaded onto the MCPServer (LIT-4091)."""
        manager = MCPServerManager()

        basic_record = LiteLLM_MCPServerTable(
            server_id="basic-db-1",
            server_name="basic_db",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            credentials={"token_endpoint_auth_method": "client_secret_basic"},
        )
        basic_server = await manager.build_mcp_server_from_table(basic_record, credentials_are_encrypted=False)
        assert basic_server.token_endpoint_auth_method == "client_secret_basic"

        default_record = LiteLLM_MCPServerTable(
            server_id="default-db-1",
            server_name="default_db",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            credentials={},
        )
        default_server = await manager.build_mcp_server_from_table(default_record, credentials_are_encrypted=False)
        assert default_server.token_endpoint_auth_method is None

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_discovers_obo_token_url_when_unset(self):
        """DB path: an OBO server with no token_exchange_endpoint in credentials and no token_url
        column runs discovery, and the resolved endpoint lands on the returned MCPServer."""
        manager = MCPServerManager()
        calls: list[bool] = []

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            calls.append(allow_origin_fallback)
            return MCPOAuthMetadata(
                scopes=None,
                authorization_url=None,
                token_url="https://discovered.example.com/token",
                registration_url=None,
            )

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        record = LiteLLM_MCPServerTable(
            server_id="obo-discover-db-1",
            server_name="obo_discover_db",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={"client_id": "cid", "client_secret": "csec"},
        )

        # prisma_client None -> the write-back no-ops; this test isolates the discovery behavior.
        with patch("litellm.proxy.proxy_server.prisma_client", None):
            server = await manager.build_mcp_server_from_table(record, credentials_are_encrypted=False)

        assert calls == [False]  # discovery ran once, origin fallback disabled for OBO
        assert server.token_url == "https://discovered.example.com/token"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_skips_discovery_when_obo_endpoint_configured(self):
        """DB path: a configured token_exchange_endpoint in the credentials JSON wins and skips
        discovery entirely, even though the token_url column is empty (the DB-specific lookup uses
        credentials_dict["token_exchange_endpoint"], not the column)."""
        manager = MCPServerManager()
        calls: list[str] = []

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            calls.append(server_url)
            raise AssertionError("discovery must not run when token_exchange_endpoint is configured")

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        record = LiteLLM_MCPServerTable(
            server_id="obo-configured-db-1",
            server_name="obo_configured_db",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={
                "client_id": "cid",
                "client_secret": "csec",
                "token_exchange_endpoint": "https://idp.example.com/token",
            },
        )

        server = await manager.build_mcp_server_from_table(record, credentials_are_encrypted=False)

        assert calls == []  # discovery never ran
        assert server.token_exchange_endpoint == "https://idp.example.com/token"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_persists_discovered_obo_token_url(self):
        """A DB-backed OBO server with no configured endpoint discovers token_url and must write it
        back to the row, so the next rebuild skips discovery instead of re-running it every time."""
        manager = MCPServerManager()

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            assert server_url == "https://example.com/mcp"
            assert allow_origin_fallback is False  # OBO never guesses the origin
            return MCPOAuthMetadata(
                scopes=None,
                authorization_url=None,
                token_url="https://discovered.example.com/token",
                registration_url=None,
            )

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        record = LiteLLM_MCPServerTable(
            server_id="obo-persist-1",
            server_name="obo_persist",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={"client_id": "cid", "client_secret": "csec", "audience": "aud"},
        )

        update_mock = AsyncMock()
        repo_instance = MagicMock()
        repo_instance.table.update = update_mock
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPServerRepository",
                return_value=repo_instance,
            ),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        ):
            server = await manager.build_mcp_server_from_table(record, credentials_are_encrypted=False)

        assert server.token_url == "https://discovered.example.com/token"
        update_mock.assert_awaited_once()
        assert update_mock.call_args.kwargs["where"] == {"server_id": "obo-persist-1"}
        assert update_mock.call_args.kwargs["data"] == {"token_url": "https://discovered.example.com/token"}

    @pytest.mark.asyncio
    async def test_persist_discovered_obo_token_url_skips_when_not_needed(self):
        """The write-back fires only for an OBO server that discovered a new endpoint: a row that
        already has token_url, a non-OBO auth_type, or a discovery that found nothing all no-op."""
        manager = MCPServerManager()
        update_mock = AsyncMock()
        repo_instance = MagicMock()
        repo_instance.table.update = update_mock

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPServerRepository",
                return_value=repo_instance,
            ),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        ):
            # already populated -> no write
            await manager._persist_discovered_obo_token_url(
                server_id="s",
                auth_type=MCPAuth.oauth2_token_exchange,
                existing_token_url="https://already.example.com/token",
                discovered_token_url="https://new.example.com/token",
            )
            # not an OBO server -> no write
            await manager._persist_discovered_obo_token_url(
                server_id="s",
                auth_type=MCPAuth.oauth2,
                existing_token_url=None,
                discovered_token_url="https://new.example.com/token",
            )
            # discovery found nothing -> no write
            await manager._persist_discovered_obo_token_url(
                server_id="s",
                auth_type=MCPAuth.oauth2_token_exchange,
                existing_token_url=None,
                discovered_token_url=None,
            )

        update_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_discovered_obo_token_url_is_best_effort(self):
        """A write-back failure must not propagate; discovery just re-runs on the next build."""
        manager = MCPServerManager()
        update_mock = AsyncMock(side_effect=Exception("db unavailable"))
        repo_instance = MagicMock()
        repo_instance.table.update = update_mock

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPServerRepository",
                return_value=repo_instance,
            ),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        ):
            await manager._persist_discovered_obo_token_url(
                server_id="s",
                auth_type=MCPAuth.oauth2_token_exchange,
                existing_token_url=None,
                discovered_token_url="https://new.example.com/token",
            )

        update_mock.assert_awaited_once()

    def test_build_mcp_server_table_preserves_timestamps(self):
        """_build_mcp_server_table must use the MCPServer's stored timestamps, not datetime.now()."""
        manager = MCPServerManager()

        created = datetime(2024, 1, 15, 10, 0, 0)
        updated = datetime(2024, 6, 20, 12, 30, 0)

        server = MCPServer(
            server_id="ts-server-2",
            name="ts_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            created_at=created,
            updated_at=updated,
        )

        table = manager._build_mcp_server_table(server)

        assert table.created_at == created
        assert table.updated_at == updated

    def test_build_mcp_server_table_none_timestamps_when_not_set(self):
        """_build_mcp_server_table must return None timestamps when not set on MCPServer."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="ts-server-3",
            name="ts_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )

        table = manager._build_mcp_server_table(server)

        assert table.created_at is None
        assert table.updated_at is None

    def test_build_mcp_server_table_preserves_tool_overrides(self):
        """Tool display/description overrides must survive registry -> API table conversion."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="override-server",
            name="deepwiki",
            server_name="deepwiki_mcp",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            tool_name_to_display_name={"read_wiki_structure": "browse_docs"},
            tool_name_to_description={"read_wiki_structure": "Browse repository documentation"},
        )

        table = manager._build_mcp_server_table(server)

        assert table.tool_name_to_display_name == {"read_wiki_structure": "browse_docs"}
        assert table.tool_name_to_description == {"read_wiki_structure": "Browse repository documentation"}

    @pytest.mark.asyncio
    async def test_round_trip_timestamps_preserved(self):
        """Timestamps survive the full round-trip: LiteLLM_MCPServerTable -> MCPServer -> LiteLLM_MCPServerTable."""
        manager = MCPServerManager()

        created = datetime(2023, 3, 10, 8, 0, 0)
        updated = datetime(2023, 9, 5, 16, 45, 0)

        table_record = LiteLLM_MCPServerTable(
            server_id="ts-server-4",
            server_name="ts_server_rt",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            created_at=created,
            updated_at=updated,
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)
        rebuilt_table = manager._build_mcp_server_table(mcp_server)

        assert rebuilt_table.created_at == created
        assert rebuilt_table.updated_at == updated

    def test_deserialize_json_list_normalizes_pydantic_models(self):
        """Prisma hydrates the ``env_vars`` JSON column into ``MCPEnvVar`` models;
        ``_deserialize_json_list`` must hand back plain dicts so ``MCPServer``
        (typed ``List[Dict[str, Any]]``) validates."""
        env_vars = [
            MCPEnvVar(name="GITHUB_TOKEN", scope=MCPEnvVarScope.user, description="PAT"),
            MCPEnvVar(name="REGION", value="us-east-1", scope=MCPEnvVarScope.global_),
        ]
        result = _deserialize_json_list(env_vars)
        assert result is not None
        assert all(isinstance(item, dict) for item in result)
        assert result[0]["name"] == "GITHUB_TOKEN"
        assert result[0]["scope"] == "user"
        assert result[1]["value"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_with_model_env_vars(self):
        """Regression: a DB row whose ``env_vars`` is a list of ``MCPEnvVar``
        models (as Prisma returns) must build into an ``MCPServer`` instead of
        raising a Pydantic ``dict_type`` validation error that silently drops
        the server from the registry."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="env-var-server-1",
            server_name="github_peruser",
            url="https://api.githubcopilot.com/mcp/",
            transport=MCPTransport.http,
            static_headers={"Authorization": "Bearer ${GITHUB_TOKEN}"},
            env_vars=[
                MCPEnvVar(
                    name="GITHUB_TOKEN",
                    scope=MCPEnvVarScope.user,
                    description="Your personal GitHub PAT",
                )
            ],
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.env_vars == [
            {
                "name": "GITHUB_TOKEN",
                "value": "",
                "scope": "user",
                "description": "Your personal GitHub PAT",
            }
        ]

    @pytest.mark.asyncio
    async def test_round_trip_source_url_preserved(self):
        """source_url survives the full round-trip: LiteLLM_MCPServerTable -> MCPServer -> LiteLLM_MCPServerTable.

        Regression test: the list endpoint (GET /v1/mcp/server) builds its
        response from the registry via this round-trip, so a dropped field
        here surfaces as a null source_url in the list response even though
        the value is stored in the DB.
        """
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="src-url-server",
            server_name="src_url_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            source_url="https://github.com/org/mcp-server",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)
        assert mcp_server.source_url == "https://github.com/org/mcp-server"

        rebuilt_table = manager._build_mcp_server_table(mcp_server)
        assert rebuilt_table.source_url == "https://github.com/org/mcp-server"

    @pytest.mark.asyncio
    async def test_round_trip_timeout_preserved(self):
        """timeout survives the full round-trip: LiteLLM_MCPServerTable -> MCPServer -> LiteLLM_MCPServerTable."""
        manager = MCPServerManager()
        table_record = LiteLLM_MCPServerTable(
            server_id="timeout-server",
            server_name="timeout_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            timeout=120.0,
        )
        mcp_server = await manager.build_mcp_server_from_table(table_record)
        assert mcp_server.timeout == 120.0

        rebuilt_table = manager._build_mcp_server_table(mcp_server)
        assert rebuilt_table.timeout == 120.0

    @pytest.mark.asyncio
    async def test_create_mcp_client_uses_server_timeout(self):
        """_create_mcp_client must pass server.timeout to MCPClient when set."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="timeout-client-server",
            name="timeout_client_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            timeout=180.0,
        )
        client = await manager._create_mcp_client(server)
        assert client.timeout == 180.0

    @pytest.mark.asyncio
    async def test_create_mcp_client_falls_back_to_global_timeout(self):
        """_create_mcp_client must fall back to MCP_CLIENT_TIMEOUT when server.timeout is None."""
        from litellm.constants import MCP_CLIENT_TIMEOUT

        manager = MCPServerManager()
        server = MCPServer(
            server_id="default-timeout-server",
            name="default_timeout_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )
        client = await manager._create_mcp_client(server)
        assert client.timeout == MCP_CLIENT_TIMEOUT

    @pytest.mark.asyncio
    async def test_create_mcp_client_zero_timeout_not_treated_as_falsy(self):
        """server.timeout=0.0 must be passed through, not fall back to MCP_CLIENT_TIMEOUT."""
        manager = MCPServerManager()
        server = MCPServer(
            server_id="zero-timeout-server",
            name="zero_timeout_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            timeout=0.0,
        )
        client = await manager._create_mcp_client(server)
        assert client.timeout == 0.0

    @pytest.mark.asyncio
    async def test_load_servers_from_config_preserves_timeout(self):
        """timeout from proxy config is loaded into MCPServer."""
        manager = MCPServerManager()
        config = {
            "my_server": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "timeout": 90.0,
            }
        }
        await manager.load_servers_from_config(config)
        servers = list(manager.config_mcp_servers.values())
        assert len(servers) == 1
        assert servers[0].timeout == 90.0

    @pytest.mark.asyncio
    async def test_call_regular_mcp_tool_timeout_returns_504(self):
        """When the MCP client call is cancelled (timeout), _call_regular_mcp_tool raises HTTPException 504."""
        from unittest.mock import AsyncMock, patch

        manager = MCPServerManager()

        async def _slow_call(*args, **kwargs):
            await asyncio.sleep(999)

        mock_client = AsyncMock()
        mock_client.call_tool = _slow_call

        server = MCPServer(
            server_id="timeout-tool-server",
            name="timeout_tool_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            timeout=0.01,
        )

        with patch.object(manager, "_create_mcp_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await manager._call_regular_mcp_tool(
                    mcp_server=server,
                    original_tool_name="some_tool",
                    arguments={},
                    tasks=[],
                    mcp_auth_header=None,
                    mcp_server_auth_headers=None,
                    oauth2_headers=None,
                    raw_headers=None,
                    proxy_logging_obj=None,
                )

        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["error"] == "timeout"
        assert "0.01s" in exc_info.value.detail["message"]


class TestMCPServerTokenExchangeColumns:
    """Token-exchange (RFC 8693) config persists through the dedicated columns added for the
    create/update REST + DB path, mirroring how ``token_url`` is stored. The credentials JSON
    blob is kept as a read-fallback so servers persisted before the columns existed still load."""

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_reads_token_exchange_columns(self):
        """The DB->runtime loader must read the three fields from the dedicated columns. Before the
        columns existed it only read the credentials blob, so column values would be dropped."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-cols",
            server_name="te_cols",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/oauth2/token",
            audience="https://upstream.example.com",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.token_exchange_endpoint == "https://idp.example.com/oauth2/token"
        assert mcp_server.audience == "https://upstream.example.com"
        assert mcp_server.subject_token_type == "urn:ietf:params:oauth:token-type:jwt"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_falls_back_to_credentials_blob(self):
        """Backwards compatibility: a server whose token-exchange config lives only in the
        credentials blob (no columns) must still load with those values."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-blob",
            server_name="te_blob",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={
                "token_exchange_endpoint": "https://idp.example.com/legacy/token",
                "audience": "legacy-audience",
                "subject_token_type": "urn:ietf:params:oauth:token-type:saml2",
            },
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.token_exchange_endpoint == "https://idp.example.com/legacy/token"
        assert mcp_server.audience == "legacy-audience"
        assert mcp_server.subject_token_type == "urn:ietf:params:oauth:token-type:saml2"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_subject_token_type_defaults(self):
        """subject_token_type falls back to the RFC 8693 access_token URN when unset."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-default",
            server_name="te_default",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/oauth2/token",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.subject_token_type == "urn:ietf:params:oauth:token-type:access_token"

    @pytest.mark.asyncio
    async def test_round_trip_token_exchange_columns_preserved(self):
        """The three fields survive LiteLLM_MCPServerTable -> MCPServer -> LiteLLM_MCPServerTable.
        Before the table builder wrote them back, a registry round-trip dropped them."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-rt",
            server_name="te_rt",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/oauth2/token",
            audience="https://upstream.example.com",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)
        rebuilt_table = manager._build_mcp_server_table(mcp_server)

        assert rebuilt_table.token_exchange_endpoint == "https://idp.example.com/oauth2/token"
        assert rebuilt_table.audience == "https://upstream.example.com"
        assert rebuilt_table.subject_token_type == "urn:ietf:params:oauth:token-type:jwt"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_reads_token_exchange_profile_column(self):
        """The profile dialect selector (rfc8693 vs entra_obo) is read from its dedicated column
        so a server created via the REST API/UI as entra_obo resolves to the Entra dialect."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-profile",
            server_name="te_profile",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
            token_exchange_profile="entra_obo",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.token_exchange_profile == "entra_obo"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_token_exchange_profile_defaults_rfc8693(self):
        """token_exchange_profile falls back to rfc8693 when neither column nor blob sets it."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-profile-default",
            server_name="te_profile_default",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/oauth2/token",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.token_exchange_profile == "rfc8693"

    @pytest.mark.asyncio
    async def test_build_mcp_server_from_table_token_exchange_profile_blob_fallback(self):
        """Backwards compatibility: a server with the profile only in the credentials blob still loads."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-profile-blob",
            server_name="te_profile_blob",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            credentials={"token_exchange_profile": "entra_obo"},
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)

        assert mcp_server.token_exchange_profile == "entra_obo"

    @pytest.mark.asyncio
    async def test_round_trip_token_exchange_profile_preserved(self):
        """token_exchange_profile survives LiteLLM_MCPServerTable -> MCPServer -> LiteLLM_MCPServerTable."""
        manager = MCPServerManager()

        table_record = LiteLLM_MCPServerTable(
            server_id="te-profile-rt",
            server_name="te_profile_rt",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_profile="entra_obo",
        )

        mcp_server = await manager.build_mcp_server_from_table(table_record)
        rebuilt_table = manager._build_mcp_server_table(mcp_server)

        assert rebuilt_table.token_exchange_profile == "entra_obo"


class TestInternalDelegatePkceWarningLog:
    @pytest.mark.asyncio
    async def test_build_mcp_server_logs_on_internal_delegate_interactive(self, caplog):
        caplog.set_level(logging.WARNING, logger="LiteLLM")
        manager = MCPServerManager()
        table_record = LiteLLM_MCPServerTable(
            server_id="warn-del-1",
            server_name="warn_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            authorization_url="https://idp.example.com/authorize",
            token_url="https://idp.example.com/token",
            available_on_public_internet=False,
            delegate_auth_to_upstream=True,
        )
        await manager.build_mcp_server_from_table(table_record)
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "internal-only" in combined
        assert "delegate_auth_to_upstream=true" in combined

    @pytest.mark.asyncio
    async def test_build_mcp_server_no_internal_delegate_log_when_public(self, caplog):
        caplog.set_level(logging.WARNING, logger="LiteLLM")
        manager = MCPServerManager()
        table_record = LiteLLM_MCPServerTable(
            server_id="warn-del-2",
            server_name="warn_server_pub",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            authorization_url="https://idp.example.com/authorize",
            token_url="https://idp.example.com/token",
            available_on_public_internet=True,
            delegate_auth_to_upstream=True,
        )
        await manager.build_mcp_server_from_table(table_record)
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "internal-only" not in combined

    def test_warn_skipped_for_client_credentials(self, caplog):
        caplog.set_level(logging.WARNING, logger="LiteLLM")
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            _warn_internal_delegate_pkce_if_applicable,
        )

        server = MCPServer(
            server_id="m2m-1",
            name="x",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            oauth2_flow="client_credentials",
            available_on_public_internet=False,
            delegate_auth_to_upstream=True,
        )
        _warn_internal_delegate_pkce_if_applicable(server, source="test")
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "internal-only" not in combined


class TestHasClientCredentialsOAuth2Flow:
    """
    Regression tests for the M2M auto-detection bug.

    Before the fix, has_client_credentials returned True whenever
    client_id + client_secret + token_url were all set, even for
    interactive OAuth setups (e.g. GitHub Enterprise). This silently
    dropped user tokens and fetched M2M tokens instead.

    The fix: M2M must be opted in explicitly via oauth2_flow="client_credentials".
    """

    def _make_server(self, **kwargs) -> MCPServer:
        return MCPServer(
            server_id="test-server",
            name="test-server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            url="https://github.example.com/mcp",
            **kwargs,
        )

    def test_all_three_fields_set_without_oauth2_flow_is_not_m2m(self):
        """
        GitHub Enterprise regression: client_id + client_secret + token_url
        should NOT trigger M2M flow unless oauth2_flow is explicitly set.
        """
        server = self._make_server(
            client_id="gh-client-id",
            client_secret="gh-client-secret",
            token_url="https://github.example.com/login/oauth/access_token",
        )
        assert server.has_client_credentials is False

    def test_explicit_client_credentials_flow_enables_m2m(self):
        """oauth2_flow='client_credentials' opts in to M2M."""
        server = self._make_server(
            client_id="svc-client-id",
            client_secret="svc-client-secret",
            token_url="https://idp.example.com/token",
            oauth2_flow="client_credentials",
        )
        assert server.has_client_credentials is True

    def test_explicit_authorization_code_flow_disables_m2m(self):
        """oauth2_flow='authorization_code' always returns False."""
        server = self._make_server(
            client_id="gh-client-id",
            client_secret="gh-client-secret",
            token_url="https://github.example.com/login/oauth/access_token",
            oauth2_flow="authorization_code",
        )
        assert server.has_client_credentials is False

    def test_no_fields_no_flow_is_not_m2m(self):
        """No credentials configured — not M2M."""
        server = self._make_server()
        assert server.has_client_credentials is False

    def test_partial_fields_without_flow_is_not_m2m(self):
        """Partial credential fields without explicit flow — not M2M."""
        server = self._make_server(
            client_id="only-client-id",
        )
        assert server.has_client_credentials is False

    def test_needs_user_oauth_token_true_without_explicit_m2m(self):
        """
        Without oauth2_flow='client_credentials', an oauth2 server with
        client fields set still needs a user OAuth token (interactive flow).
        """
        server = self._make_server(
            client_id="gh-client-id",
            client_secret="gh-client-secret",
            token_url="https://github.example.com/login/oauth/access_token",
        )
        assert server.needs_user_oauth_token is True

    def test_needs_user_oauth_token_false_with_explicit_m2m(self):
        """With oauth2_flow='client_credentials', no per-user token needed."""
        server = self._make_server(
            client_id="svc-client-id",
            client_secret="svc-client-secret",
            token_url="https://idp.example.com/token",
            oauth2_flow="client_credentials",
        )
        assert server.needs_user_oauth_token is False


# ---------------------------------------------------------------------------
# Upstream initialize-instructions cache
# ---------------------------------------------------------------------------


class TestMCPServerManagerUpstreamInstructionsCache:
    """Tests for the upstream initialize-instructions cache."""

    def test_get_returns_none_when_empty(self):
        """Empty cache returns None for any key."""
        manager = MCPServerManager()
        assert manager._upstream_initialize_instructions_by_server_id.get("nonexistent") is None

    def test_remember_stores_stripped_value(self):
        """_remember_upstream_initialize_instructions stores a stripped string."""
        manager = MCPServerManager()
        fake_server = MagicMock(server_id="srv")
        fake_client = MagicMock(_last_initialize_instructions="  hello \n")
        manager._remember_upstream_initialize_instructions(fake_server, fake_client)
        assert manager._upstream_initialize_instructions_by_server_id.get("srv") == "hello"

    def test_remember_ignores_empty_string(self):
        """Whitespace-only instructions are not stored."""
        manager = MCPServerManager()
        fake_server = MagicMock(server_id="srv")
        fake_client = MagicMock(_last_initialize_instructions="   ")
        manager._remember_upstream_initialize_instructions(fake_server, fake_client)
        assert manager._upstream_initialize_instructions_by_server_id.get("srv") is None

    def test_remember_ignores_none(self):
        """None instructions are not stored."""
        manager = MCPServerManager()
        fake_server = MagicMock(server_id="srv")
        fake_client = MagicMock(_last_initialize_instructions=None)
        manager._remember_upstream_initialize_instructions(fake_server, fake_client)
        assert manager._upstream_initialize_instructions_by_server_id.get("srv") is None

    @pytest.mark.asyncio
    async def test_load_servers_from_config_clears_cache(self):
        """Reloading config clears any previously cached upstream instructions."""
        manager = MCPServerManager()
        manager._upstream_initialize_instructions_by_server_id["old"] = "stale"
        await manager.load_servers_from_config(
            mcp_servers_config={
                "fresh_srv": {
                    "url": "https://example.com",
                    "instructions": "from yaml",
                }
            }
        )
        assert manager._upstream_initialize_instructions_by_server_id.get("old") is None

    @pytest.mark.asyncio
    async def test_load_servers_reads_instructions_from_config(self):
        """instructions field from YAML config is persisted on the MCPServer."""
        manager = MCPServerManager()
        await manager.load_servers_from_config(
            mcp_servers_config={
                "srv_a": {
                    "url": "https://a.example.com",
                    "instructions": "A instructions",
                },
                "srv_b": {
                    "url": "https://b.example.com",
                },
            }
        )
        by_name = {s.server_name: s for s in manager.config_mcp_servers.values()}
        assert "srv_a" in by_name and by_name["srv_a"].instructions == "A instructions"
        assert "srv_b" in by_name and by_name["srv_b"].instructions is None


class TestMCPServerManagerExpandPermissionList:
    """Tests for the alias/name-aware permission list expansion used by team MCP permissions."""

    def _make_server(
        self,
        server_id: str,
        server_name: str,
        alias=None,
        name=None,
    ) -> MCPServer:
        return MCPServer(
            server_id=server_id,
            name=name if name is not None else (alias or server_name),
            alias=alias,
            server_name=server_name,
            url=f"https://{server_id}.example.com",
            transport=MCPTransport.http,
        )

    def test_empty_list_returns_empty(self):
        manager = MCPServerManager()
        assert manager.expand_permission_list([]) == []

    def test_expands_server_name(self):
        manager = MCPServerManager()
        manager.config_mcp_servers["id-usw1"] = self._make_server("id-usw1", server_name="a")

        assert manager.expand_permission_list(["a"]) == ["id-usw1"]

    def test_expands_alias(self):
        manager = MCPServerManager()
        manager.config_mcp_servers["id-1"] = self._make_server(
            "id-1", server_name="internal_name", alias="public_alias"
        )

        assert manager.expand_permission_list(["public_alias"]) == ["id-1"]

    def test_passes_through_unknown_entry(self):
        """Unresolved entries pass through unchanged (with a debug log) —
        the downstream access check denies them when compared to the
        concrete request server_id."""
        manager = MCPServerManager()
        manager.config_mcp_servers["id-1"] = self._make_server("id-1", server_name="b")

        assert manager.expand_permission_list(["a"]) == ["a"]

    def test_name_collision_expands_to_all_matches(self):
        """Two servers sharing a server_name both resolve — the documented behavior."""
        manager = MCPServerManager()
        manager.config_mcp_servers["id-config"] = self._make_server("id-config", server_name="shared")
        manager.registry["id-db"] = self._make_server("id-db", server_name="shared")

        assert sorted(manager.expand_permission_list(["shared"])) == [
            "id-config",
            "id-db",
        ]

    def test_searches_config_and_registry_union(self):
        manager = MCPServerManager()
        manager.config_mcp_servers["cfg-id"] = self._make_server("cfg-id", server_name="a")
        manager.registry["reg-id"] = self._make_server("reg-id", server_name="b")

        assert manager.expand_permission_list(["a"]) == ["cfg-id"]
        assert manager.expand_permission_list(["b"]) == ["reg-id"]

    def test_id_match_takes_precedence_over_name_match(self):
        """
        If a permission entry matches a server_id directly, don't also add
        servers whose server_name happens to equal that id.
        """
        manager = MCPServerManager()
        manager.config_mcp_servers["id-1"] = self._make_server("id-1", server_name="other_name")
        manager.config_mcp_servers["id-2"] = self._make_server("id-2", server_name="id-1")

        assert manager.expand_permission_list(["id-1"]) == ["id-1"]

    def test_mixed_ids_and_names_in_same_list(self):
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-1"] = self._make_server("uuid-1", server_name="a")
        manager.config_mcp_servers["uuid-2"] = self._make_server("uuid-2", server_name="b")

        # ["uuid-1", "b"] -> uuid-1 passes through, "b" resolves to uuid-2
        assert sorted(manager.expand_permission_list(["uuid-1", "b"])) == [
            "uuid-1",
            "uuid-2",
        ]

    def test_deduplicates_overlapping_id_and_name_entries(self):
        """If a list references the same server by both id and name, return it once."""
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-1"] = self._make_server("uuid-1", server_name="a")

        assert manager.expand_permission_list(["uuid-1", "a"]) == ["uuid-1"]

    def test_simulates_cross_region_portability(self):
        """
        Same permission entry "a" resolves to different concrete IDs per region —
        the cross-region portability the customer is asking for.
        """
        usw1 = MCPServerManager()
        usw1.config_mcp_servers["hash-usw1"] = self._make_server("hash-usw1", server_name="a")

        usc1 = MCPServerManager()
        usc1.config_mcp_servers["hash-usc1"] = self._make_server("hash-usc1", server_name="a")

        assert usw1.expand_permission_list(["a"]) == ["hash-usw1"]
        assert usc1.expand_permission_list(["a"]) == ["hash-usc1"]


class TestMCPServerManagerExpandToolPermissions:
    """Tests for tool-permission dict rewriting — the privilege-escalation guard."""

    def _make_server(self, server_id: str, server_name: str, alias=None) -> MCPServer:
        return MCPServer(
            server_id=server_id,
            name=alias or server_name,
            alias=alias,
            server_name=server_name,
            url=f"https://{server_id}.example.com",
            transport=MCPTransport.http,
        )

    def test_empty_or_none_returns_empty_dict(self):
        manager = MCPServerManager()
        assert manager.expand_tool_permissions(None) == {}
        assert manager.expand_tool_permissions({}) == {}

    def test_rewrites_name_key_to_server_id(self):
        """Privilege-escalation guard: a name-based key must resolve to the
        concrete server_id, otherwise `.get(server_id)` misses and the tool
        restriction is silently dropped (caller treats None as allow-all)."""
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-a"] = self._make_server("uuid-a", server_name="my-alias")

        result = manager.expand_tool_permissions({"my-alias": ["read_file"]})
        assert result == {"uuid-a": ["read_file"]}

    def test_passes_through_existing_server_id_key(self):
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-a"] = self._make_server("uuid-a", server_name="alpha")

        result = manager.expand_tool_permissions({"uuid-a": ["read_file"]})
        assert result == {"uuid-a": ["read_file"]}

    def test_unresolved_key_passes_through_unchanged(self):
        """A stale id-keyed restriction (server since deleted, or just a
        test-fixture placeholder) must still apply when something looks it
        up by that same string — dropping would silently remove the
        restriction."""
        manager = MCPServerManager()

        result = manager.expand_tool_permissions({"stale-uuid": ["read_file"]})
        assert result == {"stale-uuid": ["read_file"]}

    def test_name_collision_unions_tool_lists(self):
        """Two servers sharing a server_name both match; their tool lists get
        the restriction (matches the list-expansion collision semantics)."""
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-1"] = self._make_server("uuid-1", server_name="shared")
        manager.registry["uuid-2"] = self._make_server("uuid-2", server_name="shared")

        result = manager.expand_tool_permissions({"shared": ["read_file"]})
        assert sorted(result.keys()) == ["uuid-1", "uuid-2"]
        assert result["uuid-1"] == ["read_file"]
        assert result["uuid-2"] == ["read_file"]

    def test_id_and_name_keys_pointing_at_same_server_union_tools(self):
        """If the admin writes both {"uuid-a": [...], "alias-a": [...]} and
        both refer to the same server, the tool lists are unioned rather
        than one overwriting the other."""
        manager = MCPServerManager()
        manager.config_mcp_servers["uuid-a"] = self._make_server("uuid-a", server_name="alias-a")

        result = manager.expand_tool_permissions({"uuid-a": ["read_file"], "alias-a": ["write_file"]})
        assert sorted(result["uuid-a"]) == ["read_file", "write_file"]


class TestOAuthDiscoverySSRFGuard:
    """SSRF guard for the OAuth metadata discovery follow-up fetches.

    The vulnerability: a malicious MCP server returns a ``WWW-Authenticate``
    header pointing at an attacker-chosen ``resource_metadata`` URL, then a
    PRM JSON whose ``authorization_servers[0]`` points at internal/loopback
    addresses, coercing the proxy into making blind GETs to cloud-metadata
    services, internal admin panels, or loopback debug endpoints.
    """

    @staticmethod
    def _patch_resolves(monkeypatch, mapping):
        """Patch ``socket.getaddrinfo`` for a deterministic SSRF-guard test.

        ``mapping`` is ``{hostname: [ip-string, ...]}``; unknown hosts raise
        ``gaierror`` (treated as "unresolvable" -> blocked by async_safe_get).
        """
        import socket as _socket

        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host not in mapping:
                raise _socket.gaierror(f"unknown host {host}")
            family = _socket.AF_INET
            return [(family, _socket.SOCK_STREAM, 0, "", (ip, port)) for ip in mapping[host]]

        monkeypatch.setattr(
            "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
            fake_getaddrinfo,
        )

    def test_same_authority_url_is_direct_fetch_eligible(self):
        # Same scheme + host + port skips DNS entirely — the well-known
        # endpoint construction in _attempt_well_known_discovery always
        # produces same-authority URLs against the admin's server_url.
        assert MCPServerManager._is_same_authority_metadata_url(
            "https://example.com/.well-known/oauth-protected-resource",
            "https://example.com/mcp",
        )

    def test_same_host_different_port_uses_safe_fetch_path(self):
        assert not MCPServerManager._is_same_authority_metadata_url(
            "https://example.com:9999/.well-known/oauth-protected-resource",
            "https://example.com/mcp",
        )

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.5",  # RFC1918
            "172.16.0.1",  # RFC1918
            "192.168.1.1",  # RFC1918
            "169.254.169.254",  # AWS / Azure / GCP IMDS
            "100.100.100.200",  # Alibaba Cloud metadata
            "0.0.0.0",  # unspecified
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fc00::1",  # IPv6 ULA
        ],
    )
    @pytest.mark.asyncio
    async def test_cross_origin_blocked_when_resolves_to_unsafe_ip(self, monkeypatch, ip):
        self._patch_resolves(monkeypatch, {"attacker.example.com": [ip]})
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await manager._fetch_single_authorization_server_metadata(
                "https://attacker.example.com",
                "https://legit-mcp.example.com/mcp",
            )

        assert result is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_origin_allowed_when_resolves_to_public_ip(self, monkeypatch):
        self._patch_resolves(monkeypatch, {"login.microsoftonline.com": ["20.190.151.7"]})
        manager = MCPServerManager()

        mock_response = MagicMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://login.microsoftonline.com/tenant/v2.0"],
            "scopes_supported": ["mcp.read"],
        }

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://login.microsoftonline.com/tenant/v2.0/.well-known/openid-configuration",
                "https://atlassian-mcp.example.com/mcp",
            )

        assert servers == ["https://login.microsoftonline.com/tenant/v2.0"]
        assert scopes == ["mcp.read"]
        mock_client.get.assert_awaited_once()
        assert mock_client.get.await_args.kwargs["follow_redirects"] is False
        assert mock_client.get.await_args.kwargs["headers"]["Host"] == "login.microsoftonline.com"

    @pytest.mark.asyncio
    async def test_cross_origin_blocked_when_unresolvable(self, monkeypatch):
        self._patch_resolves(monkeypatch, {})
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://nope.example.invalid/.well-known/oauth-authorization-server",
                "https://legit-mcp.example.com/mcp",
            )

        assert servers == []
        assert scopes is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_http_scheme_is_not_safe(self):
        manager = MCPServerManager()
        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "file:///etc/passwd",
                "https://example.com/mcp",
            )
            result = await manager._fetch_single_authorization_server_metadata(
                "gopher://example.com/",
                "https://example.com/mcp",
            )

        assert servers == []
        assert scopes is None
        assert result is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_dual_resolution_blocked_if_any_ip_unsafe(self, monkeypatch):
        # If the attacker controls a DNS record returning multiple A records,
        # one of which is private, async_safe_get rejects before any network call.
        self._patch_resolves(monkeypatch, {"dual-stack.example.com": ["8.8.8.8", "127.0.0.1"]})
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://dual-stack.example.com/.well-known/oauth-authorization-server",
                "https://legit-mcp.example.com/mcp",
            )

        assert servers == []
        assert scopes is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_oauth_metadata_refuses_unsafe_url(self, monkeypatch):
        # End-to-end: a malicious WWW-Authenticate redirecting to a loopback
        # resource_metadata URL must not produce a network call.
        self._patch_resolves(monkeypatch, {"attacker.example.com": ["127.0.0.1"]})
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://attacker.example.com/meta",
                "https://legit-mcp.example.com/mcp",
            )

        assert servers == []
        assert scopes is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_getaddrinfo_result_blocks_url(self, monkeypatch):
        # POSIX doesn't strictly forbid an empty success-list from getaddrinfo.
        # async_safe_get must fail closed rather than making a network call.
        monkeypatch.setattr(
            "litellm.litellm_core_utils.url_utils.socket.getaddrinfo",
            lambda *a, **k: [],
        )
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            servers, scopes = await manager._fetch_oauth_metadata_from_resource(
                "https://no-records.example.com/.well-known/oauth-authorization-server",
                "https://legit-mcp.example.com/mcp",
            )

        assert servers == []
        assert scopes is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_origin_redirect_is_revalidated(self, monkeypatch):
        self._patch_resolves(
            monkeypatch,
            {
                "provider.example.com": ["8.8.8.8"],
                "127.0.0.1": ["127.0.0.1"],
            },
        )
        manager = MCPServerManager()

        redirect_response = MagicMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {"location": "http://127.0.0.1/admin"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=redirect_response)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await manager._fetch_single_authorization_server_metadata(
                "https://provider.example.com",
                "https://legit-mcp.example.com/mcp",
            )

        assert result is None
        assert mock_client.get.await_count == 3

    @pytest.mark.asyncio
    async def test_same_authority_fetch_does_not_follow_redirects(self):
        # Same-authority URLs may be internal admin-configured MCP servers, so
        # they are fetched directly. Redirects are still disabled because a
        # Location target would not inherit the same-authority guarantee.
        manager = MCPServerManager()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "authorization_servers": ["https://auth.example.com"],
        }
        mock_response.raise_for_status = MagicMock()

        captured_kwargs: Dict[str, Any] = {}

        async def fake_get(url, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            await manager._fetch_oauth_metadata_from_resource(
                "https://protected.example.com/.well-known/oauth",
                "https://protected.example.com/mcp",
            )
        assert captured_kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_same_authority_auth_server_fetch_does_not_follow_redirects(self):
        # Same redirect-bypass concern for the authorization-server fetch path.
        manager = MCPServerManager()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "authorization_endpoint": "https://provider.example.com/authorize",
            "token_endpoint": "https://provider.example.com/token",
        }
        mock_response.raise_for_status = MagicMock()

        captured_kwargs: Dict[str, Any] = {}

        async def fake_get(url, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            await manager._fetch_single_authorization_server_metadata(
                "https://provider.example.com",
                "https://provider.example.com",
            )
        assert captured_kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_fetch_authorization_server_refuses_unsafe_issuer(self, monkeypatch):
        # Mirrors the GHSA-mrfv repro: PRM lists a loopback issuer URL.
        self._patch_resolves(monkeypatch, {"attacker.example.com": ["127.0.0.1"]})
        manager = MCPServerManager()

        mock_client = MagicMock()
        mock_client.get = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await manager._fetch_single_authorization_server_metadata(
                "http://attacker.example.com:19999",
                "https://legit-mcp.example.com/mcp",
            )

        assert result is None
        mock_client.get.assert_not_called()


class TestApprovalStatusGate:
    """
    Regression tests for GHSA-gm4g-h72v-jhc3.

    The runtime registry must only contain servers an admin has approved.
    A non-admin can submit a pending stdio MCP server with an attacker-chosen
    command/args; before this gate, an admin opening the per-row endpoint
    triggered ``add_server`` + ``health_check_server``, which spawned the
    attacker's process under the proxy. The data-layer gate in
    ``add_server`` / ``update_server`` blocks pending and rejected rows
    from entering the registry regardless of which caller passes them in.
    """

    def _make_server(self, server_id: str, approval_status):
        return LiteLLM_MCPServerTable(
            server_id=server_id,
            alias=f"server_{server_id}",
            description="test",
            url=None,
            transport=MCPTransport.stdio,
            command="python",
            args=["-c", "print('attacker payload')"],
            env={},
            approval_status=approval_status,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.mark.parametrize(
        "approval_status,expect_in_registry",
        [
            (MCPApprovalStatus.pending_review, False),
            (MCPApprovalStatus.rejected, False),
            (MCPApprovalStatus.active, True),
            # Legacy rows: NULL predates the approval workflow; "approved" is
            # a legacy alias for "active" still present in older deployments.
            # Both must continue to load to match the DB-level filter in
            # reload_servers_from_database().
            (None, True),
            ("approved", True),
        ],
    )
    async def test_add_server_respects_approval_status(self, approval_status, expect_in_registry):
        manager = MCPServerManager()
        server_id = f"sid-{approval_status}"
        await manager.add_server(self._make_server(server_id, approval_status))
        assert (server_id in manager.registry) is expect_in_registry

    async def test_update_server_evicts_when_transitioned_away_from_active(self):
        # An admin updates a previously-active server to rejected (or pending).
        # The stale registry entry must be evicted so subsequent tool calls
        # and health probes can't reach it.
        manager = MCPServerManager()
        await manager.add_server(self._make_server("evict-me", MCPApprovalStatus.active))
        assert "evict-me" in manager.registry

        await manager.update_server(self._make_server("evict-me", MCPApprovalStatus.rejected))
        assert "evict-me" not in manager.registry

    async def test_update_server_eviction_clears_openapi_routing_artifacts(self, tmp_path):
        """Rejecting a server must remove its OpenAPI tools and name mappings."""
        from litellm.proxy._experimental.mcp_server.tool_registry import (
            global_mcp_tool_registry,
        )
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
        )

        manager = MCPServerManager()
        await manager.add_server(self._make_server("evict-openapi", MCPApprovalStatus.active))
        assert "evict-openapi" in manager.registry

        server = manager.registry["evict-openapi"]
        server.spec_path = str(tmp_path / "unused.yaml")
        prefix = get_server_prefix(server)
        prefixed = add_server_prefix_to_name("demo_tool", prefix)

        async def _noop_handler(**kwargs):
            return None

        global_mcp_tool_registry.register_tool(
            name=prefixed,
            description="demo",
            input_schema={"type": "object"},
            handler=_noop_handler,
        )
        manager.tool_name_to_mcp_server_name_mapping["demo_tool"] = prefix
        manager.tool_name_to_mcp_server_name_mapping[prefixed] = prefix

        await manager.update_server(self._make_server("evict-openapi", MCPApprovalStatus.rejected))

        assert "evict-openapi" not in manager.registry
        assert prefixed not in global_mcp_tool_registry.tools
        assert "demo_tool" not in manager.tool_name_to_mcp_server_name_mapping
        assert prefixed not in manager.tool_name_to_mcp_server_name_mapping

    async def test_update_server_noop_for_unregistered_pending(self):
        # update_server called with a pending row that was never registered
        # should silently return without adding it. Locks in the early-return
        # so a future refactor can't accidentally route the pending row to
        # build_mcp_server_from_table.
        manager = MCPServerManager()
        await manager.update_server(self._make_server("never-seen", MCPApprovalStatus.pending_review))
        assert "never-seen" not in manager.registry


class TestRegistryTableConversionPreservesEnvVars:
    """The registry ``MCPServer`` -> ``LiteLLM_MCPServerTable`` conversions back
    the GET /v1/mcp/server list and health responses, which populate the admin
    edit form. When they dropped ``env_vars`` the form loaded an empty list and
    saving any edit silently wiped the stored vars, so ``${VAR}`` static headers
    were forwarded upstream un-interpolated.
    """

    @staticmethod
    def _server_with_env_vars() -> MCPServer:
        return MCPServer(
            server_id="env-vars-server",
            name="env_vars_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            static_headers={"X-Db-Url": "${DB_PROTOCOL}://${CORP_USER}@${DB_HOST}"},
            env_vars=[
                {
                    "name": "DB_PROTOCOL",
                    "value": "postgresql",
                    "scope": "global",
                    "description": None,
                },
                {
                    "name": "CORP_USER",
                    "value": "",
                    "scope": "user",
                    "description": "Your DB username",
                },
            ],
        )

    @staticmethod
    def _assert_env_vars_round_tripped(table: LiteLLM_MCPServerTable) -> None:
        assert table.env_vars is not None
        by_name = {entry.name: entry for entry in table.env_vars}
        assert set(by_name) == {"DB_PROTOCOL", "CORP_USER"}
        assert by_name["DB_PROTOCOL"].scope == MCPEnvVarScope.global_
        assert by_name["DB_PROTOCOL"].value == "postgresql"
        assert by_name["CORP_USER"].scope == MCPEnvVarScope.user
        assert by_name["CORP_USER"].description == "Your DB username"

    def test_build_mcp_server_table_preserves_env_vars(self):
        manager = MCPServerManager()
        table = manager._build_mcp_server_table(self._server_with_env_vars())
        self._assert_env_vars_round_tripped(table)

    @pytest.mark.asyncio
    async def test_health_check_server_preserves_env_vars(self):
        # OAuth2 without client credentials needs a per-user token, so the
        # health check is skipped (no network) and we exercise the table
        # construction path directly.
        manager = MCPServerManager()
        server = self._server_with_env_vars()
        assert server.requires_per_user_auth is True
        manager.registry[server.server_id] = server
        table = await manager.health_check_server(server.server_id)
        self._assert_env_vars_round_tripped(table)


class TestHealthCheckInterpolatesGlobalEnvVars:
    """The upstream probes (health check and the initialize-instructions
    prefetch) must substitute global ``${NAME}`` env vars into static headers
    before opening the connection. Forwarding the raw placeholder makes any
    server whose auth header is backed by a global env var fail authentication
    and flip to 'unhealthy', even though real tool calls (which do interpolate)
    keep working.
    """

    @staticmethod
    def _server() -> MCPServer:
        return MCPServer(
            server_id="global-env-server",
            name="global_env_server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.none,
            static_headers={"Authorization": "Bearer ${API_TOKEN}"},
            env_vars=[
                {
                    "name": "API_TOKEN",
                    "value": "secret-token",
                    "scope": "global",
                    "description": None,
                }
            ],
        )

    @staticmethod
    def _capture_headers(manager: MCPServerManager) -> Dict[str, Any]:
        captured: Dict[str, Any] = {}
        mock_client = AsyncMock()
        mock_client.run_with_session = AsyncMock(return_value="ok")

        async def _create(server, mcp_auth_header, extra_headers, stdio_env):
            captured["extra_headers"] = extra_headers
            return mock_client

        manager._create_mcp_client = AsyncMock(side_effect=_create)
        return captured

    @pytest.mark.asyncio
    async def test_health_check_interpolates_global_env_vars(self):
        manager = MCPServerManager()
        server = self._server()
        assert server.requires_per_user_auth is False
        manager.get_mcp_server_by_id = MagicMock(return_value=server)
        manager._remember_upstream_initialize_instructions = MagicMock()
        captured = self._capture_headers(manager)

        result = await manager.health_check_server(server.server_id)

        assert captured["extra_headers"] == {"Authorization": "Bearer secret-token"}
        assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_initialize_instructions_prefetch_interpolates_global_env_vars(self):
        manager = MCPServerManager()
        server = self._server()
        captured = self._capture_headers(manager)
        manager._remember_upstream_initialize_instructions = MagicMock()

        await manager._ensure_upstream_initialize_instructions_cached(server)

        assert captured["extra_headers"] == {"Authorization": "Bearer secret-token"}


class TestUserEnvVarsCacheEviction:
    """At capacity the per-user env var cache must shed a single oldest entry
    rather than wiping every entry, so a steady stream of distinct callers does
    not periodically stampede the DB by invalidating every still-valid value.
    """

    @staticmethod
    def _patch_cache(monkeypatch, max_size):
        from litellm.proxy._experimental.mcp_server import mcp_server_manager as m

        cache: Dict[Any, Any] = {}
        monkeypatch.setattr(m, "_user_env_vars_cache", cache)
        monkeypatch.setattr(m, "_USER_ENV_VARS_CACHE_MAX_SIZE", max_size)
        return m, cache

    def test_eviction_drops_single_oldest_entry_not_whole_cache(self, monkeypatch):
        m, cache = self._patch_cache(monkeypatch, max_size=3)

        for i in range(3):
            m._write_user_env_vars_cache(f"user{i}", "srv", {"V": str(i)})
        assert set(cache) == {("user0", "srv"), ("user1", "srv"), ("user2", "srv")}

        m._write_user_env_vars_cache("user3", "srv", {"V": "3"})

        assert len(cache) == 3
        assert ("user0", "srv") not in cache
        assert ("user3", "srv") in cache
        assert cache[("user1", "srv")][0] == {"V": "1"}

    def test_refreshing_existing_key_does_not_evict(self, monkeypatch):
        m, cache = self._patch_cache(monkeypatch, max_size=2)

        m._write_user_env_vars_cache("a", "srv", {"V": "1"})
        m._write_user_env_vars_cache("b", "srv", {"V": "2"})
        m._write_user_env_vars_cache("a", "srv", {"V": "1-new"})

        assert set(cache) == {("a", "srv"), ("b", "srv")}
        assert cache[("a", "srv")][0] == {"V": "1-new"}
        # The just-refreshed key must now sit at the tail so the next insert
        # evicts the genuinely older entry instead.
        m._write_user_env_vars_cache("c", "srv", {"V": "3"})
        assert ("b", "srv") not in cache
        assert ("a", "srv") in cache


class TestGetPublicMCPServers:
    """
    /public/mcp_hub strict-whitelist semantics — mirrors /public/model_hub
    and /public/agent_hub. Regression test for the PR #20607 OR-with-default
    behavior that made `litellm.public_mcp_servers` ignored by the hub.
    """

    def _make_server(self, server_id, available_on_public_internet=True):
        return MCPServer(
            server_id=server_id,
            name=server_id,
            server_name=server_id,
            transport=MCPTransport.http,
            available_on_public_internet=available_on_public_internet,
        )

    def _make_manager(self, servers):
        manager = MCPServerManager()
        for s in servers:
            manager.config_mcp_servers[s.server_id] = s
        return manager

    @patch("litellm.public_mcp_servers", None)
    def test_returns_empty_when_whitelist_is_none(self):
        """No /make_public call yet → hub returns nothing, regardless of
        per-server flags."""
        manager = self._make_manager(
            [
                self._make_server("a", available_on_public_internet=True),
                self._make_server("b", available_on_public_internet=True),
            ]
        )
        assert manager.get_public_mcp_servers() == []

    @patch("litellm.public_mcp_servers", [])
    def test_returns_empty_when_whitelist_is_empty(self):
        """Explicit empty whitelist → hub returns nothing."""
        manager = self._make_manager([self._make_server("a", available_on_public_internet=True)])
        assert manager.get_public_mcp_servers() == []

    @patch("litellm.public_mcp_servers", ["a"])
    def test_returns_only_whitelisted_when_flag_defaults_to_true(self):
        """
        Regression: prior to the fix, every server with
        available_on_public_internet=True (the default) leaked into the hub
        regardless of the whitelist. Whitelist must be authoritative.
        """
        manager = self._make_manager(
            [
                self._make_server("a", available_on_public_internet=True),
                self._make_server("b", available_on_public_internet=True),
            ]
        )
        result = manager.get_public_mcp_servers()
        assert [s.server_id for s in result] == ["a"]

    @patch("litellm.public_mcp_servers", ["a"])
    def test_does_not_leak_servers_via_internal_flag(self):
        """
        available_on_public_internet is an IP-gating flag, not a hub flag.
        A server with the flag True that is not in the whitelist must not
        appear in the hub.
        """
        manager = self._make_manager(
            [
                self._make_server("a", available_on_public_internet=False),
                self._make_server("b", available_on_public_internet=True),
            ]
        )
        result = manager.get_public_mcp_servers()
        assert [s.server_id for s in result] == ["a"]

    @patch("litellm.public_mcp_servers", ["does-not-exist"])
    def test_stale_whitelist_id_returns_empty(self):
        """Whitelist references an unknown server_id → no spurious results."""
        manager = self._make_manager([self._make_server("a", available_on_public_internet=True)])
        assert manager.get_public_mcp_servers() == []


class TestGetPublicMCPServersLegacyMode:
    """
    Legacy migration knob: litellm.public_mcp_hub_strict_whitelist=False
    preserves the pre-fix OR-with-default semantics for one release so
    operators that relied on the old behavior have a window to call
    /v1/mcp/make_public before /public/mcp_hub goes empty.
    """

    def _make_server(self, server_id, available_on_public_internet=True):
        return MCPServer(
            server_id=server_id,
            name=server_id,
            server_name=server_id,
            transport=MCPTransport.http,
            available_on_public_internet=available_on_public_internet,
        )

    def _make_manager(self, servers):
        manager = MCPServerManager()
        for s in servers:
            manager.config_mcp_servers[s.server_id] = s
        return manager

    @patch("litellm.public_mcp_hub_strict_whitelist", False)
    @patch("litellm.public_mcp_servers", None)
    def test_legacy_returns_default_flag_servers_when_whitelist_is_none(self):
        """Legacy mode + no whitelist → every server with the default
        available_on_public_internet=True appears (old behavior)."""
        manager = self._make_manager(
            [
                self._make_server("a", available_on_public_internet=True),
                self._make_server("b", available_on_public_internet=False),
            ]
        )
        result = manager.get_public_mcp_servers()
        assert [s.server_id for s in result] == ["a"]

    @patch("litellm.public_mcp_hub_strict_whitelist", False)
    @patch("litellm.public_mcp_servers", ["b"])
    def test_legacy_unions_whitelist_and_default_flag(self):
        """Legacy mode unions the whitelist with any
        available_on_public_internet=True server."""
        manager = self._make_manager(
            [
                self._make_server("a", available_on_public_internet=True),
                self._make_server("b", available_on_public_internet=False),
            ]
        )
        result = manager.get_public_mcp_servers()
        assert sorted(s.server_id for s in result) == ["a", "b"]


class TestCreateMcpClientV2Graft:
    """The PR4 v2-resolver graft in ``_create_mcp_client``.

    Migrated HTTP/SSE modes (``none`` plus the static ``api_key`` family) resolve through the
    injected ``UpstreamCredentialProvider`` into the ``resolved_auth`` slot; every other mode,
    and every stdio server, defers to v1's ``auth_value`` path unchanged.
    """

    def _http_server(self, **overrides: Any) -> MCPServer:
        base: Dict[str, Any] = dict(
            server_id="http-graft",
            name="graft_server",
            url="https://upstream.example.com/mcp",
            transport=MCPTransport.http,
        )
        base.update(overrides)
        return MCPServer(**base)

    async def test_none_mode_resolves_to_noop_auth(self):
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            NoOpAuth,
        )

        client = await MCPServerManager()._create_mcp_client(self._http_server(auth_type=None))

        assert isinstance(client._resolved_auth, NoOpAuth)
        assert client._mcp_auth_value is None

    @pytest.mark.parametrize(
        "auth_type, token, expected_name, expected_value",
        [
            (MCPAuth.api_key, "k-123", "X-API-Key", "k-123"),
            (MCPAuth.bearer_token, "b-123", "Authorization", "Bearer b-123"),
            (MCPAuth.token, "t-123", "Authorization", "token t-123"),
            (MCPAuth.authorization, "raw-123", "Authorization", "raw-123"),
        ],
    )
    async def test_static_family_emits_expected_header(self, auth_type, token, expected_name, expected_value):
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )

        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=auth_type, authentication_token=token)
        )

        assert isinstance(client._resolved_auth, StaticHeaderAuth)
        assert client._resolved_auth.header_name == expected_name
        assert client._resolved_auth._header_value.get_secret_value() == expected_value
        assert client._mcp_auth_value is None

    async def test_basic_mode_base64_encodes(self):
        import base64

        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            StaticHeaderAuth,
        )

        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=MCPAuth.basic, authentication_token="user:pass")
        )

        encoded = base64.b64encode(b"user:pass").decode()
        assert isinstance(client._resolved_auth, StaticHeaderAuth)
        assert client._resolved_auth.header_name == "Authorization"
        assert client._resolved_auth._header_value.get_secret_value() == f"Basic {encoded}"

    async def test_m2m_client_credentials_defers_to_v1(self):
        # M2M (oauth2 client_credentials) is not migrated: to_server_spec returns
        # None, so the graft sets no resolved auth and leaves v1 in charge (v1
        # performs the client_credentials grant itself - the static
        # authentication_token is never consumed for oauth2, so it does not flow
        # to _mcp_auth_value). Per-user oauth2 (authorization_code) is migrated to
        # v2 and is exercised separately.
        client = await MCPServerManager()._create_mcp_client(
            self._http_server(
                auth_type=MCPAuth.oauth2,
                oauth2_flow="client_credentials",
                authentication_token="legacy-token",
            )
        )

        assert client._resolved_auth is None
        assert client._mcp_auth_value is None

    async def test_static_token_missing_defers_to_v1(self):
        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=MCPAuth.api_key, authentication_token=None)
        )

        assert client._resolved_auth is None

    async def test_stdio_migrated_auth_type_still_defers_to_v1(self):
        client = await MCPServerManager()._create_mcp_client(
            MCPServer(
                server_id="stdio-graft",
                name="stdio_graft",
                transport=MCPTransport.stdio,
                command="node",
                args=["server.js"],
                auth_type=MCPAuth.api_key,
                authentication_token="k-stdio",
            )
        )

        assert client.transport_type == MCPTransport.stdio
        assert client._resolved_auth is None
        assert client._mcp_auth_value == "k-stdio"

    async def test_resolver_error_maps_to_http_exception(self):
        from litellm.proxy._experimental.mcp_server.outbound_credentials import Error
        from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
            CredError,
        )

        class _UnauthorizedProvider:
            async def resolve_credentials(self, subject, server):
                return Error(CredError.of_unauthorized("denied"))

        manager = MCPServerManager(cred_provider=_UnauthorizedProvider())

        with pytest.raises(HTTPException) as exc:
            await manager._create_mcp_client(self._http_server(auth_type=None))

        assert exc.value.status_code == 401

    async def test_per_request_override_defers_to_v1(self):
        # A per-request override (mcp_auth_header) must win over the shared static token,
        # exactly as v1 did, so a migrated static server defers to v1 when one is present.
        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=MCPAuth.bearer_token, authentication_token="shared-tok"),
            mcp_auth_header="caller-override",
        )

        assert client._resolved_auth is None
        assert client._mcp_auth_value == "caller-override"

    async def test_conflicting_extra_header_skips_resolved_auth_on_v2(self):
        # An Authorization already supplied via extra_headers (guardrail hook like the JWT
        # signer, static_headers, or a forwarded caller header) must win. The server stays on
        # the v2 path but skips resolved_auth, so nothing overwrites the inbound header.
        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=MCPAuth.bearer_token, authentication_token="shared-tok"),
            extra_headers={"Authorization": "Bearer hook-jwt"},
        )

        assert client._resolved_auth is None
        assert client._mcp_auth_value is None
        assert client._get_auth_headers()["Authorization"] == "Bearer hook-jwt"

    async def test_none_with_extra_header_stays_v2_without_clobbering(self):
        from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
            NoOpAuth,
        )

        # none resolves to NoOpAuth, which writes no header, so it cannot clobber an inbound
        # Authorization; it stays on the v2 path and the inbound header is preserved verbatim.
        client = await MCPServerManager()._create_mcp_client(
            self._http_server(auth_type=None),
            extra_headers={"Authorization": "Bearer hook-jwt"},
        )

        assert isinstance(client._resolved_auth, NoOpAuth)
        assert client._get_auth_headers()["Authorization"] == "Bearer hook-jwt"


def _upstream_status_error(status_code: int, challenge: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upstream.example/mcp")
    response = httpx.Response(
        status_code,
        headers={"WWW-Authenticate": challenge},
        request=request,
    )
    return httpx.HTTPStatusError("upstream rejected token", request=request, response=response)


class TestMCPToolsListAuthSurfacing:
    """Regression: MCP tools/list 401 auth failures must surface as MCPUpstreamAuthError.

    Previously a missing/expired per-user OAuth token, or an upstream 401 for any
    non-carveout auth_type, was swallowed to an empty tool list, so a single-server
    client saw a 200 with no tools instead of a 401 challenge. The listing helpers
    now raise MCPUpstreamAuthError on a 401 regardless of auth_type; the single-server
    routes turn it into a 401 + WWW-Authenticate while the aggregator absorbs it to an
    empty list. Only a 401 challenges; a 403 (forbidden) degrades like any other error.
    """

    @pytest.mark.asyncio
    async def test_fetch_tools_with_timeout_surfaces_upstream_401(self):
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )

        manager = MCPServerManager()
        challenge = 'Bearer resource_metadata="https://upstream.example/.well-known/oauth-protected-resource"'
        client = MagicMock()
        client.list_tools = AsyncMock(side_effect=_upstream_status_error(401, challenge))

        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await manager._fetch_tools_with_timeout(client, "static-key-server")

        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate == challenge
        assert exc_info.value.server_name == "static-key-server"

    @pytest.mark.asyncio
    async def test_fetch_tools_with_timeout_absorbs_upstream_403(self):
        """Only a 401 drives the re-auth challenge. A 403 (authenticated but
        forbidden, e.g. insufficient scope) is not a re-auth signal, so even
        with a WWW-Authenticate header it degrades to an empty list rather than
        surfacing a challenge."""
        manager = MCPServerManager()
        challenge = 'Bearer error="insufficient_scope", scope="read:tools"'
        client = MagicMock()
        client.list_tools = AsyncMock(side_effect=_upstream_status_error(403, challenge))

        assert await manager._fetch_tools_with_timeout(client, "forbidden-server") == []

    @pytest.mark.asyncio
    async def test_fetch_tools_with_timeout_returns_empty_on_non_auth_error(self):
        manager = MCPServerManager()
        client = MagicMock()
        client.list_tools = AsyncMock(side_effect=RuntimeError("upstream 500"))

        assert await manager._fetch_tools_with_timeout(client, "srv") == []

    @pytest.mark.asyncio
    async def test_get_tools_from_server_surfaces_unusable_user_token(self):
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )

        manager = MCPServerManager()
        server = MCPServer(server_id="oauth-srv", name="oauth-srv", transport=MCPTransport.http)
        challenge = 'Bearer resource_metadata="/.well-known/oauth-protected-resource/mcp/oauth-srv"'
        manager._create_mcp_client = AsyncMock(
            side_effect=HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": challenge},
            )
        )

        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await manager._get_tools_from_server(server)

        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate == challenge
        assert exc_info.value.server_name == "oauth-srv"

    @pytest.mark.asyncio
    async def test_get_tools_from_server_absorbs_non_challenge_http_error(self):
        """A non-auth HTTPException (500) stays absorbed so one misconfigured server cannot blank
        the listing; 401/403 are the challenge-class statuses routed to MCPUpstreamAuthError."""
        manager = MCPServerManager()
        server = MCPServer(server_id="stdio-srv", name="stdio-srv", transport=MCPTransport.http)
        manager._create_mcp_client = AsyncMock(
            side_effect=HTTPException(
                status_code=500,
                detail="MCP stdio command 'foo' is not in the allowlist",
            )
        )

        assert await manager._get_tools_from_server(server) == []

    @pytest.mark.asyncio
    async def test_get_tools_from_server_suppresses_upstream_challenge_for_dcr_bridge(self):
        """A dcr_bridge server must never relay the upstream's own WWW-Authenticate: it points
        clients at the upstream protected-resource metadata, which fails the RFC 9728 resource
        match against the gateway URL they dialed. Stripping it makes the single-server route
        fabricate the gateway well-known challenge, whose content is the bridge facade."""
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )
        from litellm.types.mcp import MCPAuth

        manager = MCPServerManager()
        bridge_server = MCPServer(
            server_id="bridge-srv",
            name="bridge-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.true_passthrough,
            dcr_bridge=True,
        )
        upstream_challenge = 'Bearer resource_metadata="https://upstream.example/.well-known/oauth-protected-resource"'
        client = MagicMock()
        client.list_tools = AsyncMock(side_effect=_upstream_status_error(401, upstream_challenge))
        manager._create_mcp_client = AsyncMock(return_value=client)

        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await manager._get_tools_from_server(bridge_server)

        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate is None
        assert exc_info.value.server_name == "bridge-srv"

    @pytest.mark.asyncio
    async def test_get_tools_from_server_suppresses_resolver_challenge_for_dcr_bridge(self):
        """The client-build-time HTTPException conversion path applies the same suppression."""
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )
        from litellm.types.mcp import MCPAuth

        manager = MCPServerManager()
        bridge_server = MCPServer(
            server_id="bridge-srv",
            name="bridge-srv",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth_delegate,
            dcr_bridge=True,
        )
        manager._create_mcp_client = AsyncMock(
            side_effect=HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": 'Bearer resource_metadata="https://upstream.example/prm"'},
            )
        )

        with pytest.raises(MCPUpstreamAuthError) as exc_info:
            await manager._get_tools_from_server(bridge_server)

        assert exc_info.value.status_code == 401
        assert exc_info.value.www_authenticate is None

    @pytest.mark.asyncio
    async def test_aggregate_list_tools_absorbs_unauthenticated_server(self):
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )

        manager = MCPServerManager()
        good = MCPServer(server_id="good", name="good", transport=MCPTransport.http)
        bad = MCPServer(server_id="bad", name="bad", transport=MCPTransport.http)
        manager.get_allowed_mcp_servers = AsyncMock(return_value=["good", "bad"])
        manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda server_id: {"good": good, "bad": bad}.get(server_id)
        )
        good_tool = MCPTool(name="good-do_thing", description="do thing", inputSchema={})

        async def fake_get_tools(server, **kwargs):
            if server.server_id == "bad":
                raise MCPUpstreamAuthError(
                    status_code=401,
                    www_authenticate='Bearer realm="x"',
                    server_name="bad",
                )
            return [good_tool]

        manager._get_tools_from_server = fake_get_tools

        result = await manager.list_tools()

        assert [t.name for t in result] == ["good-do_thing"]


def test_should_strip_caller_authorization_for_token_exchange():
    """OBO: the inbound bearer is the subject token (exchanged), never forwarded upstream raw."""
    server = MCPServer(
        server_id="te-strip",
        name="te-strip-server",
        url="https://up.example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret="csec",
    )
    assert _should_strip_caller_authorization(mcp_server=server, raw_headers=None, user_api_key_auth=None) is True


class _UpstreamAuthError(Exception):
    """Mimics a wrapped upstream 401 the way _extract_upstream_auth_failure detects it."""

    def __init__(self, status_code: int = 401) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = httpx.Response(status_code)


class _RetryFakeClient:
    """A fake MCPClient whose call_tool fails on the first attempt and (optionally) succeeds after."""

    def __init__(self, *, raises=None, result=None) -> None:
        from litellm.experimental_mcp_client.client import MCPClient

        self._raises = raises
        self._result = result
        self._MCPClient = MCPClient
        self.attempts = 0

    async def call_tool(self, params, host_progress_callback=None, raise_on_error=False):
        self.attempts += 1
        if self._raises is not None:
            if raise_on_error:
                raise self._raises
            return self._MCPClient.error_tool_result(self._raises)
        return self._result


def _obo_server() -> MCPServer:
    return MCPServer(
        server_id="obo-srv",
        name="obo",
        url="https://upstream.example/mcp",
        transport=MCPTransport.sse,
        auth_type=MCPAuth.oauth2_token_exchange,
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret="csec",
    )


class TestOBOCallToolRetry:
    """The token_exchange (OBO) tool-call path re-mints the exchanged token once on an upstream 401."""

    def _manager(self):
        manager = MCPServerManager()
        manager._cred_provider = MagicMock()
        manager._cred_provider.invalidate_credentials = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_upstream_401_invalidates_and_retries_once(self):
        manager = self._manager()
        success = CallToolResult(content=[], isError=False)
        first = _RetryFakeClient(raises=_UpstreamAuthError(401))
        retry = _RetryFakeClient(result=success)
        manager._create_mcp_client = AsyncMock(return_value=retry)

        result = await manager._obo_call_tool_with_retry(
            client=first,
            call_tool_params=MagicMock(),
            host_progress_callback=None,
            mcp_server=_obo_server(),
            server_auth_header=None,
            extra_headers=None,
            stdio_env=None,
            subject_token="caller-jwt",
            user_api_key_auth=None,
        )

        assert result is success
        manager._cred_provider.invalidate_credentials.assert_awaited_once()
        manager._create_mcp_client.assert_awaited_once()
        assert first.attempts == 1 and retry.attempts == 1

    @pytest.mark.asyncio
    async def test_non_auth_error_does_not_retry(self):
        manager = self._manager()
        first = _RetryFakeClient(raises=ValueError("tool blew up"))
        manager._create_mcp_client = AsyncMock()

        result = await manager._obo_call_tool_with_retry(
            client=first,
            call_tool_params=MagicMock(),
            host_progress_callback=None,
            mcp_server=_obo_server(),
            server_auth_header=None,
            extra_headers=None,
            stdio_env=None,
            subject_token="caller-jwt",
            user_api_key_auth=None,
        )

        assert result.isError is True
        manager._cred_provider.invalidate_credentials.assert_not_awaited()
        manager._create_mcp_client.assert_not_awaited()
        assert first.attempts == 1

    @pytest.mark.asyncio
    async def test_second_401_degrades_without_looping(self):
        manager = self._manager()
        first = _RetryFakeClient(raises=_UpstreamAuthError(401))
        # The retry client still fails; with raise_on_error defaulting False it returns isError.
        retry = _RetryFakeClient(raises=_UpstreamAuthError(401))
        manager._create_mcp_client = AsyncMock(return_value=retry)

        result = await manager._obo_call_tool_with_retry(
            client=first,
            call_tool_params=MagicMock(),
            host_progress_callback=None,
            mcp_server=_obo_server(),
            server_auth_header=None,
            extra_headers=None,
            stdio_env=None,
            subject_token="caller-jwt",
            user_api_key_auth=None,
        )

        assert result.isError is True
        manager._create_mcp_client.assert_awaited_once()
        assert first.attempts == 1 and retry.attempts == 1


class TestOBOConcurrencyLimit:
    """OBO (token_exchange) tool calls must honor the server's max_concurrent_requests.

    Regression: the token_exchange dispatch built its coroutine outside
    _limit_outbound_concurrency, so OBO calls skipped the per-server semaphore the
    non-OBO path enforces and a caller could exceed the admin-configured cap.
    """

    @pytest.mark.asyncio
    async def test_obo_dispatch_respects_max_concurrent_requests(self):
        max_concurrent = 2
        overflow = 3
        server = MCPServer(
            server_id="obo-concurrency",
            name="obo",
            url="https://upstream.example/mcp",
            transport=MCPTransport.sse,
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
            max_concurrent_requests=max_concurrent,
        )

        release = asyncio.Event()
        inflight = {"current": 0, "peak": 0}

        class _ConcurrencyRecordingClient:
            async def call_tool(self, params, host_progress_callback=None, raise_on_error=False):
                inflight["current"] += 1
                inflight["peak"] = max(inflight["peak"], inflight["current"])
                try:
                    await release.wait()
                finally:
                    inflight["current"] -= 1
                return CallToolResult(content=[], isError=False)

        manager = MCPServerManager()
        manager._create_mcp_client = AsyncMock(return_value=_ConcurrencyRecordingClient())

        async def _dispatch():
            return await manager._call_regular_mcp_tool(
                mcp_server=server,
                original_tool_name="do_thing",
                arguments={},
                tasks=[],
                mcp_auth_header=None,
                mcp_server_auth_headers=None,
                oauth2_headers={"Authorization": "Bearer subject-jwt"},
                raw_headers=None,
                proxy_logging_obj=None,
            )

        callers = [asyncio.create_task(_dispatch()) for _ in range(max_concurrent + overflow)]

        stable = 0
        previous = -1
        for _ in range(1000):
            await asyncio.sleep(0)
            current = inflight["current"]
            if current == previous:
                stable += 1
                if current > 0 and stable >= 10:
                    break
            else:
                stable = 0
                previous = current

        peak_while_blocked = inflight["peak"]
        release.set()
        results = await asyncio.gather(*callers)

        assert peak_while_blocked == max_concurrent
        assert inflight["current"] == 0
        assert all(result.isError is False for result in results)


class TestOBOEndpointDiscovery:
    """An oauth2_token_exchange server with no configured token endpoint discovers it (RFC 9728 ->
    RFC 8414) like the oauth2 flow does; an explicitly configured endpoint skips discovery."""

    @pytest.mark.parametrize(
        "auth_type, endpoint, token_url, expected",
        [
            (MCPAuth.oauth2_token_exchange, None, None, True),  # OBO, nothing configured -> discover
            (MCPAuth.oauth2_token_exchange, "https://idp/token", None, False),  # endpoint set -> skip
            (MCPAuth.oauth2_token_exchange, None, "https://idp/token", False),  # token_url set -> skip
            (MCPAuth.oauth2, None, None, False),  # not OBO
            (MCPAuth.none, None, None, False),
            (None, None, None, False),
        ],
    )
    def test_decision(self, auth_type, endpoint, token_url, expected):
        assert MCPServerManager._obo_needs_endpoint_discovery(auth_type, endpoint, token_url) is expected

    @pytest.mark.asyncio
    async def test_config_obo_without_endpoint_discovers_token_endpoint(self):
        manager = MCPServerManager()
        discovered = MCPOAuthMetadata(
            scopes=None,
            authorization_url=None,
            token_url="https://discovered.example.com/token",
            registration_url=None,
        )
        seen = []

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            seen.append((server_url, allow_origin_fallback))
            return discovered

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        await manager.load_servers_from_config(
            {
                "obo": {
                    "url": "https://example.com/mcp",
                    "transport": MCPTransport.http,
                    "auth_type": MCPAuth.oauth2_token_exchange,
                    "client_id": "cid",
                    "client_secret": "csec",
                }
            }
        )

        server = next(iter(manager.config_mcp_servers.values()))
        # OBO discovery runs, and never guesses the resource origin as the IdP (origin fallback off).
        assert seen == [("https://example.com/mcp", False)]
        # The discovered token endpoint lands on token_url, which _token_exchange_spec reads.
        assert server.token_url == "https://discovered.example.com/token"

    @pytest.mark.asyncio
    async def test_config_obo_with_configured_endpoint_skips_discovery(self):
        manager = MCPServerManager()

        async def fake_discovery(server_url: str, *, allow_origin_fallback: bool = True):
            raise AssertionError("discovery must not run when the endpoint is configured")

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        await manager.load_servers_from_config(
            {
                "obo": {
                    "url": "https://example.com/mcp",
                    "transport": MCPTransport.http,
                    "auth_type": MCPAuth.oauth2_token_exchange,
                    "token_exchange_endpoint": "https://configured.example.com/token",
                    "client_id": "cid",
                    "client_secret": "csec",
                }
            }
        )

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.token_exchange_endpoint == "https://configured.example.com/token"

    @pytest.mark.asyncio
    async def test_descovery_metadata_does_not_guess_origin_when_disallowed(self):
        # The authoritative RFC 9728 -> RFC 8414 chain stays, but with allow_origin_fallback=False the
        # resource origin is never assumed to be the IdP, so no token endpoint is invented.
        manager = MCPServerManager()
        server_url = "https://example.com/public/mcp"
        request = httpx.Request("GET", server_url)
        response_obj = httpx.Response(
            status_code=401, request=request, headers={"WWW-Authenticate": 'Bearer scope="read"'}
        )
        response_obj.raise_for_status = MagicMock(
            side_effect=lambda: (_ for _ in ()).throw(
                httpx.HTTPStatusError("unauthorized", request=request, response=response_obj)
            )
        )
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=response_obj)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
                return_value=mock_client,
            ),
            patch.object(manager, "_fetch_oauth_metadata_from_resource", AsyncMock(return_value=([], None))),
            patch.object(manager, "_attempt_well_known_discovery", AsyncMock(return_value=([], None))),
            patch.object(manager, "_fetch_authorization_server_metadata", AsyncMock()) as mock_fetch_auth,
        ):
            result = await manager._descovery_metadata(server_url, allow_origin_fallback=False)

        # No advertised AS -> with the guess disabled, the AS-metadata fetch is never attempted, so no
        # token endpoint is discovered (only the scopes parsed from the challenge survive).
        mock_fetch_auth.assert_not_awaited()
        assert result is None or result.token_url is None

    @pytest.mark.asyncio
    async def test_descovery_metadata_closes_stream_on_401_oauth_challenge(self):
        manager = MCPServerManager()
        server_url = "https://example.com/mcp"
        request = httpx.Request("GET", server_url)
        error_response = MagicMock()
        error_response.headers = {}

        streaming_response = MagicMock()
        streaming_response.status_code = 401
        streaming_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("unauthorized", request=request, response=error_response)
        )
        streaming_response.aclose = AsyncMock()

        mock_handler = MagicMock()
        mock_handler.get = AsyncMock(return_value=streaming_response)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
                return_value=mock_handler,
            ),
            patch.object(manager, "_attempt_well_known_discovery", AsyncMock(return_value=([], None))),
            patch.object(manager, "_fetch_oauth_metadata_from_resource", AsyncMock(return_value=([], None))),
            patch.object(manager, "_fetch_authorization_server_metadata", AsyncMock(return_value=None)),
        ):
            result = await manager._descovery_metadata(server_url)

        streaming_response.aclose.assert_awaited_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_descovery_metadata_returns_none_for_no_auth_sse_server(self):
        manager = MCPServerManager()
        server_url = "https://example.com/mcp"

        streaming_response = MagicMock()
        streaming_response.status_code = 200
        streaming_response.raise_for_status = MagicMock()
        streaming_response.aclose = AsyncMock()

        mock_handler = MagicMock()
        mock_handler.get = AsyncMock(return_value=streaming_response)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
                return_value=mock_handler,
            ),
            patch.object(manager, "_attempt_well_known_discovery", AsyncMock(return_value=([], None))),
            patch.object(manager, "_fetch_authorization_server_metadata", AsyncMock(return_value=None)),
        ):
            result = await manager._descovery_metadata(server_url)

        mock_handler.get.assert_awaited_once_with(server_url, stream=True)
        streaming_response.aclose.assert_awaited_once()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_preflight_challenge_carries_step_up_error_and_claims():
    """An Entra Conditional Access rejection must surface error=insufficient_claims plus the base64
    claims in the single-server 401 challenge, so an MSAL-family client can drive the step-up and retry."""
    import base64

    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
    from litellm.proxy._experimental.mcp_server.outbound_credentials.types import CredError

    claims = '{"access_token":{"acrs":{"essential":true,"value":"c1"}}}'

    class _FakeProvider:
        async def resolve_credentials(self, subject, server):
            return Error(CredError.of_unauthorized("step-up required", claims=claims))

    manager = MCPServerManager(cred_provider=_FakeProvider())
    server = MCPServer(
        server_id="te-ca",
        name="te-ca-server",
        url="https://up.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2_token_exchange,
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret="csec",
    )
    with pytest.raises(HTTPException) as exc_info:
        await manager.preflight_token_exchange(
            server=server,
            oauth2_headers={"Authorization": "Bearer subj"},
            user_api_key_auth=None,
        )
    assert exc_info.value.status_code == 401
    headers = exc_info.value.headers or {}
    www = headers.get("WWW-Authenticate") or headers.get("www-authenticate") or ""
    assert 'error="insufficient_claims"' in www
    assert base64.b64encode(claims.encode()).decode() in www
    assert "resource_metadata" in www


@pytest.mark.asyncio
async def test_aggregate_list_still_absorbs_step_up_challenged_server():
    """A step-up (claims-bearing) 401 from one server must not change the aggregate contract: the
    multi-server listing still absorbs it and returns the healthy servers' tools."""
    from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError

    manager = MCPServerManager()
    good = MCPServer(server_id="good", name="good", transport=MCPTransport.http)
    ca = MCPServer(server_id="ca", name="ca", transport=MCPTransport.http)
    manager.get_allowed_mcp_servers = AsyncMock(return_value=["good", "ca"])
    manager.get_mcp_server_by_id = MagicMock(side_effect=lambda server_id: {"good": good, "ca": ca}.get(server_id))
    good_tool = MCPTool(name="good-do_thing", description="do thing", inputSchema={})

    async def fake_get_tools(server, **kwargs):
        if server.server_id == "ca":
            raise MCPUpstreamAuthError(
                status_code=401,
                www_authenticate=('Bearer resource_metadata="/x", error="insufficient_claims", claims="eyJhIjoxfQ=="'),
                server_name="ca",
            )
        return [good_tool]

    manager._get_tools_from_server = fake_get_tools

    result = await manager.list_tools()

    assert [t.name for t in result] == ["good-do_thing"]


class TestDbBuildReadsOauth2FlowColumnVerbatim:
    """The DB build must not re-infer the flow from field shape: rows are stamped at
    write time and by the startup backfill, and a DCR-registered interactive server
    has the exact M2M shape (client creds + token_url, no persisted authorization_url)
    whenever discovery is unavailable. Inference survives only for config-loaded
    servers and the request-time backstop in _get_allowed_mcp_servers."""

    def _row(self, oauth2_flow):
        return LiteLLM_MCPServerTable(
            server_id="flow-column-row",
            alias="flow_column_row",
            description="",
            url="https://up.example.com/mcp",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
            oauth2_flow=oauth2_flow,
            token_url="https://idp.example.com/token",
            credentials={"client_id": "cid", "client_secret": "csec"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_null_flow_m2m_shape_row_is_not_inferred_m2m(self):
        manager = MCPServerManager()
        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            built = await manager.build_mcp_server_from_table(self._row(None), credentials_are_encrypted=False)

        assert built.oauth2_flow is None
        assert built.has_client_credentials is False
        assert built.needs_user_oauth_token is True

    @pytest.mark.asyncio
    async def test_explicit_flow_column_is_read_verbatim(self):
        manager = MCPServerManager()
        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            built = await manager.build_mcp_server_from_table(
                self._row("client_credentials"), credentials_are_encrypted=False
            )

        assert built.oauth2_flow == "client_credentials"
        assert built.has_client_credentials is True
        assert built.needs_user_oauth_token is False

    @pytest.mark.asyncio
    async def test_authorization_code_flow_column_is_read_verbatim(self):
        manager = MCPServerManager()
        with patch.object(manager, "_descovery_metadata", new=AsyncMock(return_value=None)):
            built = await manager.build_mcp_server_from_table(
                self._row("authorization_code"), credentials_are_encrypted=False
            )

        assert built.oauth2_flow == "authorization_code"
        assert built.has_client_credentials is False
        assert built.needs_user_oauth_token is True


class TestRequestTimeOauth2FlowBackstop:
    """The single request-time resolution helpers every security site shares:
    effective_oauth2_flow (the enum/boolean decision) and
    resolve_oauth2_flow_for_request (the egress object copy)."""

    def _oauth2_server(self, **overrides):
        base = dict(
            server_id="flow-server",
            name="flow_server",
            transport=MCPTransport.http,
            auth_type=MCPAuth.oauth2,
        )
        base.update(overrides)
        return MCPServer(**base)

    def test_effective_flow_stamped_values_returned_verbatim(self):
        assert (
            MCPServerManager.effective_oauth2_flow(self._oauth2_server(oauth2_flow="client_credentials"))
            == "client_credentials"
        )
        assert (
            MCPServerManager.effective_oauth2_flow(self._oauth2_server(oauth2_flow="authorization_code"))
            == "authorization_code"
        )

    def test_effective_flow_null_m2m_shape_resolves_client_credentials(self):
        server = self._oauth2_server(
            oauth2_flow=None,
            client_id="cid",
            client_secret="csecret",
            token_url="https://idp.example.com/token",
        )
        assert MCPServerManager.effective_oauth2_flow(server) == "client_credentials"

    def test_effective_flow_null_pure_pkce_resolves_none(self):
        assert MCPServerManager.effective_oauth2_flow(self._oauth2_server(oauth2_flow=None)) is None

    def test_resolve_for_request_stamped_row_is_unchanged_identity(self):
        server = self._oauth2_server(oauth2_flow="client_credentials")
        assert MCPServerManager.resolve_oauth2_flow_for_request(server) is server

    def test_resolve_for_request_null_pure_pkce_is_unchanged_identity(self):
        server = self._oauth2_server(oauth2_flow=None)
        assert MCPServerManager.resolve_oauth2_flow_for_request(server) is server

    def test_resolve_for_request_null_m2m_shape_copies_client_credentials(self, caplog):
        import logging

        server = self._oauth2_server(
            oauth2_flow=None,
            client_id="cid",
            client_secret="csecret",
            token_url="https://idp.example.com/token",
        )
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            resolved = MCPServerManager.resolve_oauth2_flow_for_request(server)

        assert resolved is not server
        assert resolved.oauth2_flow == "client_credentials"
        assert server.oauth2_flow is None  # original untouched
        # Finding 2: the warning must NOT promise the backfill will stamp this row.
        joined = " ".join(caplog.messages)
        assert "no persisted oauth2_flow" in joined
        assert "next proxy boot" not in joined
        assert "will NOT self-heal" in joined


def test_build_mcp_server_table_carries_oauth2_flow():
    """GET /v1/mcp/server (list and by-id) serves registry servers through this
    conversion; dropping oauth2_flow here blinds the dashboard to the persisted
    flow, so the edit page cannot prefill and M2M gating never activates."""
    manager = MCPServerManager()
    server = MCPServer(
        server_id="flow-table-server",
        name="flow_table_server",
        server_name="flow_table_server",
        alias="flow_table_server",
        url="https://up.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
    )

    table = manager._build_mcp_server_table(server)

    assert table.oauth2_flow == "client_credentials"


def test_build_mcp_server_table_carries_null_oauth2_flow():
    """A legacy row the backfill left unstamped must surface as oauth2_flow=None in
    the GET response, so the dashboard maps it to undefined and prompts the admin to
    choose a flow rather than showing a guessed default."""
    manager = MCPServerManager()
    server = MCPServer(
        server_id="null-flow-server",
        name="null_flow_server",
        server_name="null_flow_server",
        alias="null_flow_server",
        url="https://up.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow=None,
    )

    table = manager._build_mcp_server_table(server)

    assert table.oauth2_flow is None
