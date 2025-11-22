import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../../../")

import httpx

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _deserialize_json_dict,
)
from litellm.proxy._types import LiteLLM_MCPServerTable, MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPOAuthMetadata, MCPServer
from mcp import ReadResourceResult, Resource
from mcp.types import GetPromptResult, Prompt, ResourceTemplate, TextResourceContents


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

        manager.add_update_server(stdio_server)

        # Verify server was added
        assert "stdio-server-1" in manager.registry
        added_server = manager.registry["stdio-server-1"]

        assert added_server.server_id == "stdio-server-1"
        assert added_server.name == "test_stdio_server"
        assert added_server.transport == MCPTransport.stdio
        assert added_server.command == "python"
        assert added_server.args == ["-m", "server"]
        assert added_server.env == {"DEBUG": "1", "TEST": "1"}

    def test_create_mcp_client_stdio(self):
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

        client = manager._create_mcp_client(stdio_server)

        assert client.transport_type == MCPTransport.stdio
        assert client.stdio_config is not None
        assert client.stdio_config["command"] == "node"
        assert client.stdio_config["args"] == ["server.js"]
        assert client.stdio_config["env"] == {"NODE_ENV": "test"}

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
        manager.get_mcp_server_by_id = MagicMock(
            side_effect=lambda x: server1 if x == "github" else server2
        )

        # Mock _get_tools_from_server to return different results
        async def mock_get_tools_from_server(
            server, mcp_auth_header=None, mcp_protocol_version=None
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

        result = await manager.list_tools(
            mcp_server_auth_headers=mcp_server_auth_headers
        )

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
            server, mcp_auth_header=None, mcp_protocol_version=None
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
            server, mcp_auth_header=None, mcp_protocol_version=None
        ):
            assert (
                mcp_auth_header == "server-specific-token"
            )  # Should use server-specific header
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

        with patch.object(manager, "_create_mcp_client", return_value=mock_client):
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

        with patch.object(manager, "_create_mcp_client", return_value=mock_client):
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

        with patch.object(manager, "_create_mcp_client", return_value=mock_client) as mock_create_client, patch.object(
            manager,
            "_create_prefixed_resources",
            return_value=prefixed_resources,
        ) as mock_prefix:
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

        with patch.object(manager, "_create_mcp_client", return_value=mock_client) as mock_create_client, patch.object(
            manager,
            "_create_prefixed_resource_templates",
            return_value=prefixed_templates,
        ) as mock_prefix:
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

        with patch.object(manager, "_create_mcp_client", return_value=mock_client) as mock_create_client:
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
                "https://protected.example.com/.well-known/oauth"
            )

        assert servers == [
            "https://auth1.example.com",
            "https://auth2.example.com",
        ]
        assert scopes == ["read", "write"]

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
            raise httpx.HTTPStatusError(
                "unauthorized", request=request, response=response_obj
            )

        response_obj.raise_for_status = MagicMock(side_effect=raise_http_error)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=response_obj)

        mock_metadata = MCPOAuthMetadata(
            scopes=None,
            authorization_url="https://example.com/auth",
            token_url="https://example.com/token",
            registration_url=None,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.get_async_httpx_client",
            return_value=mock_client,
        ), patch.object(
            manager,
            "_fetch_oauth_metadata_from_resource",
            AsyncMock(return_value=([], None)),
        ), patch.object(
            manager,
            "_attempt_well_known_discovery",
            AsyncMock(return_value=([], None)),
        ), patch.object(
            manager,
            "_fetch_authorization_server_metadata",
            AsyncMock(return_value=mock_metadata),
        ) as mock_fetch_auth:
            result = await manager._descovery_metadata(server_url)

        mock_fetch_auth.assert_awaited_once_with(["https://example.com"])
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

        async def fake_discovery(server_url: str):
            assert server_url == "https://example.com/mcp"
            return discovered_metadata

        manager._descovery_metadata = fake_discovery  # type: ignore[attr-defined]

        config = {
            "example": {
                "url": "https://example.com/mcp",
                "transport": MCPTransport.http,
                "auth_type": MCPAuth.oauth2,
                "scopes": ["config"],
                "authorization_url": "https://config.example.com/auth",
            }
        }

        await manager.load_servers_from_config(config)

        server = next(iter(manager.config_mcp_servers.values()))
        assert server.scopes == ["config"]  # config overrides discovery
        assert server.authorization_url == "https://config.example.com/auth"
        assert server.token_url == "https://discovered.example.com/token"
        assert (
            server.registration_url == "https://discovered.example.com/register"
        )

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
            server, mcp_auth_header=None, mcp_protocol_version=None
        ):
            assert (
                mcp_auth_header == "server-specific-token"
            )  # Should use server-specific header via server_name
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
        server = MagicMock()
        server.server_id = "test-server"
        server.name = "test-server"

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock successful _get_tools_from_server
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            tool1 = MagicMock()
            tool1.name = "tool1"
            tool2 = MagicMock()
            tool2.name = "tool2"
            return [tool1, tool2]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Perform health check
        result = await manager.health_check_server("test-server")

        # Verify results
        assert result["server_id"] == "test-server"
        assert result["status"] == "healthy"
        assert result["tools_count"] == 2
        assert result["error"] is None
        assert "last_health_check" in result
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0  # Allow 0 for very fast mocks

    @pytest.mark.asyncio
    async def test_health_check_server_unhealthy(self):
        """Test health check for an unhealthy server"""
        manager = MCPServerManager()

        # Mock server
        server = MagicMock()
        server.server_id = "test-server"
        server.name = "test-server"

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock failed _get_tools_from_server
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            raise Exception("Connection timeout")

        manager._get_tools_from_server = mock_get_tools_from_server

        # Perform health check
        result = await manager.health_check_server("test-server")

        # Verify results
        assert result["server_id"] == "test-server"
        assert result["status"] == "unhealthy"
        assert result["error"] == "Connection timeout"
        assert "last_health_check" in result
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0  # Allow 0 for very fast mocks

    @pytest.mark.asyncio
    async def test_health_check_server_not_found(self):
        """Test health check for a server that doesn't exist"""
        manager = MCPServerManager()

        # Mock server not found
        manager.get_mcp_server_by_id = MagicMock(return_value=None)

        # Perform health check
        result = await manager.health_check_server("non-existent-server")

        # Verify results
        assert result["server_id"] == "non-existent-server"
        assert result["status"] == "unknown"
        assert result["error"] == "Server not found"
        assert result["response_time_ms"] is None
        assert "last_health_check" in result

    @pytest.mark.asyncio
    async def test_health_check_all_servers(self):
        """Test health check for all servers"""
        manager = MCPServerManager()

        # Mock servers
        server1 = MagicMock()
        server1.server_id = "server1"
        server1.name = "server1"

        server2 = MagicMock()
        server2.server_id = "server2"
        server2.name = "server2"

        # Mock registry
        manager.registry = {"server1": server1, "server2": server2}

        # Mock get_mcp_server_by_id
        def mock_get_server_by_id(server_id):
            if server_id == "server1":
                return server1
            elif server_id == "server2":
                return server2
            return None

        manager.get_mcp_server_by_id = mock_get_server_by_id

        # Mock _get_tools_from_server with different results
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            if server.server_id == "server1":
                tool = MagicMock()
                tool.name = "tool1"
                return [tool]
            elif server.server_id == "server2":
                raise Exception("Connection failed")
            return []

        manager._get_tools_from_server = mock_get_tools_from_server

        # Perform health check for all servers
        result = await manager.health_check_all_servers()

        # Verify results
        assert len(result) == 2
        assert "server1" in result
        assert "server2" in result

        # Check server1 (healthy)
        assert result["server1"]["status"] == "healthy"
        assert result["server1"]["tools_count"] == 1
        assert result["server1"]["error"] is None

        # Check server2 (unhealthy)
        assert result["server2"]["status"] == "unhealthy"
        assert result["server2"]["error"] == "Connection failed"

    @pytest.mark.asyncio
    async def test_health_check_server_with_auth_header(self):
        """Test health check with authentication header"""
        manager = MCPServerManager()

        # Mock server
        server = MagicMock()
        server.server_id = "test-server"
        server.name = "test-server"

        manager.get_mcp_server_by_id = MagicMock(return_value=server)

        # Mock _get_tools_from_server to verify auth header is passed
        async def mock_get_tools_from_server(server, mcp_auth_header=None):
            assert mcp_auth_header == "test-token"
            tool = MagicMock()
            tool.name = "tool1"
            return [tool]

        manager._get_tools_from_server = mock_get_tools_from_server

        # Perform health check with auth header
        result = await manager.health_check_server("test-server", "test-token")

        # Verify results
        assert result["server_id"] == "test-server"
        assert result["status"] == "healthy"
        assert result["tools_count"] == 1

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
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        assert (
            "Tool blocked_tool is not allowed for server test-server"
            in exc_info.value.detail["error"]
        )
        assert (
            "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]
        )

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
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        assert (
            "Tool banned_tool is not allowed for server test-server"
            in exc_info.value.detail["error"]
        )
        assert (
            "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]
        )

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
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        assert (
            "Tool tool3 is not allowed for server test-server"
            in exc_info.value.detail["error"]
        )

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
        manager._create_mcp_client = MagicMock(return_value=object())

        # Tools returned upstream (unprefixed from provider)
        upstream_tool = MagicMock()
        upstream_tool.name = "send_email"
        upstream_tool.description = "Send an email"
        upstream_tool.inputSchema = {}

        manager._fetch_tools_with_timeout = AsyncMock(return_value=[upstream_tool])

        # Case 1: add_prefix=True (default for multi-server) -> expect prefixed
        tools_prefixed = await manager._get_tools_from_server(server, add_prefix=True)
        assert len(tools_prefixed) == 1
        assert tools_prefixed[0].name == "zapier-send_email"

        # Case 2: add_prefix=False (single-server) -> expect unprefixed
        tools_unprefixed = await manager._get_tools_from_server(
            server, add_prefix=False
        )
        assert len(tools_unprefixed) == 1
        assert tools_unprefixed[0].name == "send_email"

    def test_create_prefixed_tools_updates_mapping_for_both_forms(self):
        """_create_prefixed_tools should populate mapping for prefixed and original names even when not adding prefix in output."""
        manager = MCPServerManager()

        server = MCPServer(
            server_id="jira",
            name="jira",
            transport=MCPTransport.http,
        )

        # Input tools as would come from upstream
        t1 = MagicMock()
        t1.name = "create_issue"
        t1.description = ""
        t1.inputSchema = {}
        t2 = MagicMock()
        t2.name = "close_issue"
        t2.description = ""
        t2.inputSchema = {}

        # Do not add prefix in returned objects
        out_tools = manager._create_prefixed_tools([t1, t2], server, add_prefix=False)

        # Returned names should be unprefixed
        names = sorted([t.name for t in out_tools])
        assert names == ["close_issue", "create_issue"]

        # Mapping should include both original and prefixed names -> resolves calls either way
        assert manager.tool_name_to_mcp_server_name_mapping["create_issue"] == "jira"
        assert (
            manager.tool_name_to_mcp_server_name_mapping["jira-create_issue"] == "jira"
        )
        assert manager.tool_name_to_mcp_server_name_mapping["close_issue"] == "jira"
        assert (
            manager.tool_name_to_mcp_server_name_mapping["jira-close_issue"] == "jira"
        )

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
        base_tool = MagicMock()
        base_tool.name = "create_zap"
        base_tool.description = ""
        base_tool.inputSchema = {}
        _ = manager._create_prefixed_tools([base_tool], server, add_prefix=False)

        # Unprefixed resolution
        resolved_server_unpref = manager._get_mcp_server_from_tool_name("create_zap")
        print(resolved_server_unpref)
        assert resolved_server_unpref is not None
        assert resolved_server_unpref.server_id == server.server_id

        # Prefixed resolution
        resolved_server_pref = manager._get_mcp_server_from_tool_name(
            "zapier-create_zap"
        )
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
            filtered_response = await _get_tools_for_single_server(
                server, server_auth_header=None
            )

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
            all_tools_response = await _get_tools_for_single_server(
                server, server_auth_header=None
            )

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
            all_tools_response = await _get_tools_for_single_server(
                server, server_auth_header=None
            )

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
        manager.add_update_server(server)
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
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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
        assert (
            "Tool deletepet is not allowed for server my_api_mcp"
            in exc_info.value.detail["error"]
        )
        assert (
            "Contact proxy admin to allow this tool" in exc_info.value.detail["error"]
        )

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

        async def mock_call_tool(params):
            # Return a mock CallToolResult
            result = MagicMock(spec=CallToolResult)
            result.content = [{"type": "text", "text": "Tool executed successfully"}]
            result.isError = False
            return result

        mock_client.call_tool.side_effect = mock_call_tool

        # Mock _create_mcp_client to return our mock client
        manager._create_mcp_client = MagicMock(return_value=mock_client)

        # Mock user auth with no restrictions
        user_api_key_auth = MagicMock()
        user_api_key_auth.object_permission = None
        user_api_key_auth.object_permission_id = None

        # Mock proxy logging
        proxy_logging_obj = MagicMock()
        proxy_logging_obj._create_mcp_request_object_from_kwargs = MagicMock(
            return_value={}
        )
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


if __name__ == "__main__":
    pytest.main([__file__])
