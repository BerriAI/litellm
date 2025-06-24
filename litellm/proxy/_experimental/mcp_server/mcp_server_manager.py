"""
MCP Client Manager

This class is responsible for managing MCP clients with support for both SSE and HTTP streamable transports.

This is a Proxy
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, cast

from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
    UserAPIKeyAuthMCP,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPSpecVersion,
    MCPSpecVersionType,
    MCPTransport,
    MCPTransportType,
    UserAPIKeyAuth,
)
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPServer


class MCPServerManager:
    def __init__(self):
        self.registry: Dict[str, MCPServer] = {}
        self.config_mcp_servers: Dict[str, MCPServer] = {}
        """
        eg.
        [
            "server-1": {
                "name": "zapier_mcp_server",
                "url": "https://actions.zapier.com/mcp/sk-ak-2ew3bofIeQIkNoeKIdXrF1Hhhp/sse"
                "transport": "sse",
                "auth_type": "api_key",
                "spec_version": "2025-03-26"
            },
            "uuid-2": {
                "name": "google_drive_mcp_server",
                "url": "https://actions.zapier.com/mcp/sk-ak-2ew3bofIeQIkNoeKIdXrF1Hhhp/sse"
            }
        ]
        """

        self.tool_name_to_mcp_server_name_mapping: Dict[str, str] = {}
        """
        {
            "gmail_send_email": "zapier_mcp_server",
        }
        """

    def get_registry(self) -> Dict[str, MCPServer]:
        """
        Get the registered MCP Servers from the registry and union with the config MCP Servers
        """
        return self.config_mcp_servers | self.registry

    def load_servers_from_config(self, mcp_servers_config: Dict[str, Any]):
        """
        Load the MCP Servers from the config
        """
        verbose_logger.debug("Loading MCP Servers from config-----")
        for server_name, server_config in mcp_servers_config.items():
            _mcp_info: dict = server_config.get("mcp_info", None) or {}
            mcp_info = MCPInfo(**_mcp_info)
            mcp_info["server_name"] = server_name
            mcp_info["description"] = server_config.get("description", None)

            # Generate stable server ID based on parameters
            server_id = self._generate_stable_server_id(
                server_name=server_name,
                url=server_config["url"],
                transport=server_config.get("transport", MCPTransport.http),
                spec_version=server_config.get("spec_version", MCPSpecVersion.mar_2025),
                auth_type=server_config.get("auth_type", None),
            )

            new_server = MCPServer(
                server_id=server_id,
                name=server_name,
                url=server_config["url"],
                # TODO: utility fn the default values
                transport=server_config.get("transport", MCPTransport.http),
                spec_version=server_config.get("spec_version", MCPSpecVersion.mar_2025),
                auth_type=server_config.get("auth_type", None),
                mcp_info=mcp_info,
            )
            self.config_mcp_servers[server_id] = new_server
        verbose_logger.debug(
            f"Loaded MCP Servers: {json.dumps(self.config_mcp_servers, indent=4, default=str)}"
        )

        self.initialize_tool_name_to_mcp_server_name_mapping()

    def remove_server(self, mcp_server: LiteLLM_MCPServerTable):
        """
        Remove a server from the registry
        """
        if mcp_server.alias in self.get_registry():
            del self.registry[mcp_server.alias]
            verbose_logger.debug(f"Removed MCP Server: {mcp_server.alias}")
        elif mcp_server.server_id in self.get_registry():
            del self.registry[mcp_server.server_id]
            verbose_logger.debug(f"Removed MCP Server: {mcp_server.server_id}")
        else:
            verbose_logger.warning(
                f"Server ID {mcp_server.server_id} not found in registry"
            )

    def add_update_server(self, mcp_server: LiteLLM_MCPServerTable):
        if mcp_server.server_id not in self.get_registry():
            new_server = MCPServer(
                server_id=mcp_server.server_id,
                name=mcp_server.alias or mcp_server.server_id,
                url=mcp_server.url,
                transport=cast(MCPTransportType, mcp_server.transport),
                spec_version=cast(MCPSpecVersionType, mcp_server.spec_version),
                auth_type=cast(MCPAuthType, mcp_server.auth_type),
                mcp_info=MCPInfo(
                    server_name=mcp_server.alias or mcp_server.server_id,
                    description=mcp_server.description,
                ),
            )
            self.registry[mcp_server.server_id] = new_server
            verbose_logger.debug(
                f"Added MCP Server: {mcp_server.alias or mcp_server.server_id}"
            )

    async def get_allowed_mcp_servers(
        self, user_api_key_auth: Optional[UserAPIKeyAuth] = None
    ) -> List[str]:
        """
        Get the allowed MCP Servers for the user
        """
        allowed_mcp_servers = await UserAPIKeyAuthMCP.get_allowed_mcp_servers(
            user_api_key_auth
        )
        verbose_logger.debug(
            f"Allowed MCP Servers for user api key auth: {allowed_mcp_servers}"
        )
        if len(allowed_mcp_servers) > 0:
            return allowed_mcp_servers
        else:
            verbose_logger.debug(
                "No allowed MCP Servers found for user api key auth, returning default registry servers"
            )
            return list(self.get_registry().keys())

    async def list_tools(
        self, 
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
    ) -> List[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Returns:
            List[MCPTool]: Combined list of tools from all servers
        """
        allowed_mcp_servers = await self.get_allowed_mcp_servers(user_api_key_auth)

        list_tools_result: List[MCPTool] = []
        verbose_logger.debug("SERVER MANAGER LISTING TOOLS")

        for server_id in allowed_mcp_servers:
            server = self.get_mcp_server_by_id(server_id)
            if server is None:
                verbose_logger.warning(f"MCP Server {server_id} not found")
                continue
            try:
                tools = await self._get_tools_from_server(
                    server=server,
                    mcp_auth_header=mcp_auth_header,
                )
                list_tools_result.extend(tools)
            except Exception as e:
                verbose_logger.exception(
                    f"Error listing tools from server {server.name}: {str(e)}"
                )

        return list_tools_result

    #########################################################
    # Methods that call the upstream MCP servers
    #########################################################
    def _create_mcp_client(self, server: MCPServer, mcp_auth_header: Optional[str] = None) -> MCPClient:
        """
        Create an MCPClient instance for the given server.

        Args:
            server (MCPServer): The server configuration
            mcp_auth_header: MCP auth header to be passed to the MCP server. This is optional and will be used if provided.

        Returns:
            MCPClient: Configured MCP client instance
        """
        transport = server.transport or MCPTransport.sse
        return MCPClient(
            server_url=server.url,
            transport_type=transport,
            auth_type=server.auth_type,
            auth_value=mcp_auth_header or server.authentication_token,
            timeout=60.0,
        )

    async def _get_tools_from_server(self, server: MCPServer, mcp_auth_header: Optional[str] = None) -> List[MCPTool]:
        """
        Helper method to get tools from a single MCP server.

        Args:
            server (MCPServer): The server to query tools from

        Returns:
            List[MCPTool]: List of tools available on the server
        """
        verbose_logger.debug(f"Connecting to url: {server.url}")
        verbose_logger.info("_get_tools_from_server...")

        client = self._create_mcp_client(
            server=server,
            mcp_auth_header=mcp_auth_header,
        )
        async with client:
            tools = await client.list_tools()
            verbose_logger.debug(f"Tools from {server.name}: {tools}")

            # Update tool to server mapping
            for tool in tools:
                self.tool_name_to_mcp_server_name_mapping[tool.name] = server.name

            return tools
    
    async def call_tool(
        self, 
        name: str, 
        arguments: Dict[str, Any],
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
        mcp_auth_header: Optional[str] = None,
    ) -> CallToolResult:
        """
        Call a tool with the given name and arguments
        """
        mcp_server = self._get_mcp_server_from_tool_name(name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")

        client = self._create_mcp_client(
            server=mcp_server,
            mcp_auth_header=mcp_auth_header,
        )
        async with client:
            call_tool_params = MCPCallToolRequestParams(
                name=name,
                arguments=arguments,
            )
            return await client.call_tool(call_tool_params)
    
    #########################################################
    # End of Methods that call the upstream MCP servers
    #########################################################


    def initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        On startup, initialize the tool name to MCP server name mapping
        """
        try:
            if asyncio.get_running_loop():
                asyncio.create_task(
                    self._initialize_tool_name_to_mcp_server_name_mapping()
                )
        except RuntimeError as e:  # no running event loop
            verbose_logger.exception(
                f"No running event loop - skipping tool name to MCP server name mapping initialization: {str(e)}"
            )

    async def _initialize_tool_name_to_mcp_server_name_mapping(self):
        """
        Call list_tools for each server and update the tool name to MCP server name mapping
        """
        for server in self.get_registry().values():
            tools = await self._get_tools_from_server(server)
            for tool in tools:
                self.tool_name_to_mcp_server_name_mapping[tool.name] = server.name

    def _get_mcp_server_from_tool_name(self, tool_name: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the tool name
        """
        if tool_name in self.tool_name_to_mcp_server_name_mapping:
            for server in self.get_registry().values():
                if server.name == self.tool_name_to_mcp_server_name_mapping[tool_name]:
                    return server
        return None

    async def _add_mcp_servers_from_db_to_in_memory_registry(self):
        from litellm.proxy._experimental.mcp_server.db import get_all_mcp_servers
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        # perform authz check to filter the mcp servers user has access to
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )
        db_mcp_servers = await get_all_mcp_servers(prisma_client)
        # ensure the global_mcp_server_manager is up to date with the db
        for server in db_mcp_servers:
            self.add_update_server(server)

    def get_mcp_server_by_id(self, server_id: str) -> Optional[MCPServer]:
        """
        Get the MCP Server from the server id
        """
        for server in self.get_registry().values():
            if server.server_id == server_id:
                return server
        return None

    def _generate_stable_server_id(
        self,
        server_name: str,
        url: str,
        transport: str,
        spec_version: str,
        auth_type: Optional[str] = None,
    ) -> str:
        """
        Generate a stable server ID based on server parameters using a hash function.

        This is critical to ensure the server_id is stable across server restarts.
        Some users store MCPs on the config.yaml and permission management is based on server_ids.

        Eg a key might have mcp_servers = ["1234"], if the server_id changes across restarts, the key will no longer have access to the MCP.

        Args:
            server_name: Name of the server
            url: Server URL
            transport: Transport type (sse, http, etc.)
            spec_version: MCP spec version
            auth_type: Authentication type (optional)

        Returns:
            A deterministic server ID string
        """
        # Create a string from all the identifying parameters
        params_string = (
            f"{server_name}|{url}|{transport}|{spec_version}|{auth_type or ''}"
        )

        # Generate SHA-256 hash
        hash_object = hashlib.sha256(params_string.encode("utf-8"))
        hash_hex = hash_object.hexdigest()

        # Take first 32 characters and format as UUID-like string
        return hash_hex[:32]


global_mcp_server_manager: MCPServerManager = MCPServerManager()
