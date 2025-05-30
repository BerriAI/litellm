"""
MCP Client Manager

This class is responsible for managing MCP SSE clients.

This is a Proxy
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, cast

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPAuthType,
    MCPSpecVersion,
    MCPSpecVersionType,
    MCPTransport,
    MCPTransportType,
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
            new_server = MCPServer(
                server_id=str(uuid.uuid4()),
                name=server_name,
                url=server_config["url"],
                # TODO: utility fn the default values
                transport=server_config.get("transport", MCPTransport.sse),
                spec_version=server_config.get("spec_version", MCPSpecVersion.mar_2025),
                auth_type=server_config.get("auth_type", None),
                mcp_info=mcp_info,
            )
            server_id = str(uuid.uuid4())
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

    async def list_tools(self) -> List[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Returns:
            List[MCPTool]: Combined list of tools from all servers
        """
        list_tools_result: List[MCPTool] = []
        verbose_logger.debug("SERVER MANAGER LISTING TOOLS")

        for _, server in self.get_registry().items():
            try:
                tools = await self._get_tools_from_server(server)
                list_tools_result.extend(tools)
            except Exception as e:
                verbose_logger.exception(
                    f"Error listing tools from server {server.name}: {str(e)}"
                )

        return list_tools_result

    async def _get_tools_from_server(self, server: MCPServer) -> List[MCPTool]:
        """
        Helper method to get tools from a single MCP server.

        Args:
            server (MCPServer): The server to query tools from

        Returns:
            List[MCPTool]: List of tools available on the server
        """
        verbose_logger.debug(f"Connecting to url: {server.url}")

        verbose_logger.info("_get_tools_from_server...")
        # send transport to connect to the server
        if server.transport is None or server.transport == MCPTransport.sse:
            async with sse_client(url=server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools_result = await session.list_tools()
                    verbose_logger.debug(f"Tools from {server.name}: {tools_result}")

                    # Update tool to server mapping
                    for tool in tools_result.tools:
                        self.tool_name_to_mcp_server_name_mapping[
                            tool.name
                        ] = server.name

                    return tools_result.tools
        elif server.transport == MCPTransport.http:
            # TODO: implement http transport
            return []
        else:
            # TODO: throw error on transport found or skip
            return []

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

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """
        Call a tool with the given name and arguments
        """
        mcp_server = self._get_mcp_server_from_tool_name(name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")
        elif mcp_server.transport is None or mcp_server.transport == MCPTransport.sse:
            async with sse_client(url=mcp_server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(name, arguments)
        elif mcp_server.transport == MCPTransport.http:
            # TODO: implement http transport
            raise NotImplementedError("HTTP transport is not implemented yet")
        else:
            return CallToolResult(content=[], isError=True)

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


global_mcp_server_manager: MCPServerManager = MCPServerManager()
