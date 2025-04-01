"""
MCP Client Manager

This class is responsible for managing MCP SSE clients.

This is a Proxy
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.types.mcp_server.mcp_server_manager import MCPInfo, MCPSSEServer


class MCPServerManager:
    def __init__(self):
        self.mcp_servers: List[MCPSSEServer] = []
        """
        eg.
        [
            {
                "name": "zapier_mcp_server",
                "url": "https://actions.zapier.com/mcp/sk-ak-2ew3bofIeQIkNoeKIdXrF1Hhhp/sse"
            },
            {
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

    def load_servers_from_config(self, mcp_servers_config: Dict[str, Any]):
        """
        Load the MCP Servers from the config
        """
        for server_name, server_config in mcp_servers_config.items():
            _mcp_info: dict = server_config.get("mcp_info", None) or {}
            mcp_info = MCPInfo(**_mcp_info)
            mcp_info["server_name"] = server_name
            self.mcp_servers.append(
                MCPSSEServer(
                    name=server_name,
                    url=server_config["url"],
                    mcp_info=mcp_info,
                )
            )
        verbose_logger.debug(
            f"Loaded MCP Servers: {json.dumps(self.mcp_servers, indent=4, default=str)}"
        )

        self.initialize_tool_name_to_mcp_server_name_mapping()

    async def list_tools(self) -> List[MCPTool]:
        """
        List all tools available across all MCP Servers.

        Returns:
            List[MCPTool]: Combined list of tools from all servers
        """
        list_tools_result: List[MCPTool] = []
        verbose_logger.debug("SSE SERVER MANAGER LISTING TOOLS")

        for server in self.mcp_servers:
            tools = await self._get_tools_from_server(server)
            list_tools_result.extend(tools)

        return list_tools_result

    async def _get_tools_from_server(self, server: MCPSSEServer) -> List[MCPTool]:
        """
        Helper method to get tools from a single MCP server.

        Args:
            server (MCPSSEServer): The server to query tools from

        Returns:
            List[MCPTool]: List of tools available on the server
        """
        verbose_logger.debug(f"Connecting to url: {server.url}")

        async with sse_client(url=server.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                verbose_logger.debug(f"Tools from {server.name}: {tools_result}")

                # Update tool to server mapping
                for tool in tools_result.tools:
                    self.tool_name_to_mcp_server_name_mapping[tool.name] = server.name

                return tools_result.tools

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
        for server in self.mcp_servers:
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
        async with sse_client(url=mcp_server.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    def _get_mcp_server_from_tool_name(self, tool_name: str) -> Optional[MCPSSEServer]:
        """
        Get the MCP Server from the tool name
        """
        if tool_name in self.tool_name_to_mcp_server_name_mapping:
            for server in self.mcp_servers:
                if server.name == self.tool_name_to_mcp_server_name_mapping[tool_name]:
                    return server
        return None


global_mcp_server_manager: MCPServerManager = MCPServerManager()
