"""
MCP Client Manager

This class is responsible for managing MCP SSE clients.

This is a Proxy 
"""

from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_logger


class MCPSSEServer(BaseModel):
    name: str
    url: str
    client_session: Optional[ClientSession] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MCPServerManager:
    def __init__(self, mcp_servers: List[MCPSSEServer]):
        self.mcp_servers: List[MCPSSEServer] = mcp_servers
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

    async def list_tools(self):
        """
        List all tools available in all the MCP Servers
        """
        for server in self.mcp_servers:
            async with sse_client(url=server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    server.client_session = session
                    await server.client_session.initialize()
                    list_tools_result = await server.client_session.list_tools()
                    verbose_logger.debug(
                        f"Tools from {server.name}: {list_tools_result}"
                    )
                    for tool in list_tools_result.tools:
                        self.tool_name_to_mcp_server_name_mapping[tool.name] = (
                            server.name
                        )

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """
        Call a tool with the given name and arguments
        """
        mcp_server = self._get_mcp_server_from_tool_name(name)
        if mcp_server is None:
            raise ValueError(f"Tool {name} not found")
        async with sse_client(url=mcp_server.url) as (read, write):
            async with ClientSession(read, write) as session:
                mcp_server.client_session = session
                await mcp_server.client_session.initialize()
                return await mcp_server.client_session.call_tool(name, arguments)

    def _get_mcp_server_from_tool_name(self, tool_name: str) -> Optional[MCPSSEServer]:
        """
        Get the MCP Server from the tool name
        """
        if tool_name in self.tool_name_to_mcp_server_name_mapping:
            for server in self.mcp_servers:
                if server.name == self.tool_name_to_mcp_server_name_mapping[tool_name]:
                    return server
        return None
