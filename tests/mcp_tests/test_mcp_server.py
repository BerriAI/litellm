# Create server parameters for stdio connection
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.mcp_client_manager import (
    MCPServerManager,
    MCPSSEServer,
)


MCP_SERVERS = [
    MCPSSEServer(name="zapier_mcp_server", url=os.environ.get("ZAPIER_MCP_SERVER_URL")),
]

mcp_server_manager = MCPServerManager(mcp_servers=MCP_SERVERS)


@pytest.mark.asyncio
async def test_mcp_server_manager():
    tools = await mcp_server_manager.list_tools()
    print("TOOLS FROM MCP SERVER MANAGER== ", tools)

    result = await mcp_server_manager.call_tool(
        name="gmail_send_email", arguments={"body": "Test"}
    )
    print("RESULT FROM CALLING TOOL FROM MCP SERVER MANAGER== ", result)


"""
TODO test with multiple MCP servers and calling a specific 

"""
