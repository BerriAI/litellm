# Create server parameters for stdio connection
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    MCPServer,
)


mcp_server_manager = MCPServerManager()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Local only test")
async def test_mcp_server_manager():
    mcp_server_manager.load_servers_from_config(
        {
            "zapier_mcp_server": {
                "url": os.environ.get("ZAPIER_MCP_SERVER_URL"),
            }
        }
    )
    tools = await mcp_server_manager.list_tools()
    print("TOOLS FROM MCP SERVER MANAGER== ", tools)

    result = await mcp_server_manager.call_tool(
        name="gmail_send_email", arguments={"body": "Test"}
    )
    print("RESULT FROM CALLING TOOL FROM MCP SERVER MANAGER== ", result)
