import asyncio
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.mcp import MCPAuth
from litellm.proxy._types import MCPTransport

async def main():
    manager = MCPServerManager()
    server = MCPServer(server_id="lifecycle-live", name="temporary", url="https://mcp.figma.com/mcp", transport=MCPTransport.http, auth_type=MCPAuth.true_passthrough)
    manager._set_oauth_discovery_deferred(server.server_id, True)
    resolved = await manager.ensure_oauth_metadata_discovered(server)
    print(json.dumps({"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "authorization_url": resolved.authorization_url, "retained_immediately": manager._oauth_discovery_slot(server.server_id) is not None}), flush=True)
    await asyncio.sleep(301)
    print(json.dumps({"elapsed_seconds": 301, "retained_after_session_ttl": manager._oauth_discovery_slot(server.server_id) is not None}), flush=True)

asyncio.run(main())
