"""BYOK upstream MCP server for the rig: answers whoami with a fingerprint of the x-api-key it received.

usage: python upstream.py <port>
"""
import asyncio
import hashlib
import sys

from mcp.server.mcpserver import Context, MCPServer

server = MCPServer(name="byok-echo")


@server.tool()
def whoami(ctx: Context) -> str:
    """Fingerprint of the API key this request carried."""
    headers = dict(ctx.headers or {})
    key = next((v for k, v in headers.items() if k.lower() == "x-api-key"), "")
    return "fp=" + (hashlib.sha256(key.encode()).hexdigest()[:12] if key else "none")


asyncio.run(
    server.run_streamable_http_async(host="127.0.0.1", port=int(sys.argv[1]), json_response=True, stateless_http=True)
)
