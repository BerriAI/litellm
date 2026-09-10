import contextlib
import json

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Mount

server = Server("Dotted argument echo")
properties = {
    "filter.category": {"type": "string"},
    "query": {"type": "string"},
    "options": {"type": "object"},
}

@server.list_tools()
async def list_tools():
    return [Tool(name=name, description="Echo arguments exactly as received", inputSchema={
        "type": "object", "properties": properties, "required": required,
    }) for name, required in (("echo_required", ["filter.category", "query"]), ("echo_optional", ["query"]))]

@server.call_tool()
async def call_tool(name, arguments):
    print(json.dumps({"tool": name, "arguments": arguments}), flush=True)
    return [TextContent(type="text", text=json.dumps(arguments))]

manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True,
    security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False))

async def handle(scope, receive, send):
    await manager.handle_request(scope, receive, send)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with manager.run():
        yield

app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)
uvicorn.run(app, host="0.0.0.0", port=8080)
