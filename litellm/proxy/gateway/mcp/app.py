"""Composition root for MCP Gateway v2.

``build_gateway(deps)`` is the single place the whole object graph is assembled
(Seemann's composition root). Everything else receives its dependencies by
constructor injection — no module-level singletons, no service locator.

S0 wires the chassis only: an SDK low-level Server with skeleton handlers
(empty tool list, typed not-implemented call), mounted over the streamable-HTTP
session manager. Each later section fills in a real handler and registers it
here exactly once.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from litellm.proxy.gateway.mcp.foundation import GatewayDeps, GatewayError, reason
from litellm.proxy.gateway.mcp.foundation.types import CallToolResult, TextContent, Tool

_SERVER_NAME = "litellm-mcp-gateway"
_SERVER_VERSION = "2.0.0"
_INSTRUCTIONS = "LiteLLM MCP gateway: aggregated, namespaced upstream tools."


def _not_implemented_result(name: str) -> CallToolResult:
    err = GatewayError(not_implemented=f"tools/call not wired in S0: {name!r}")
    return CallToolResult(
        content=[TextContent(type="text", text=reason(err))],
        isError=True,
    )


def build_server(deps: GatewayDeps) -> Server[object, object]:
    server: Server[object, object] = Server(
        name=_SERVER_NAME,
        version=_SERVER_VERSION,
        instructions=_INSTRUCTIONS,
    )

    async def list_tools() -> list[Tool]:
        return []

    async def call_tool(name: str, arguments: dict[str, object]) -> CallToolResult:
        return _not_implemented_result(name)

    _ = server.list_tools()(list_tools)
    _ = server.call_tool()(call_tool)
    return server


def build_gateway(deps: GatewayDeps) -> Starlette:
    """Assemble and return the gateway ASGI app from injected dependencies.

    The ONLY construction site. Two calls yield independent apps; there is no
    shared module-level state.
    """
    server = build_server(deps)
    manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        async with manager.run():
            yield

    return Starlette(
        lifespan=lifespan,
        routes=[Mount("/mcp", app=manager.handle_request)],
    )
