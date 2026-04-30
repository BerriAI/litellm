"""
Shared ContextVars for the MCP server layer.

Lives in its own module to avoid circular imports between
mcp_server_manager.py and server.py.
"""

from contextvars import ContextVar
from typing import Optional

# Set server-side in proxy_server.py route handlers when a request arrives via
# /toolset/{name}/mcp or the toolset fallback in dynamic_mcp_route.
# Never populated from client-supplied headers.
_mcp_active_toolset_id: ContextVar[Optional[str]] = ContextVar(
    "_mcp_active_toolset_id", default=None
)

# Per-request merged InitializeResult.instructions; set in MCP HTTP/SSE handlers.
_mcp_gateway_initialize_instructions: ContextVar[Optional[str]] = ContextVar(
    "_mcp_gateway_initialize_instructions", default=None
)
