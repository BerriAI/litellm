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

# Set when an MCP request arrives via a path-scoped URL like /mcp/{server_name}
# (or /{server_name}/mcp).  In that case the client has already disambiguated the
# server by URL, so tool names from list_tools / list_prompts / list_resources /
# list_resource_templates SHOULD NOT be prefixed with the server name.  When the
# request comes in on the aggregated /mcp endpoint (no server in the path) this
# flag stays False and the aggregated list keeps its server-name prefixes so
# tool names remain globally unique.
# This is intentionally transport-agnostic: stdio, SSE and HTTP-streamable servers
# all honour the same flag, so behaviour stays consistent across transports.
_mcp_request_scoped_to_single_server: ContextVar[bool] = ContextVar(
    "_mcp_request_scoped_to_single_server", default=False
)
