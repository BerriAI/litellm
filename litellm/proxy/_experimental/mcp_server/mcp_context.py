"""
Shared ContextVars for the MCP server layer.

Lives in its own module to avoid circular imports between
mcp_server_manager.py and server.py.
"""

from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple

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

# Per-request memoization of _get_allowed_mcp_servers. Set to a fresh dict at
# the entry of an MCP HTTP request; downstream callers (probe, list_tools,
# call_tool, etc.) reuse the resolved server list instead of re-running the
# full key/team/end_user/agent/org permission chain.
_mcp_request_allowed_servers_cache: ContextVar[
    Optional[Dict[Tuple[Optional[Tuple[str, ...]], Optional[str]], Any]]
] = ContextVar("_mcp_request_allowed_servers_cache", default=None)
