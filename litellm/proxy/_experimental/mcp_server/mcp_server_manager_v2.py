"""v2-owned MCP egress manager (the cutover seam).

``MCPServerManagerV2`` subclasses v1's ``MCPServerManager`` and, in later steps, overrides the
per-server egress methods (``_get_tools_from_server``, ``call_tool``, the prompt/resource ops) to
route through the v2 ``UpstreamConnection`` + ``resolve()`` instead of ``_create_mcp_client``.
Registry, RBAC, cross-server aggregation, namespacing, and static-header resolution are inherited
from v1 unchanged. It is the egress manager, constructed at the composition root (see
``mcp_server_manager._make_global_mcp_server_manager``); there is no opt-in flag (v2 is the egress
implementation). v1's per-server methods remain reachable via ``super()`` for modes not yet
overridden, so the migration is the override progression, not a runtime toggle.

Step 6a lands the skeleton only: no overrides, so behavior is identical to v1. The egress overrides
land in 6b/6c.
"""

from __future__ import annotations

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager


class MCPServerManagerV2(MCPServerManager):
    """v2 egress manager; see the module docstring. No overrides yet (step 6a)."""
