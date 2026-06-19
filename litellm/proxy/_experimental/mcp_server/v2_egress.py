"""v2 MCP egress transport (the chokepoint): scaffolding.

This phase makes v2 own the upstream MCP connection. When ``LITELLM_USE_V2_MCP_EGRESS`` is enabled,
a v2 manager (built in later steps) implements the handler-facing egress surface via an
``UpstreamConnection`` that attaches ``resolve()``'s ``httpx.Auth`` plus resolved static/env-var
headers directly to the SDK client, replacing v1's ``_create_mcp_client`` and the
``resolve_mcp_auth`` header graft.

Step 1 lands only the flag and the egress contract (``MCPEgressManager``). The implementation
(``UpstreamConnection``, the static-headers resolver, the per-user token bridges,
``MCPServerManagerV2``) arrives in later steps; the contract grows with the surface as it is
implemented (call_tool/dispatch and the reused registry/RBAC lookups are added when the v2 manager
is assembled). The CLI flag wiring lands at the cutover step.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Union

if TYPE_CHECKING:
    from mcp import ReadResourceResult, Resource
    from mcp.types import GetPromptResult, Prompt
    from mcp.types import Tool as MCPTool
    from pydantic import AnyUrl

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

_V2_EGRESS_ENV_FLAG = "LITELLM_USE_V2_MCP_EGRESS"


def v2_egress_enabled() -> bool:
    """True when v2 owns the MCP egress transport (set via the env flag)."""
    return os.getenv(_V2_EGRESS_ENV_FLAG, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class MCPEgressManager(Protocol):
    """The per-server egress operations the inbound handler invokes on the manager.

    v1's ``MCPServerManager`` satisfies this today; ``MCPServerManagerV2`` (later steps) will
    implement it via the ``UpstreamConnection``. Registry/RBAC lookups
    (``get_allowed_mcp_servers``, ``get_registry``, ...) are reused from v1 and are intentionally
    not part of this egress contract; ``call_tool``/dispatch is added when the v2 manager is
    assembled.
    """

    async def _get_tools_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[MCPTool]: ...

    async def get_prompts_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Prompt]: ...

    async def get_resources_from_server(
        self,
        server: MCPServer,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        add_prefix: bool = True,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> List[Resource]: ...

    async def read_resource_from_server(
        self,
        server: MCPServer,
        url: AnyUrl,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> ReadResourceResult: ...

    async def get_prompt_from_server(
        self,
        server: MCPServer,
        prompt_name: str,
        arguments: Optional[Dict[str, object]] = None,
        mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        raw_headers: Optional[Dict[str, str]] = None,
    ) -> GetPromptResult: ...
