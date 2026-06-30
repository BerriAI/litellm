"""
TypedDict mirror of ``LiteLLM_ObjectPermissionBase`` for in-memory dict
payloads passed through validators that mutate before persistence (e.g.
MCP server identifier normalization in object_permission_utils).

Lives in ``litellm/types/`` so SDK-side modules (``litellm.types.agents``)
can adopt the type without violating the SDK-must-not-import-from-proxy
layering rule.
"""

from typing import Optional

from typing_extensions import TypedDict


class ObjectPermissionDict(TypedDict, total=False):
    mcp_servers: Optional[list[str]]
    mcp_access_groups: Optional[list[str]]
    mcp_tool_permissions: Optional[dict[str, list[str]]]
    mcp_toolsets: Optional[list[str]]
    blocked_tools: Optional[list[str]]
    vector_stores: Optional[list[str]]
    agents: Optional[list[str]]
    agent_access_groups: Optional[list[str]]
    models: Optional[list[str]]
    search_tools: Optional[list[str]]
