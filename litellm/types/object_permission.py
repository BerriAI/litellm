"""
TypedDict mirror of ``LiteLLM_ObjectPermissionBase`` for in-memory dict
payloads passed through validators that mutate before persistence (e.g.
MCP server identifier normalization in object_permission_utils).

Lives in ``litellm/types/`` so SDK-side modules (``litellm.types.agents``)
can adopt the type without violating the SDK-must-not-import-from-proxy
layering rule.
"""

from typing_extensions import TypedDict


class ObjectPermissionDict(TypedDict, total=False):
    mcp_servers: list[str] | None
    mcp_access_groups: list[str] | None
    mcp_tool_permissions: dict[str, list[str]] | None
    mcp_toolsets: list[str] | None
    blocked_tools: list[str] | None
    vector_stores: list[str] | None
    agents: list[str] | None
    agent_access_groups: list[str] | None
    models: list[str] | None
    search_tools: list[str] | None
    mcp_tool_search_enabled: bool | None
