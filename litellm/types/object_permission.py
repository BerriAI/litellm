"""
TypedDict mirror of ``LiteLLM_ObjectPermissionBase`` for in-memory dict
payloads passed through validators that mutate before persistence (e.g.
MCP server identifier normalization in object_permission_utils).

Lives in ``litellm/types/`` so SDK-side modules (``litellm.types.agents``)
can adopt the type without violating the SDK-must-not-import-from-proxy
layering rule.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

_PERMISSION: Final = TypeAdapter(Mapping[str, object])


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
    skills: ReadOnly[list[str] | None]
    apps: ReadOnly[Sequence[str] | None]
    mcp_auth_profiles: list[str] | None  # mutable-ok: JSON payload  # writable-ok: normalized in place


def without_empty_auth_profile_grants(permission: Mapping[str, object]) -> Mapping[str, object]:
    if permission.get("mcp_auth_profiles"):
        return permission
    return {key: value for key, value in permission.items() if key != "mcp_auth_profiles"}


def row_without_empty_auth_profile_grants(
    row: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: legacy callers update the returned row
    permission: Final = row.get("object_permission")
    if not isinstance(permission, Mapping):
        return dict(row)
    return {
        **row,
        "object_permission": dict(without_empty_auth_profile_grants(_PERMISSION.validate_python(permission))),
    }


def response_without_empty_auth_profile_grants(row: object) -> object:
    if not isinstance(row, BaseModel):
        return row
    permission: Final = getattr(row, "object_permission", None)
    if not isinstance(permission, BaseModel) or getattr(permission, "mcp_auth_profiles", None):
        return row
    return row_without_empty_auth_profile_grants(row.model_dump(mode="json"))


def empty_auth_profile_grants_exclusion(row: object) -> Mapping[str, Mapping[str, bool]] | None:
    permission: Final = getattr(row, "object_permission", None)
    if not isinstance(permission, BaseModel) or getattr(permission, "mcp_auth_profiles", None):
        return None
    return {"object_permission": {"mcp_auth_profiles": True}}
