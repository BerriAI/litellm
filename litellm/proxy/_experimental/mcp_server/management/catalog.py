"""Static catalog of the built-in management MCP tools.

Each entry maps one MCP tool onto exactly one proxy management REST handler.
The input schema is generated from a pydantic model so the wire contract and
the dispatcher's argument validation can never drift apart.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import mcp.types as mcp_types
from pydantic import BaseModel, Field

from litellm.proxy._types import (
    GenerateKeyRequest,
    KeyRequest,
    UpdateKeyRequest,
)
from litellm.types.access_group import (
    AccessGroupCreateRequest,
    AccessGroupUpdateRequest,
)


class ListVirtualKeysArguments(BaseModel, frozen=True):
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")
    user_id: str | None = Field(
        None,
        description="Filter keys by user ID. Exact match by default; set substring_matching=true (admin only) for case-insensitive substring matching.",
    )
    team_id: str | None = Field(None, description="Filter keys by team ID")
    organization_id: str | None = Field(None, description="Filter keys by organization ID")
    key_hash: str | None = Field(None, description="Filter keys by key hash")
    key_alias: str | None = Field(
        None,
        description="Filter keys by key alias. Exact match by default; set substring_matching=true for case-insensitive substring matching.",
    )
    search: str | None = Field(
        None,
        description="Combined search: matches keys whose token (key hash) equals the value OR whose key_alias contains it (case-insensitive).",
    )
    return_full_object: bool = Field(False, description="Return full key object")
    include_team_keys: bool = Field(False, description="Include all keys for teams that user is an admin of.")
    include_created_by_keys: bool = Field(False, description="Include keys created by the user")
    sort_by: str | None = Field(None, description="Column to sort by (e.g. 'user_id', 'created_at', 'spend')")
    sort_order: str = Field("desc", description="Sort order ('asc' or 'desc')")
    expand: list[str] | None = Field(None, description="Expand related objects (e.g. 'user')")
    status: str | None = Field(
        None,
        description="Filter by status: 'active', 'expired', 'revoked' or 'deleted'. Omit to return live keys regardless of status.",
    )
    project_id: str | None = Field(None, description="Filter keys by project ID")
    access_group_id: str | None = Field(None, description="Filter keys by access group ID")
    agent_id: str | None = Field(None, description="Filter keys by agent ID")
    substring_matching: bool = Field(
        False,
        description="If true, match key_alias (any caller) and user_id (proxy admins only) as case-insensitive substrings instead of exact values.",
    )
    expires: str | None = Field(
        None,
        description="Filter keys by expiration: 'expired' or 'active'. Omit to return keys regardless of expiration.",
    )


class GetVirtualKeyArguments(BaseModel, frozen=True):
    key: str = Field(
        description="Key to look up. Prefer the key's sha256 hash so the raw key stays out of URLs and logs."
    )


class AccessGroupIdArguments(BaseModel, frozen=True):
    access_group_id: str = Field(description="The access group ID")


class UpdateAccessGroupArguments(BaseModel, frozen=True):
    access_group_id: str = Field(description="The access group ID")
    data: AccessGroupUpdateRequest


class NoArguments(BaseModel, frozen=True):
    pass


@dataclass(frozen=True, slots=True)
class ManagementTool:
    name: str
    title: str
    description: str
    http_method: str
    rest_path: str
    arguments_model: type[BaseModel]
    read_only: bool
    destructive: bool
    idempotent: bool


_ADMIN_NOTE: Final = " Requires a proxy-admin LiteLLM key."

MANAGEMENT_TOOLS: Final[tuple[ManagementTool, ...]] = (
    ManagementTool(
        name="list_virtual_keys",
        title="List virtual keys",
        description="List virtual API keys with pagination and filters. Calls GET /key/list." + _ADMIN_NOTE,
        http_method="GET",
        rest_path="/key/list",
        arguments_model=ListVirtualKeysArguments,
        read_only=True,
        destructive=False,
        idempotent=True,
    ),
    ManagementTool(
        name="get_virtual_key",
        title="Get virtual key info",
        description="Get details for one virtual key. Calls GET /key/info." + _ADMIN_NOTE,
        http_method="GET",
        rest_path="/key/info",
        arguments_model=GetVirtualKeyArguments,
        read_only=True,
        destructive=False,
        idempotent=True,
    ),
    ManagementTool(
        name="create_virtual_key",
        title="Create virtual key",
        description="Generate a new virtual API key. Calls POST /key/generate." + _ADMIN_NOTE,
        http_method="POST",
        rest_path="/key/generate",
        arguments_model=GenerateKeyRequest,
        read_only=False,
        destructive=False,
        idempotent=False,
    ),
    ManagementTool(
        name="update_virtual_key",
        title="Update virtual key",
        description="Update an existing virtual key (merge patch semantics). Calls POST /key/update." + _ADMIN_NOTE,
        http_method="POST",
        rest_path="/key/update",
        arguments_model=UpdateKeyRequest,
        read_only=False,
        destructive=False,
        idempotent=False,
    ),
    ManagementTool(
        name="delete_virtual_keys",
        title="Delete virtual keys",
        description="Permanently delete virtual keys by token or alias. Calls POST /key/delete." + _ADMIN_NOTE,
        http_method="POST",
        rest_path="/key/delete",
        arguments_model=KeyRequest,
        read_only=False,
        destructive=True,
        idempotent=False,
    ),
    ManagementTool(
        name="list_access_groups",
        title="List access groups",
        description="List all access groups. Calls GET /v1/access_group." + _ADMIN_NOTE,
        http_method="GET",
        rest_path="/v1/access_group",
        arguments_model=NoArguments,
        read_only=True,
        destructive=False,
        idempotent=True,
    ),
    ManagementTool(
        name="get_access_group",
        title="Get access group",
        description="Get one access group by ID. Calls GET /v1/access_group/{access_group_id}." + _ADMIN_NOTE,
        http_method="GET",
        rest_path="/v1/access_group/{access_group_id}",
        arguments_model=AccessGroupIdArguments,
        read_only=True,
        destructive=False,
        idempotent=True,
    ),
    ManagementTool(
        name="create_access_group",
        title="Create access group",
        description="Create a new access group. Calls POST /v1/access_group." + _ADMIN_NOTE,
        http_method="POST",
        rest_path="/v1/access_group",
        arguments_model=AccessGroupCreateRequest,
        read_only=False,
        destructive=False,
        idempotent=False,
    ),
    ManagementTool(
        name="update_access_group",
        title="Update access group",
        description="Update an existing access group. Calls PUT /v1/access_group/{access_group_id}." + _ADMIN_NOTE,
        http_method="PUT",
        rest_path="/v1/access_group/{access_group_id}",
        arguments_model=UpdateAccessGroupArguments,
        read_only=False,
        destructive=False,
        idempotent=True,
    ),
    ManagementTool(
        name="delete_access_group",
        title="Delete access group",
        description="Permanently delete an access group. Calls DELETE /v1/access_group/{access_group_id}."
        + _ADMIN_NOTE,
        http_method="DELETE",
        rest_path="/v1/access_group/{access_group_id}",
        arguments_model=AccessGroupIdArguments,
        read_only=False,
        destructive=True,
        idempotent=True,
    ),
)

MANAGEMENT_TOOLS_BY_NAME: Final[Mapping[str, ManagementTool]] = MappingProxyType(
    {tool.name: tool for tool in MANAGEMENT_TOOLS}
)


def path_param_names(tool: ManagementTool) -> frozenset[str]:
    return frozenset(
        segment[1:-1] for segment in tool.rest_path.split("/") if segment.startswith("{") and segment.endswith("}")
    )


def mcp_tools() -> tuple[mcp_types.Tool, ...]:
    return tuple(
        mcp_types.Tool(
            name=tool.name,
            title=tool.title,
            description=tool.description,
            input_schema=tool.arguments_model.model_json_schema(),
            annotations=mcp_types.ToolAnnotations(
                read_only_hint=tool.read_only,
                destructive_hint=tool.destructive,
                idempotent_hint=tool.idempotent,
            ),
        )
        for tool in MANAGEMENT_TOOLS
    )
