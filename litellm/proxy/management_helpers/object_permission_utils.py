"""
Common utility functions for handling object permission updates across
organizations, teams, and keys.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from fastapi import HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import ObjectPermissionDict, SpecialMCPServerName, SpecialMCPServerNames
from litellm.proxy.utils import PrismaClient
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.repositories.table_repositories import MCPServerRepository

if TYPE_CHECKING:
    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionTable,
        LiteLLM_TeamTableCachedObj,
    )


async def attach_object_permission_to_dict(
    data_dict: Dict,
    prisma_client: PrismaClient,
) -> Dict:
    """
    Helper method to attach object_permission to a dictionary if object_permission_id is set.

    This function:
    1. Checks if the dictionary has an object_permission_id
    2. If found, queries the database for the corresponding object permission
    3. Converts the object permission to a dictionary format
    4. Attaches it to the input dictionary under the 'object_permission' key

    Args:
        data_dict: The dictionary to attach object_permission to
        prisma_client: The database client

    Returns:
        Dict: The input dictionary with object_permission attached if found

    Raises:
        ValueError: If prisma_client is None
    """
    if prisma_client is None:
        raise ValueError("Prisma client not found")

    object_permission_id = data_dict.get("object_permission_id")
    if object_permission_id:
        object_permission = await ObjectPermissionRepository(prisma_client).table.find_unique(
            where={"object_permission_id": object_permission_id},
        )
        if object_permission:
            # Convert to dict if needed
            try:
                object_permission = object_permission.model_dump()
            except Exception:
                object_permission = object_permission.dict()
            data_dict["object_permission"] = object_permission
    return data_dict


async def handle_update_object_permission_common(
    data_json: Dict,
    existing_object_permission_id: Optional[str],
    prisma_client: Optional[PrismaClient],
) -> Optional[str]:
    """
    Common logic for handling object permission updates across organizations, teams, and keys.

    This function:
    1. Extracts `object_permission` from data_json
    2. Looks up existing object permission if it exists
    3. Merges new permissions with existing ones
    4. Upserts to the LiteLLM_ObjectPermissionTable
    5. Returns the object_permission_id

    Args:
        data_json: The data dictionary containing the object_permission to update
        existing_object_permission_id: The current object_permission_id from the entity (can be None)
        prisma_client: The database client

    Returns:
        Optional[str]: The object_permission_id after the update/creation, or None if no object_permission to process

    Raises:
        ValueError: If prisma_client is None
    """
    if prisma_client is None:
        raise ValueError("Prisma client not found")

    #########################################################
    # Ensure `object_permission` is not added to the data_json
    # We need to update the entity at the object_permission_id level in the LiteLLM_ObjectPermissionTable
    #########################################################
    new_object_permission: Union[dict, str] = data_json.pop("object_permission", None)
    if new_object_permission is None:
        return None

    # Lookup existing object permission ID and update that entry
    object_permission_id_to_use: str = existing_object_permission_id or str(uuid.uuid4())
    existing_object_permissions_dict: Dict = {}

    existing_object_permission = await ObjectPermissionRepository(prisma_client).table.find_unique(
        where={"object_permission_id": object_permission_id_to_use},
    )

    # Update the object permission
    if existing_object_permission is not None:
        existing_object_permissions_dict = existing_object_permission.model_dump(exclude_unset=True, exclude_none=True)

    # Handle string JSON object permission
    if isinstance(new_object_permission, str):
        new_object_permission = json.loads(new_object_permission)

    if isinstance(new_object_permission, dict):
        existing_object_permissions_dict.update(new_object_permission)

    #########################################################
    # Serialize mcp_tool_permissions JSON field to avoid GraphQL parsing issues
    # (e.g., server IDs starting with "3e64" being interpreted as floats)
    #########################################################
    if "mcp_tool_permissions" in existing_object_permissions_dict:
        existing_object_permissions_dict["mcp_tool_permissions"] = safe_dumps(
            existing_object_permissions_dict["mcp_tool_permissions"]
        )

    #########################################################
    # Commit the update to the LiteLLM_ObjectPermissionTable
    #########################################################
    created_object_permission_row = await ObjectPermissionRepository(prisma_client).table.upsert(
        where={"object_permission_id": object_permission_id_to_use},
        data={
            "create": existing_object_permissions_dict,
            "update": existing_object_permissions_dict,
        },
    )

    verbose_proxy_logger.debug(f"created_object_permission_row: {created_object_permission_row}")

    return created_object_permission_row.object_permission_id


async def _set_object_permission(
    data_json: dict,
    prisma_client: Optional[PrismaClient],
):
    """
    Creates the LiteLLM_ObjectPermissionTable record for the key/team.
    Handles permissions for vector stores and mcp servers.
    """
    if prisma_client is None or "object_permission" not in data_json:
        return data_json

    permission_data = data_json["object_permission"]
    if not isinstance(permission_data, dict):
        data_json.pop("object_permission")
        return data_json

    # Clean data: exclude None values and object_permission_id
    clean_data = {k: v for k, v in permission_data.items() if v is not None and k != "object_permission_id"}

    # Serialize mcp_tool_permissions to JSON string for GraphQL compatibility
    if "mcp_tool_permissions" in clean_data:
        clean_data["mcp_tool_permissions"] = safe_dumps(clean_data["mcp_tool_permissions"])

    created_permission = await ObjectPermissionRepository(prisma_client).table.create(data=clean_data)

    data_json["object_permission_id"] = created_permission.object_permission_id
    data_json.pop("object_permission")
    return data_json


def _dedupe_preserving_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _mcp_server_identifier_matches(server: Any, identifier: str) -> bool:
    return identifier in {
        getattr(server, "server_id", None),
        getattr(server, "alias", None),
        getattr(server, "server_name", None),
        getattr(server, "name", None),
    }


async def _get_db_mcp_servers_by_identifiers(
    identifiers: Set[str],
    prisma_client: Optional[PrismaClient],
) -> List[Any]:
    if prisma_client is None or not identifiers:
        return []

    identifier_list = list(identifiers)
    return await MCPServerRepository(prisma_client).table.find_many(
        where={
            "OR": [
                {"server_id": {"in": identifier_list}},
                {"alias": {"in": identifier_list}},
                {"server_name": {"in": identifier_list}},
            ]
        }
    )


async def _resolve_mcp_server_identifiers_to_ids(
    identifiers: Set[str],
    prisma_client: Optional[PrismaClient],
) -> Dict[str, Set[str]]:
    """
    Resolve MCP permission entries written as server_id, alias, or server_name
    to canonical server IDs.

    DB rows are authoritative when available; the in-memory registry is still
    consulted for config-file servers, which are not persisted in the MCP table.
    """
    if not identifiers:
        return {}

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    resolved: Dict[str, Set[str]] = {identifier: set() for identifier in identifiers}

    for server in await _get_db_mcp_servers_by_identifiers(
        identifiers=identifiers,
        prisma_client=prisma_client,
    ):
        server_id = getattr(server, "server_id", None)
        if not server_id:
            continue
        for identifier in identifiers:
            if _mcp_server_identifier_matches(server, identifier):
                resolved[identifier].add(server_id)

    for registry_key, server in global_mcp_server_manager.get_registry().items():
        server_id = getattr(server, "server_id", None) or registry_key
        if not server_id:
            continue
        for identifier in identifiers:
            if identifier == registry_key or _mcp_server_identifier_matches(server, identifier):
                resolved[identifier].add(server_id)

    return resolved


def _rewrite_object_permission_mcp_servers(
    object_permission: ObjectPermissionDict,
    identifier_to_server_ids: Dict[str, Set[str]],
) -> None:
    mcp_servers = object_permission.get("mcp_servers")
    if not isinstance(mcp_servers, list):
        return

    normalized_servers: List[str] = []
    for identifier in mcp_servers:
        if identifier == SpecialMCPServerNames.no_mcp_servers.value:
            normalized_servers.append(SpecialMCPServerNames.no_mcp_servers.value)
            continue
        normalized_servers.extend(sorted(identifier_to_server_ids.get(identifier, [])))
    object_permission["mcp_servers"] = _dedupe_preserving_order(normalized_servers)


def _rewrite_object_permission_mcp_tool_permissions(
    object_permission: ObjectPermissionDict,
    identifier_to_server_ids: Dict[str, Set[str]],
) -> None:
    mcp_tool_permissions = object_permission.get("mcp_tool_permissions")
    if not isinstance(mcp_tool_permissions, dict):
        return

    normalized_tool_permissions: Dict[str, List[str]] = {}
    for identifier, tools in mcp_tool_permissions.items():
        if not isinstance(tools, list):
            tools = []
        for server_id in sorted(identifier_to_server_ids.get(identifier, [])):
            normalized_tool_permissions.setdefault(server_id, [])
            normalized_tool_permissions[server_id].extend(tools)

    object_permission["mcp_tool_permissions"] = {
        server_id: _dedupe_preserving_order(tools) for server_id, tools in normalized_tool_permissions.items()
    }


def _rewrite_object_permission_mcp_identifiers(
    object_permission: Optional[ObjectPermissionDict],
    identifier_to_server_ids: Dict[str, Set[str]],
) -> None:
    if not object_permission or not isinstance(object_permission, dict):
        return

    _rewrite_object_permission_mcp_servers(
        object_permission=object_permission,
        identifier_to_server_ids=identifier_to_server_ids,
    )
    _rewrite_object_permission_mcp_tool_permissions(
        object_permission=object_permission,
        identifier_to_server_ids=identifier_to_server_ids,
    )


def _flatten_resolved_mcp_server_ids(
    identifier_to_server_ids: Dict[str, Set[str]],
) -> Set[str]:
    return {server_id for server_ids in identifier_to_server_ids.values() for server_id in server_ids}


async def _resolve_team_allowed_mcp_servers(
    team_object_permission: "LiteLLM_ObjectPermissionTable",
    prisma_client: Optional[PrismaClient] = None,
) -> Set[str]:
    """
    Resolve the full set of MCP server IDs a team has access to.

    Combines:
    - Direct mcp_servers list
    - Servers from mcp_access_groups
    - Server IDs referenced in mcp_tool_permissions keys
    """
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    direct_servers: List[str] = team_object_permission.mcp_servers or []
    if SpecialMCPServerName.all_proxy_servers.value in direct_servers:
        return _get_all_mcp_server_ids()
    access_group_servers: List[str] = await MCPRequestHandler._get_mcp_servers_from_access_groups(
        team_object_permission.mcp_access_groups or []
    )
    raw_tool_perms = team_object_permission.mcp_tool_permissions or {}
    if isinstance(raw_tool_perms, str):
        raw_tool_perms = json.loads(raw_tool_perms)
    tool_perm_servers: List[str] = list(raw_tool_perms.keys())
    raw_servers = set(direct_servers + access_group_servers + tool_perm_servers)
    resolved_servers = await _resolve_mcp_server_identifiers_to_ids(
        identifiers=raw_servers,
        prisma_client=prisma_client,
    )
    unresolved_servers = {server_id for server_id in raw_servers if not resolved_servers.get(server_id)}
    return _flatten_resolved_mcp_server_ids(resolved_servers) | unresolved_servers


def _get_allow_all_keys_server_ids() -> Set[str]:
    """Return the set of MCP server IDs marked with allow_all_keys=True."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    return set(global_mcp_server_manager.get_allow_all_keys_server_ids())


def _get_all_mcp_server_ids() -> set[str]:
    """Return every MCP server id registered on the proxy (config + DB union)."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    return set(global_mcp_server_manager.get_registry().keys())


async def _existing_object_permission_mcp_servers(
    object_permission_id: Optional[str],
    prisma_client: Optional[PrismaClient],
) -> list[str]:
    if not object_permission_id or prisma_client is None:
        return []
    existing = await ObjectPermissionRepository(prisma_client).table.find_unique(
        where={"object_permission_id": object_permission_id},
    )
    if existing is None:
        return []
    return existing.mcp_servers or []


async def enforce_all_proxy_mcp_servers_grant_is_admin_only(
    requested_mcp_servers: Optional[list[str]],
    existing_object_permission_id: Optional[str],
    is_proxy_admin: bool,
    prisma_client: Optional[PrismaClient],
) -> None:
    """
    Only a proxy admin may newly grant the all-proxy MCP sentinel.

    Scoping a team to every MCP server on the proxy is a proxy-wide authorization
    decision, so a caller who is not a proxy admin (e.g. a team admin managing their
    own team) cannot add ``all-proxy-mcpservers``. A sentinel a proxy admin already
    granted is left untouched, so unrelated edits to such a team still succeed.

    Raises HTTPException(403) when a non-admin tries to add the sentinel.
    """
    sentinel = SpecialMCPServerName.all_proxy_servers.value
    if is_proxy_admin or sentinel not in (requested_mcp_servers or []):
        return
    existing_mcp_servers = await _existing_object_permission_mcp_servers(
        object_permission_id=existing_object_permission_id,
        prisma_client=prisma_client,
    )
    if sentinel in existing_mcp_servers:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "Only a proxy admin can grant a team access to all proxy MCP servers ('all-proxy-mcpservers')."
        },
    )


async def _get_team_allowed_mcp_servers(
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
    prisma_client: Optional[PrismaClient] = None,
) -> Set[str]:
    """
    Get the full set of MCP server IDs a team allows.

    If team has no object_permission or no MCP config, returns empty set
    (meaning only allow_all_keys servers are permitted).
    """
    if team_obj is None:
        return set()

    team_object_permission = team_obj.object_permission
    if team_object_permission is None:
        return set()

    return await _resolve_team_allowed_mcp_servers(
        team_object_permission=team_object_permission,
        prisma_client=prisma_client,
    )


def _extract_requested_mcp_server_ids(
    object_permission: Optional[ObjectPermissionDict],
) -> Set[str]:
    """
    Extract all MCP server IDs referenced in a key's object_permission dict.

    Includes:
    - mcp_servers list
    - Keys from mcp_tool_permissions
    """
    if not object_permission or not isinstance(object_permission, dict):
        return set()

    server_ids: Set[str] = set()
    mcp_servers = object_permission.get("mcp_servers")
    if isinstance(mcp_servers, list):
        server_ids.update(mcp_servers)
        server_ids.discard(SpecialMCPServerNames.no_mcp_servers.value)

    mcp_tool_permissions = object_permission.get("mcp_tool_permissions")
    if isinstance(mcp_tool_permissions, dict):
        server_ids.update(mcp_tool_permissions.keys())

    return server_ids


def _extract_requested_mcp_access_groups(
    object_permission: Optional[ObjectPermissionDict],
) -> Set[str]:
    """Extract MCP access groups from a key's object_permission dict."""
    if not object_permission or not isinstance(object_permission, dict):
        return set()

    groups = object_permission.get("mcp_access_groups")
    if isinstance(groups, list):
        return set(groups)
    return set()


def _extract_requested_mcp_toolsets(
    object_permission: Optional[ObjectPermissionDict],
) -> Set[str]:
    """Extract MCP toolset IDs from a key's object_permission dict."""
    if not object_permission or not isinstance(object_permission, dict):
        return set()

    toolsets = object_permission.get("mcp_toolsets")
    if isinstance(toolsets, list):
        return set(toolsets)
    return set()


async def validate_key_mcp_servers_against_team(
    object_permission: Optional[ObjectPermissionDict],
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
    prisma_client: Optional[PrismaClient] = None,
    is_proxy_admin: bool = False,
) -> Optional[ObjectPermissionDict]:
    """
    Validate that MCP servers requested on a key are within the allowed scope.

    Rules:
    - If key is in a team: key's mcp_servers must be a subset of
      (team's allowed servers + allow_all_keys servers)
    - If key is NOT in a team and the caller is a proxy admin: any server or
      access group may be assigned. A proxy admin can already reach every MCP
      server, and runtime access is granted directly from the key's own
      object_permission, so the key is scoped to exactly what the admin selected
    - If key is NOT in a team and the caller is not a proxy admin: key's
      mcp_servers must only contain allow_all_keys servers
    - If team has no MCP config: key can only use allow_all_keys servers

    Raises HTTPException(403) if validation fails.
    """
    teamless_admin_assignment = team_obj is None and is_proxy_admin
    requested_servers = _extract_requested_mcp_server_ids(object_permission)
    requested_access_groups = _extract_requested_mcp_access_groups(object_permission)

    requested_toolsets = _extract_requested_mcp_toolsets(object_permission)

    # Nothing to validate
    if not requested_servers and not requested_access_groups and not requested_toolsets:
        return object_permission

    allow_all_keys_servers = _get_allow_all_keys_server_ids()
    team_allowed_servers = await _get_team_allowed_mcp_servers(
        team_obj=team_obj,
        prisma_client=prisma_client,
    )

    # Combined allowed set = team servers + allow_all_keys servers
    all_allowed_servers = team_allowed_servers | allow_all_keys_servers

    # Validate requested server IDs
    if requested_servers:
        # Normalize aliases/names before authorization. Only entries that do not
        # resolve to a server in the DB or config registry are treated as stale.
        identifier_to_server_ids = await _resolve_mcp_server_identifiers_to_ids(
            identifiers=requested_servers,
            prisma_client=prisma_client,
        )
        stale_identifiers = {
            identifier for identifier in requested_servers if not identifier_to_server_ids.get(identifier)
        }
        if stale_identifiers:
            verbose_proxy_logger.warning(
                "validate_key_mcp_servers_against_team: ignoring stale MCP server "
                f"identifiers (no longer in registry or DB): {sorted(stale_identifiers)}"
            )
        _rewrite_object_permission_mcp_identifiers(
            object_permission=object_permission,
            identifier_to_server_ids=identifier_to_server_ids,
        )
        active_requested_servers = _flatten_resolved_mcp_server_ids(identifier_to_server_ids)

        allowed_servers = all_allowed_servers
        if teamless_admin_assignment:
            allowed_servers = all_allowed_servers | active_requested_servers

        disallowed_servers = active_requested_servers - allowed_servers
        if disallowed_servers:
            if team_obj is not None:
                team_id = team_obj.team_id
                detail = (
                    f"Key requests MCP servers not allowed by team '{team_id}': "
                    f"{sorted(disallowed_servers)}. "
                    f"Team allows: {sorted(team_allowed_servers)}. "
                    f"Global (allow_all_keys) servers: {sorted(allow_all_keys_servers)}."
                )
            else:
                detail = (
                    f"Key is not in a team. Only globally available (allow_all_keys) MCP servers "
                    f"can be assigned: {sorted(allow_all_keys_servers)}. "
                    f"Disallowed servers: {sorted(disallowed_servers)}."
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": detail},
            )

    # Validate requested access groups (must be subset of team's access groups)
    if requested_access_groups:
        team_access_groups: Set[str] = set()
        if (
            team_obj is not None
            and team_obj.object_permission is not None
            and team_obj.object_permission.mcp_access_groups
        ):
            team_access_groups = set(team_obj.object_permission.mcp_access_groups)

        allowed_access_groups = team_access_groups
        if teamless_admin_assignment:
            allowed_access_groups = team_access_groups | requested_access_groups

        disallowed_groups = requested_access_groups - allowed_access_groups
        if disallowed_groups:
            if team_obj is not None:
                team_id = team_obj.team_id
                detail = (
                    f"Key requests MCP access groups not allowed by team '{team_id}': "
                    f"{sorted(disallowed_groups)}. "
                    f"Team allows: {sorted(team_access_groups)}."
                )
            else:
                detail = (
                    f"Key is not in a team. MCP access groups cannot be assigned to "
                    f"keys outside of a team. Disallowed groups: {sorted(disallowed_groups)}."
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": detail},
            )

    _validate_requested_toolsets(
        requested_toolsets=requested_toolsets,
        team_obj=team_obj,
        is_proxy_admin=is_proxy_admin,
    )

    return object_permission


def _validate_requested_toolsets(
    requested_toolsets: set[str],
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
    is_proxy_admin: bool,
) -> None:
    """
    Validate mcp_toolsets requested on a key.

    Non-admin callers cannot assign toolsets to a personal (no team) key. Team
    keys must request a subset of the team's own toolset allowlist.
    """
    if not requested_toolsets:
        return
    if team_obj is None:
        if is_proxy_admin:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "Key is not in a team. MCP toolsets cannot be assigned to "
                    "personal keys by non-admin callers. Disallowed toolsets: "
                    f"{sorted(requested_toolsets)}."
                )
            },
        )
    team_op = team_obj.object_permission
    team_mcp_toolsets = team_op.mcp_toolsets if team_op is not None else None
    if not team_mcp_toolsets:
        return
    disallowed_toolsets = requested_toolsets - set(team_mcp_toolsets)
    if not disallowed_toolsets:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": (
                f"Key requests MCP toolsets not allowed by team '{team_obj.team_id}': "
                f"{sorted(disallowed_toolsets)}. "
                f"Team allows: {sorted(team_mcp_toolsets)}."
            )
        },
    )


def _extract_requested_vector_stores(
    object_permission: Optional[ObjectPermissionDict],
) -> set[str]:
    """Return vector_store IDs from a key's object_permission dict."""
    if not object_permission or not isinstance(object_permission, dict):
        return set()
    raw = object_permission.get("vector_stores")
    if isinstance(raw, list):
        return {str(x) for x in raw if x}
    return set()


async def validate_key_vector_stores_against_team(
    object_permission: Optional[ObjectPermissionDict],
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
    is_proxy_admin: bool = False,
) -> None:
    """
    Reject vector_stores requested on a personal (no team) key by a non-admin
    caller. Vector store access is granted at use-time from the key's
    object_permission.vector_stores list, so the assignment is the authorization
    boundary. Team keys and proxy admins are unaffected.
    """
    requested = _extract_requested_vector_stores(object_permission)
    if not requested:
        return
    if team_obj is not None or is_proxy_admin:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": (
                "Key is not in a team. Vector stores cannot be assigned to "
                "personal keys by non-admin callers. Disallowed vector stores: "
                f"{sorted(requested)}."
            )
        },
    )


def _extract_requested_search_tools(
    object_permission: Optional[ObjectPermissionDict],
) -> list[str]:
    """Return search_tool_name values from a key's object_permission dict."""
    if not object_permission or not isinstance(object_permission, dict):
        return []
    raw = object_permission.get("search_tools")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x]


async def validate_key_search_tools_against_team(
    object_permission: Optional[ObjectPermissionDict],
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
    is_proxy_admin: bool = False,
) -> None:
    """
    Validate key object_permission.search_tools is a subset of the team's allowlist.

    Empty team allowlist means no restriction at team layer (skip).
    Non-admin callers cannot assign search_tools to a personal (no team) key.
    """
    requested = _extract_requested_search_tools(object_permission)
    if not requested:
        return

    if team_obj is None and not is_proxy_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "Key is not in a team. search_tools cannot be assigned to "
                    "personal keys by non-admin callers. Disallowed search tools: "
                    f"{sorted(requested)}."
                )
            },
        )

    team_tools: List[str] = []
    if team_obj is not None and team_obj.object_permission is not None:
        st = team_obj.object_permission.search_tools
        if st:
            team_tools = list(st)

    if not team_tools:
        return

    disallowed = set(requested) - set(team_tools)
    if disallowed:
        team_id = team_obj.team_id if team_obj is not None else "unknown"
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    f"Key requests search tools not allowed by team '{team_id}': "
                    f"{sorted(disallowed)}. Team allows: {sorted(team_tools)}."
                )
            },
        )
