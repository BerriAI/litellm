"""
Common utility functions for handling object permission updates across
organizations, teams, and keys.
"""

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from fastapi import HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient

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
        object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": object_permission_id},
            )
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
    object_permission_id_to_use: str = existing_object_permission_id or str(
        uuid.uuid4()
    )
    existing_object_permissions_dict: Dict = {}

    existing_object_permission = (
        await prisma_client.db.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": object_permission_id_to_use},
        )
    )

    # Update the object permission
    if existing_object_permission is not None:
        existing_object_permissions_dict = existing_object_permission.model_dump(
            exclude_unset=True, exclude_none=True
        )

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
    created_object_permission_row = (
        await prisma_client.db.litellm_objectpermissiontable.upsert(
            where={"object_permission_id": object_permission_id_to_use},
            data={
                "create": existing_object_permissions_dict,
                "update": existing_object_permissions_dict,
            },
        )
    )

    verbose_proxy_logger.debug(
        f"created_object_permission_row: {created_object_permission_row}"
    )

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
    clean_data = {
        k: v
        for k, v in permission_data.items()
        if v is not None and k != "object_permission_id"
    }

    # Serialize mcp_tool_permissions to JSON string for GraphQL compatibility
    if "mcp_tool_permissions" in clean_data:
        clean_data["mcp_tool_permissions"] = safe_dumps(
            clean_data["mcp_tool_permissions"]
        )

    created_permission = await prisma_client.db.litellm_objectpermissiontable.create(
        data=clean_data
    )

    data_json["object_permission_id"] = created_permission.object_permission_id
    data_json.pop("object_permission")
    return data_json


async def _resolve_team_allowed_mcp_servers(
    team_object_permission: "LiteLLM_ObjectPermissionTable",
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
    access_group_servers: List[str] = (
        await MCPRequestHandler._get_mcp_servers_from_access_groups(
            team_object_permission.mcp_access_groups or []
        )
    )
    raw_tool_perms = team_object_permission.mcp_tool_permissions or {}
    if isinstance(raw_tool_perms, str):
        try:
            raw_tool_perms = json.loads(raw_tool_perms)
        except json.JSONDecodeError:
            verbose_proxy_logger.warning(
                "Failed to deserialize mcp_tool_permissions as JSON; treating as empty. "
                "Value: %r",
                raw_tool_perms,
            )
            raw_tool_perms = {}
    tool_perm_servers: List[str] = list(raw_tool_perms.keys())
    return set(direct_servers + access_group_servers + tool_perm_servers)


def _get_allow_all_keys_server_ids() -> Set[str]:
    """Return the set of MCP server IDs marked with allow_all_keys=True."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    return set(global_mcp_server_manager.get_allow_all_keys_server_ids())


async def _get_team_allowed_mcp_servers(
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
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

    return await _resolve_team_allowed_mcp_servers(team_object_permission)


def _extract_requested_mcp_server_ids(
    object_permission: Optional[dict],
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

    mcp_tool_permissions = object_permission.get("mcp_tool_permissions")
    if isinstance(mcp_tool_permissions, dict):
        server_ids.update(mcp_tool_permissions.keys())

    return server_ids


def _extract_requested_mcp_access_groups(
    object_permission: Optional[dict],
) -> Set[str]:
    """Extract MCP access groups from a key's object_permission dict."""
    if not object_permission or not isinstance(object_permission, dict):
        return set()

    groups = object_permission.get("mcp_access_groups")
    if isinstance(groups, list):
        return set(groups)
    return set()


async def validate_key_mcp_servers_against_team(
    object_permission: Optional[dict],
    team_obj: Optional["LiteLLM_TeamTableCachedObj"],
):
    """
    Validate that MCP servers requested on a key are within the allowed scope.

    Rules:
    - If key is in a team: key's mcp_servers must be a subset of
      (team's allowed servers + allow_all_keys servers)
    - If key is NOT in a team: key's mcp_servers must only contain
      allow_all_keys servers
    - If team has no MCP config: key can only use allow_all_keys servers

    Raises HTTPException(403) if validation fails.
    """
    requested_servers = _extract_requested_mcp_server_ids(object_permission)
    requested_access_groups = _extract_requested_mcp_access_groups(object_permission)

    # Nothing to validate
    if not requested_servers and not requested_access_groups:
        return

    allow_all_keys_servers = _get_allow_all_keys_server_ids()
    team_allowed_servers = await _get_team_allowed_mcp_servers(team_obj)

    # Combined allowed set = team servers + allow_all_keys servers
    all_allowed_servers = team_allowed_servers | allow_all_keys_servers

    # Validate requested server IDs
    if requested_servers:
        disallowed_servers = requested_servers - all_allowed_servers
        if disallowed_servers:
            if team_obj is not None:
                detail = (
                    f"Key requests MCP servers not allowed by team '{team_obj.team_id}': "
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

        disallowed_groups = requested_access_groups - team_access_groups
        if disallowed_groups:
            if team_obj is not None:
                detail = (
                    f"Key requests MCP access groups not allowed by team '{team_obj.team_id}': "
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
