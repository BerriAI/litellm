"""
Common utility functions for handling object permission updates across
organizations, teams, and keys.
"""

import json
from typing import Dict, List, Optional, Set, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionBase,
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
from litellm.proxy.utils import PrismaClient
        


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
        object_permission = await prisma_client.db.litellm_objectpermissiontable.find_unique(
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
        k: v for k, v in permission_data.items()
        if v is not None and k != "object_permission_id"
    }
    
    # Serialize mcp_tool_permissions to JSON string for GraphQL compatibility
    if "mcp_tool_permissions" in clean_data:
        clean_data["mcp_tool_permissions"] = safe_dumps(clean_data["mcp_tool_permissions"])
    
    created_permission = await prisma_client.db.litellm_objectpermissiontable.create(
        data=clean_data
    )
    
    data_json["object_permission_id"] = created_permission.object_permission_id
    data_json.pop("object_permission")
    return data_json


async def get_team_mcp_permissions(
    team_obj: LiteLLM_TeamTableCachedObj,
) -> Optional[Dict]:
    """
    Returns the team's MCP permissions: {"mcp_servers": [...], "mcp_access_groups": [...]}.
    Resolves access groups to server IDs.
    Returns None if team has no object_permission.
    """
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    object_permission = getattr(team_obj, "object_permission", None)
    if object_permission is None:
        return None

    direct_servers: List[str] = object_permission.mcp_servers or []
    access_groups: List[str] = object_permission.mcp_access_groups or []
    tool_permission_keys: List[str] = list(
        (object_permission.mcp_tool_permissions or {}).keys()
    )

    # Resolve access groups to server IDs
    resolved_from_groups: List[str] = []
    if access_groups:
        resolved_from_groups = (
            await MCPRequestHandler._get_mcp_servers_from_access_groups(access_groups)
        )

    all_servers: Set[str] = set(direct_servers + resolved_from_groups + tool_permission_keys)

    return {
        "mcp_servers": list(all_servers),
        "mcp_access_groups": access_groups,
    }


async def validate_key_mcp_servers_against_team(
    object_permission: Optional[LiteLLM_ObjectPermissionBase],
    team_obj: Optional[LiteLLM_TeamTableCachedObj],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """
    Validates that the requested MCP servers/access groups/tool permissions
    on a key are allowed by the key's team.

    - Admin bypass: skips validation for proxy admins.
    - Deny-by-default: if team has no MCP config, only allow_all_keys servers pass.
    - Server validation: requested mcp_servers must be subset of team's expanded set + allow_all_keys.
    - Access group validation: resolve requested mcp_access_groups to server IDs,
      check those are subset of team's expanded server set.
    - Tool permission validation: server IDs in mcp_tool_permissions keys must be in team's allowed set.

    Raises HTTPException(403) on failure.
    """
    if object_permission is None:
        return

    # Admin bypass
    if _user_has_admin_view(user_api_key_dict):
        return

    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    allow_all_keys_ids: Set[str] = set(
        global_mcp_server_manager.get_allow_all_keys_server_ids()
    )

    requested_servers: List[str] = object_permission.mcp_servers or []
    requested_access_groups: List[str] = object_permission.mcp_access_groups or []
    requested_tool_perm_keys: List[str] = list(
        (object_permission.mcp_tool_permissions or {}).keys()
    )

    # Nothing requested - nothing to validate
    if not requested_servers and not requested_access_groups and not requested_tool_perm_keys:
        return

    # Build the team's allowed server set
    team_mcp = await get_team_mcp_permissions(team_obj) if team_obj else None

    if team_mcp is None:
        # Team has no MCP config - deny-by-default: only allow_all_keys servers pass
        disallowed = (
            set(requested_servers) | set(requested_tool_perm_keys)
        ) - allow_all_keys_ids
        if disallowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Team has no MCP configuration. The following MCP servers are not allowed: {sorted(disallowed)}"
                },
            )
        # Also resolve requested access groups to server IDs and check those
        if requested_access_groups:
            resolved = await MCPRequestHandler._get_mcp_servers_from_access_groups(
                requested_access_groups
            )
            disallowed_from_groups = set(resolved) - allow_all_keys_ids
            if disallowed_from_groups:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": f"Team has no MCP configuration. Access groups resolve to unauthorized servers: {sorted(disallowed_from_groups)}"
                    },
                )
        return

    team_allowed_servers: Set[str] = set(team_mcp["mcp_servers"]) | allow_all_keys_ids

    # Validate direct server IDs
    disallowed_servers = set(requested_servers) - team_allowed_servers
    if disallowed_servers:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Key requests MCP servers not allowed by team: {sorted(disallowed_servers)}"
            },
        )

    # Validate access groups: resolve to server IDs, then check subset
    if requested_access_groups:
        resolved_servers = await MCPRequestHandler._get_mcp_servers_from_access_groups(
            requested_access_groups
        )
        disallowed_from_groups = set(resolved_servers) - team_allowed_servers
        if disallowed_from_groups:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Key's MCP access groups resolve to servers not allowed by team: {sorted(disallowed_from_groups)}"
                },
            )

    # Validate tool permission keys
    disallowed_tool_keys = set(requested_tool_perm_keys) - team_allowed_servers
    if disallowed_tool_keys:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"Key's mcp_tool_permissions reference servers not allowed by team: {sorted(disallowed_tool_keys)}"
            },
        )


async def get_allowed_mcp_access_groups_for_user(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> Optional[Set[str]]:
    """
    Returns the set of MCP access groups the user can see (via their teams + key).
    Returns None for admins (all groups allowed).
    """
    if _user_has_admin_view(user_api_key_dict):
        return None

    from litellm.proxy._experimental.mcp_server.ui_session_utils import (
        build_effective_auth_contexts,
    )
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    allowed_groups: Set[str] = set()
    auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

    for auth_context in auth_contexts:
        groups = await MCPRequestHandler.get_mcp_access_groups(auth_context)
        allowed_groups.update(groups)

    return allowed_groups