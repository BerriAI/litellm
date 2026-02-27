"""
Common utility functions for handling object permission updates across
organizations, teams, and keys.
"""

import json
from litellm._uuid import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


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


async def get_allowed_mcp_access_groups_for_user(
    user_api_key_dict: "UserAPIKeyAuth",
    prisma_client: Optional[PrismaClient],
) -> Optional[set]:
    """
    Return the set of MCP access group IDs the user can assign (via teams or key).
    Returns None if user is admin (can assign any) or if MCP modules are unavailable.
    """
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

    if _user_has_admin_view(user_api_key_dict):
        return None  # Admin can assign any

    try:
        from litellm.proxy._experimental.mcp_server.ui_session_utils import (
            build_effective_auth_contexts,
        )
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import user_api_key_cache, proxy_logging_obj
    except ImportError:
        return set()  # No MCP - user has no access

    if prisma_client is None or user_api_key_cache is None:
        return set()

    allowed_access_group_ids: set = set()
    auth_contexts = await build_effective_auth_contexts(user_api_key_dict)

    for auth_context in auth_contexts:
        if auth_context.team_id:
            team_obj = await get_team_object(
                team_id=auth_context.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=getattr(
                    user_api_key_dict, "parent_otel_span", None
                ),
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj and team_obj.object_permission:
                groups = team_obj.object_permission.mcp_access_groups or []
                allowed_access_group_ids.update(groups)

    if user_api_key_dict.object_permission:
        key_groups = user_api_key_dict.object_permission.mcp_access_groups or []
        allowed_access_group_ids.update(key_groups)

    return allowed_access_group_ids


async def validate_mcp_object_permission_for_key(
    user_api_key_dict: "UserAPIKeyAuth",
    object_permission: Optional[Union[Dict[str, Any], Any]],
    prisma_client: Optional[PrismaClient],
) -> None:
    """
    Validate that a non-admin user can only assign MCP servers and access groups
    they have access to (via their teams or key). With view_all mode, users see
    all servers but must not be able to assign servers/groups they lack access to.

    Raises:
        HTTPException: 403 if user tries to assign MCP servers or access groups
            they do not have access to.
    """
    from fastapi import HTTPException, status

    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

    if object_permission is None:
        return

    # Admins can assign any MCP servers/groups
    if _user_has_admin_view(user_api_key_dict):
        return

    # Extract mcp_servers and mcp_access_groups from object_permission
    mcp_servers: list = []
    mcp_access_groups: list = []
    if isinstance(object_permission, dict):
        mcp_servers = object_permission.get("mcp_servers") or []
        mcp_access_groups = object_permission.get("mcp_access_groups") or []
    else:
        mcp_servers = getattr(object_permission, "mcp_servers", None) or []
        mcp_access_groups = getattr(object_permission, "mcp_access_groups", None) or []

    if not mcp_servers and not mcp_access_groups:
        return

    try:
        from litellm.proxy._experimental.mcp_server.ui_session_utils import (
            build_effective_auth_contexts,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.proxy_server import user_api_key_cache
    except ImportError:
        verbose_proxy_logger.warning(
            "MCP modules not available, cannot validate MCP object permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "MCP object permission validation is unavailable. "
                    "Cannot assign mcp_servers or mcp_access_groups to this key."
                )
            },
        )

    if prisma_client is None or user_api_key_cache is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    "MCP object permission validation is unavailable (missing dependencies). "
                    "Cannot assign mcp_servers or mcp_access_groups to this key."
                )
            },
        )

    allowed_server_ids: set = set()
    auth_contexts = await build_effective_auth_contexts(user_api_key_dict)
    for auth_context in auth_contexts:
        server_ids = await global_mcp_server_manager.get_allowed_mcp_servers(
            auth_context
        )
        allowed_server_ids.update(server_ids)

    allowed_access_group_ids = await get_allowed_mcp_access_groups_for_user(
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )
    if allowed_access_group_ids is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Unable to determine allowed MCP access groups."},
        )

    # Validate requested mcp_servers
    disallowed_servers = [
        s for s in mcp_servers if s not in allowed_server_ids
    ]
    if disallowed_servers:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    f"You do not have access to assign the following MCP servers to this key: {disallowed_servers}. "
                    "You can only assign MCP servers that your teams have access to."
                )
            },
        )

    # Validate requested mcp_access_groups
    disallowed_groups = [
        g for g in mcp_access_groups if g not in allowed_access_group_ids
    ]
    if disallowed_groups:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": (
                    f"You do not have access to assign the following MCP access groups to this key: {disallowed_groups}. "
                    "You can only assign MCP access groups that your teams have access to."
                )
            },
        )