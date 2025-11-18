"""
Common utility functions for handling object permission updates across
organizations, teams, and keys.
"""

import json
from litellm._uuid import uuid
from typing import Dict, Optional, Union

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