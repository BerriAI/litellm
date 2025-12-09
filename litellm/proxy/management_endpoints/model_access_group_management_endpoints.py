"""
Allow proxy admin to manage model access groups

Endpoints here:
- POST /model_group/new - Create a new access group with multiple model names
"""

import json
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# Clear cache and reload models to pick up the access group changes
from litellm.proxy.management_endpoints.model_management_endpoints import (
    clear_cache,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    AccessGroupInfo,
    DeleteModelGroupResponse,
    ListAccessGroupsResponse,
    NewModelGroupRequest,
    NewModelGroupResponse,
    UpdateModelGroupRequest,
)

router = APIRouter()


def validate_models_exist(
    model_names: List[str], llm_router
) -> Tuple[bool, List[str]]:
    """
    Validate that all requested model names exist in the router.
    Checks only exact model name matches.
    
    Returns:
        Tuple[bool, List[str]]: (all_valid, missing_models)
    """
    if llm_router is None:
        return False, model_names
    
    router_model_names = set(llm_router.get_model_names())
    missing = [m for m in model_names if m not in router_model_names]
    return (len(missing) == 0, missing)


def add_access_group_to_deployment(
    model_info: Dict[str, Any], access_group: str
) -> Tuple[Dict[str, Any], bool]:
    """
    Add an access group to a deployment's model_info.
    
    Args:
        model_info: The model_info dictionary from the deployment
        access_group: The access group name to add
        
    Returns:
        Tuple[Dict[str, Any], bool]: (updated_model_info, was_modified)
    """
    access_groups = model_info.get("access_groups", [])
    
    # Check if access group already exists
    if access_group in access_groups:
        return model_info, False
    
    # Add the access group
    access_groups.append(access_group)
    model_info["access_groups"] = access_groups
    
    return model_info, True


async def update_deployments_with_access_group(
    model_names: List[str],
    access_group: str,
    prisma_client: PrismaClient,
) -> int:
    """
    Update all deployments for the given model names to include the access group.
    
    Args:
        model_names: List of model names whose deployments should be updated
        access_group: The access group name to add
        prisma_client: Database client
        
    Returns:
        int: Number of deployments updated
    """
    models_updated = 0
    
    for model_name in model_names:
        verbose_proxy_logger.debug(
            f"Updating deployments for model_name: {model_name}"
        )
        
        # Get all deployments with this model_name
        deployments = await prisma_client.db.litellm_proxymodeltable.find_many(
            where={"model_name": model_name}
        )
        
        verbose_proxy_logger.debug(
            f"Found {len(deployments)} deployments for model_name: {model_name}"
        )
        
        # If no deployments found, this is a config model (not in DB)
        if len(deployments) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Can't find model '{model_name}' in Database. Access group management is only supported for database models."
                },
            )
        
        # Update each deployment
        for deployment in deployments:
            model_info = deployment.model_info or {}
            
            # Add access group using helper
            updated_model_info, was_modified = add_access_group_to_deployment(
                model_info=model_info,
                access_group=access_group,
            )
            
            # Only update in DB if modified
            if was_modified:
                await prisma_client.db.litellm_proxymodeltable.update(
                    where={"model_id": deployment.model_id},
                    data={"model_info": json.dumps(updated_model_info)},
                )
                
                models_updated += 1
                verbose_proxy_logger.debug(
                    f"Updated deployment {deployment.model_id} with access group: {access_group}"
                )
    
    return models_updated


def remove_access_group_from_deployment(
    model_info: Dict[str, Any], access_group: str
) -> Tuple[Dict[str, Any], bool]:
    """
    Remove an access group from a deployment's model_info.
    
    Args:
        model_info: The model_info dictionary from the deployment
        access_group: The access group name to remove
        
    Returns:
        Tuple[Dict[str, Any], bool]: (updated_model_info, was_modified)
    """
    access_groups = model_info.get("access_groups", [])
    
    # Check if access group exists
    if access_group not in access_groups:
        return model_info, False
    
    # Remove the access group
    access_groups.remove(access_group)
    model_info["access_groups"] = access_groups
    
    return model_info, True


async def get_all_access_groups_from_db(
    prisma_client: PrismaClient,
) -> Dict[str, AccessGroupInfo]:
    """
    Get all access groups from the database.
    
    Returns:
        Dict[str, AccessGroupInfo]: Dictionary mapping access_group name to info
    """
    # Get all deployments
    deployments = await prisma_client.db.litellm_proxymodeltable.find_many()
    
    # Build access group map
    access_group_map: Dict[str, Dict[str, Any]] = {}
    
    for deployment in deployments:
        model_info = deployment.model_info or {}
        access_groups = model_info.get("access_groups", [])
        model_name = deployment.model_name
        
        for access_group in access_groups:
            if access_group not in access_group_map:
                access_group_map[access_group] = {
                    "model_names": set(),
                    "deployment_count": 0,
                }
            
            access_group_map[access_group]["model_names"].add(model_name)
            access_group_map[access_group]["deployment_count"] += 1
    
    # Convert to AccessGroupInfo objects
    result = {}
    for access_group, data in access_group_map.items():
        result[access_group] = AccessGroupInfo(
            access_group=access_group,
            model_names=sorted(list(data["model_names"])),
            deployment_count=data["deployment_count"],
        )
    
    return result


@router.post(
    "/access_group/new",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewModelGroupResponse,
)
async def create_model_group(
    data: NewModelGroupRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new access group containing multiple model names.
    
    An access group is a named collection of model groups that can be referenced
    by teams/keys for simplified access control.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:4000/access_group/new' \\
      -H 'Authorization: Bearer sk-1234' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "access_group": "production-models",
        "model_names": ["gpt-4", "claude-3-opus", "gemini-pro"]
      }'
    ```
    
    Parameters:
    - access_group: str - The access group name (e.g., "production-models")
    - model_names: List[str] - List of existing model groups to include
    
    Returns:
    - NewModelGroupResponse with the created access group details
    
    Raises:
    - HTTPException 400: If any model names don't exist
    - HTTPException 500: If database operations fail
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
    )
    
    verbose_proxy_logger.debug(
        f"Creating access group: {data.access_group} with models: {data.model_names}"
    )
    
    # Validation: Check if access_group is provided
    if not data.access_group or not data.access_group.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "access_group is required and cannot be empty"},
        )
    
    # Validation: Check if model_names list is provided and not empty
    if not data.model_names or len(data.model_names) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "model_names list is required and cannot be empty"},
        )
    
    # Validation: Check if all models exist in the router
    all_valid, missing_models = validate_models_exist(
        model_names=data.model_names,
        llm_router=llm_router,
    )
    
    if not all_valid:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Model(s) not found: {', '.join(missing_models)}"},
        )
    
    # Check if database is connected
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected. Cannot create access group."},
        )
    
    try:
        # Check if access group already exists
        existing_access_groups = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )
        
        if data.access_group in existing_access_groups:
            raise HTTPException(
                status_code=409,
                detail={"error": f"Access group '{data.access_group}' already exists. Use PUT /access_group/{data.access_group}/update to modify it."},
            )
        
        # Update deployments using helper function
        models_updated = await update_deployments_with_access_group(
            model_names=data.model_names,
            access_group=data.access_group,
            prisma_client=prisma_client,
        )
        
        await clear_cache()
        
        verbose_proxy_logger.info(
            f"Successfully created access group '{data.access_group}' with {models_updated} models updated"
        )
        
        return NewModelGroupResponse(
            access_group=data.access_group,
            model_names=data.model_names,
            models_updated=models_updated,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error creating access group '{data.access_group}': {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to create access group: {str(e)}"},
        )


@router.get(
    "/access_group/list",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListAccessGroupsResponse,
)
async def list_access_groups(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all access groups.
    
    Returns a list of all access groups with their model names and deployment counts.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:4000/access_group/list' \\
      -H 'Authorization: Bearer sk-1234'
    ```
    
    Returns:
    - ListAccessGroupsResponse with all access groups
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected."},
        )
    
    try:
        access_groups_map = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )
        
        # Sort by access group name
        access_groups_list = sorted(
            access_groups_map.values(),
            key=lambda x: x.access_group,
        )
        
        return ListAccessGroupsResponse(access_groups=access_groups_list)
        
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing access groups: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to list access groups: {str(e)}"},
        )


@router.get(
    "/access_group/{access_group}/info",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AccessGroupInfo,
)
async def get_access_group_info(
    access_group: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get information about a specific access group.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:4000/access_group/production-models/info' \\
      -H 'Authorization: Bearer sk-1234'
    ```
    
    Parameters:
    - access_group: str - The access group name (URL path parameter)
    
    Returns:
    - AccessGroupInfo with the access group details
    
    Raises:
    - HTTPException 404: If access group not found
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected."},
        )
    
    try:
        access_groups_map = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )
        
        if access_group not in access_groups_map:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Access group '{access_group}' not found"},
            )
        
        return access_groups_map[access_group]
        
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error getting access group info for '{access_group}': {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get access group info: {str(e)}"},
        )


@router.put(
    "/access_group/{access_group}/update",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewModelGroupResponse,
)
async def update_access_group(
    access_group: str,
    data: UpdateModelGroupRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an access group's model names.
    
    This will:
    1. Remove the access group from all current deployments
    2. Add the access group to all deployments for the new model_names list
    
    Example:
    ```bash
    curl -X PUT 'http://localhost:4000/access_group/production-models/update' \\
      -H 'Authorization: Bearer sk-1234' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "model_names": ["gpt-4", "claude-3-sonnet"]
      }'
    ```
    
    Parameters:
    - access_group: str - The access group name (URL path parameter)
    - model_names: List[str] - New list of model groups to include
    
    Returns:
    - NewModelGroupResponse with the updated access group details
    
    Raises:
    - HTTPException 400: If any model names don't exist
    - HTTPException 404: If access group not found
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected."},
        )
    
    verbose_proxy_logger.debug(
        f"Updating access group: {access_group} with models: {data.model_names}"
    )
    
    # Validation: Check if model_names list is provided and not empty
    if not data.model_names or len(data.model_names) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "model_names list is required and cannot be empty"},
        )
    
    # Validation: Check if access group exists
    try:
        access_groups_map = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )
        if access_group not in access_groups_map:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Access group '{access_group}' not found"},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to check access group existence: {str(e)}"},
        )
    
    # Validation: Check if all new models exist
    all_valid, missing_models = validate_models_exist(
        model_names=data.model_names,
        llm_router=llm_router,
    )
    
    if not all_valid:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Model(s) not found: {', '.join(missing_models)}"},
        )
    
    try:
        # Step 1: Remove access group from ALL DB deployments (skip config models)
        all_deployments = await prisma_client.db.litellm_proxymodeltable.find_many()
        
        for deployment in all_deployments:
            model_info = deployment.model_info or {}

            
            updated_model_info, was_modified = remove_access_group_from_deployment(
                model_info=model_info,
                access_group=access_group,
            )
            
            if was_modified:
                await prisma_client.db.litellm_proxymodeltable.update(
                    where={"model_id": deployment.model_id},
                    data={"model_info": json.dumps(updated_model_info)},
                )
        
        # Step 2: Add access group to new model_names
        models_updated = await update_deployments_with_access_group(
            model_names=data.model_names,
            access_group=access_group,
            prisma_client=prisma_client,
        )
        
        # Clear cache and reload models to pick up the access group changes
        await clear_cache()
        
        verbose_proxy_logger.info(
            f"Successfully updated access group '{access_group}' with {models_updated} models updated"
        )
        
        return NewModelGroupResponse(
            access_group=access_group,
            model_names=data.model_names,
            models_updated=models_updated,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error updating access group '{access_group}': {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update access group: {str(e)}"},
        )


@router.delete(
    "/access_group/{access_group}/delete",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=DeleteModelGroupResponse,
)
async def delete_access_group(
    access_group: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an access group.
    
    Removes the access group from all deployments that have it.
    
    Example:
    ```bash
    curl -X DELETE 'http://localhost:4000/access_group/production-models/delete' \\
      -H 'Authorization: Bearer sk-1234'
    ```
    
    Parameters:
    - access_group: str - The access group name (URL path parameter)
    
    Returns:
    - DeleteModelGroupResponse with deletion details
    
    Raises:
    - HTTPException 404: If access group not found
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected."},
        )
    
    verbose_proxy_logger.debug(f"Deleting access group: {access_group}")
    
    # Validation: Check if access group exists
    try:
        access_groups_map = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )
        if access_group not in access_groups_map:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Access group '{access_group}' not found"},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to check access group existence: {str(e)}"},
        )
    
    try:
        # Remove access group from all DB deployments (skip config models)
        all_deployments = await prisma_client.db.litellm_proxymodeltable.find_many()
        models_updated = 0
        
        for deployment in all_deployments:
            model_info = deployment.model_info or {}
            
            updated_model_info, was_modified = remove_access_group_from_deployment(
                model_info=model_info,
                access_group=access_group,
            )
            
            if was_modified:
                await prisma_client.db.litellm_proxymodeltable.update(
                    where={"model_id": deployment.model_id},
                    data={"model_info": json.dumps(updated_model_info)},
                )
                models_updated += 1
        
        # Clear cache and reload models to pick up the access group changes
        await clear_cache()
        
        verbose_proxy_logger.info(
            f"Successfully deleted access group '{access_group}' from {models_updated} deployments"
        )
        
        return DeleteModelGroupResponse(
            access_group=access_group,
            models_updated=models_updated,
            message=f"Access group '{access_group}' deleted successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error deleting access group '{access_group}': {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to delete access group: {str(e)}"},
        )

