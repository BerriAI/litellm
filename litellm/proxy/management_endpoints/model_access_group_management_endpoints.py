"""
Allow proxy admin to manage model access groups

Endpoints here:
- POST /model_group/new - Create a new access group with multiple model names
"""

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    NewModelGroupRequest,
    NewModelGroupResponse,
)

router = APIRouter()


def validate_models_exist(
    model_names: List[str], llm_router
) -> Tuple[bool, List[str]]:
    """
    Validate that all requested model names exist in the router.
    
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
                    data={"model_info": prisma_client.jsonify_object(updated_model_info)},
                )
                
                models_updated += 1
                verbose_proxy_logger.debug(
                    f"Updated deployment {deployment.model_id} with access group: {access_group}"
                )
    
    return models_updated


@router.post(
    "/model_group/new",
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
    curl -X POST 'http://localhost:4000/model_group/new' \\
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
        proxy_config,
        proxy_logging_obj,
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
        # Update deployments using helper function
        models_updated = await update_deployments_with_access_group(
            model_names=data.model_names,
            access_group=data.access_group,
            prisma_client=prisma_client,
        )
        
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

