"""
Allow proxy admin to manage model access groups

Endpoints here:
- POST /model_group/new - Create a new access group with multiple model names
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# Clear cache and reload models to pick up the access group changes
from litellm.proxy.management_endpoints.model_management_endpoints import (
    clear_cache,
    live_model_ids_snapshot,
    model_info_as_mapping,
    reload_serving_verdict,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.model_repository import ModelRepository
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    AccessGroupInfo,
    DeleteModelGroupResponse,
    ListAccessGroupsResponse,
    NewModelGroupRequest,
    NewModelGroupResponse,
    UpdateModelGroupRequest,
)

router: Final = APIRouter()


def validate_models_exist(model_names: list[str], llm_router) -> tuple[bool, list[str]]:
    """
    Validate that all requested model names exist in the router.
    Checks only exact model name matches.

    Returns:
        Tuple[bool, List[str]]: (all_valid, missing_models)
    """
    if llm_router is None:
        return False, model_names

    router_model_names: Final = set(llm_router.get_model_names())
    missing: Final = [m for m in model_names if m not in router_model_names]
    return (len(missing) == 0, missing)


def add_access_group_to_deployment(model_info: dict[str, Any], access_group: str) -> tuple[dict[str, Any], bool]:
    """
    Add an access group to a deployment's model_info.

    Args:
        model_info: The model_info dictionary from the deployment
        access_group: The access group name to add

    Returns:
        Tuple[Dict[str, Any], bool]: (updated_model_info, was_modified)
    """
    access_groups: Final = model_info.get("access_groups", [])

    # Check if access group already exists
    if access_group in access_groups:
        return model_info, False

    # Add the access group
    access_groups.append(access_group)
    model_info["access_groups"] = access_groups

    return model_info, True


def _raise_http_if_reload_degraded_serving(
    before: frozenset[str],
    written_models: Sequence[tuple[str, object]],
    access_group: str,
) -> None:
    """Same verdict as the model-write endpoints, expressed through this file's
    HTTPException error convention, with the metadata-only obligation: these writes
    change group membership, not the models themselves, so a row that was already not
    serving before the reload is never blamed here; only a model this reload stopped
    serving is reported."""
    missing, collateral = reload_serving_verdict(before=before, written_models=written_models, written_must_serve=False)
    gone: Final = tuple(dict.fromkeys((*missing, *collateral)))
    if not gone:
        return
    raise HTTPException(
        status_code=500,
        detail={
            "error": (
                f"Access group '{access_group}' was saved to the database, but model id(s) {list(gone)} that "
                "this pod was serving are no longer live after the reload it triggered. Other pods reload on "
                "their own interval. Check server logs for 'Error upserting deployment' for the cause."
            )
        },
    )


async def _tag_deployment_with_access_group(
    model_id: str,
    model_info: object,
    access_group: str,
    prisma_client: PrismaClient,
) -> tuple[str, Mapping[str, object]] | None:
    """Write `access_group` into one deployment's model_info; returns the
    (model_id, updated model_info) pair when a write happened, None when the
    deployment already carried the group."""
    updated_model_info, was_modified = add_access_group_to_deployment(
        model_info=dict(_readable_model_info_or_raise(model_id=model_id, model_info=model_info)),
        access_group=access_group,
    )
    if not was_modified:
        return None
    await ModelRepository(prisma_client).table.update(
        where={"model_id": model_id},
        data={"model_info": json.dumps(updated_model_info)},
    )
    verbose_proxy_logger.debug("Updated deployment %s with access group: %s", model_id, access_group)
    return (model_id, updated_model_info)


def _readable_model_info_or_raise(model_id: str, model_info: object) -> Mapping[str, object]:
    """These helpers rewrite the model_info column wholesale, so a present-but-unreadable
    value must refuse loudly rather than be silently replaced with a fresh object; an
    absent value stays a legitimate empty start."""
    parsed: Final = model_info_as_mapping(model_info)
    if parsed is None and model_info is not None:
        raise ValueError(f"model_info for deployment {model_id} is not a readable JSON object; refusing to rewrite it")
    return parsed or {}


async def _strip_access_group_from_deployment(
    model_id: str,
    model_info: object,
    access_group: str,
    prisma_client: PrismaClient,
) -> tuple[str, Mapping[str, object]] | None:
    """Remove `access_group` from one deployment's model_info; returns the
    (model_id, updated model_info) pair when a write happened, None when the
    deployment did not carry the group."""
    updated_model_info, was_modified = remove_access_group_from_deployment(
        model_info=dict(_readable_model_info_or_raise(model_id=model_id, model_info=model_info)),
        access_group=access_group,
    )
    if not was_modified:
        return None
    await ModelRepository(prisma_client).table.update(
        where={"model_id": model_id},
        data={"model_info": json.dumps(updated_model_info)},
    )
    return (model_id, updated_model_info)


async def update_deployments_with_access_group(
    model_names: list[str],
    access_group: str,
    prisma_client: PrismaClient,
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    """
    Update all deployments for the given model names to include the access group.

    Args:
        model_names: List of model names whose deployments should be updated
        access_group: The access group name to add
        prisma_client: Database client

    Returns:
        The (model_id, updated model_info) pair of every deployment actually written,
        so callers can verify each one survived the post-write reload
    """
    deployments: Final = await ModelRepository(prisma_client).table.find_many(where={"model_name": {"in": model_names}})
    verbose_proxy_logger.debug("Found %s deployments for model_names: %s", len(deployments), model_names)

    found_names: Final = {deployment.model_name for deployment in deployments}
    for model_name in model_names:
        if model_name not in found_names:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Can't find model '{model_name}' in Database. Access group management is only supported for database models."
                },
            )

    tagged: Final = [
        await _tag_deployment_with_access_group(
            model_id=deployment.model_id,
            model_info=deployment.model_info,
            access_group=access_group,
            prisma_client=prisma_client,
        )
        for deployment in deployments
    ]
    return tuple(pair for pair in tagged if pair is not None)


async def update_specific_deployments_with_access_group(
    model_ids: list[str],
    access_group: str,
    prisma_client: PrismaClient,
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    """
    Update specific deployments (by model_id) to include the access group.

    Unlike update_deployments_with_access_group which tags ALL deployments sharing
    a model_name, this function only tags the specific deployments identified by
    their unique model_id. Returns the (model_id, updated model_info) pair of every
    deployment actually written.
    """
    verbose_proxy_logger.debug("Updating specific deployment model_ids: %s", model_ids)
    tagged: Final = [
        await _tag_deployment_with_access_group(
            model_id=model_id,
            model_info=(await _find_deployment_or_400(model_id=model_id, prisma_client=prisma_client)),
            access_group=access_group,
            prisma_client=prisma_client,
        )
        for model_id in model_ids
    ]
    return tuple(pair for pair in tagged if pair is not None)


async def _find_deployment_or_400(model_id: str, prisma_client: PrismaClient) -> Mapping[str, object] | None:
    deployment: Final = await ModelRepository(prisma_client).table.find_unique(where={"model_id": model_id})
    if deployment is None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Deployment with model_id '{model_id}' not found in Database."},
        )
    return deployment.model_info


def remove_access_group_from_deployment(model_info: dict[str, Any], access_group: str) -> tuple[dict[str, Any], bool]:
    """
    Remove an access group from a deployment's model_info.

    Args:
        model_info: The model_info dictionary from the deployment
        access_group: The access group name to remove

    Returns:
        Tuple[Dict[str, Any], bool]: (updated_model_info, was_modified)
    """
    access_groups: Final = model_info.get("access_groups", [])

    # Check if access group exists
    if access_group not in access_groups:
        return model_info, False

    # Remove the access group
    access_groups.remove(access_group)
    model_info["access_groups"] = access_groups

    return model_info, True


async def get_all_access_groups_from_db(
    prisma_client: PrismaClient,
) -> dict[str, AccessGroupInfo]:
    """
    Get all access groups from the database.

    Returns:
        Dict[str, AccessGroupInfo]: Dictionary mapping access_group name to info
    """
    # Get all deployments
    deployments: Final = await ModelRepository(prisma_client).table.find_many()

    # Build access group map
    access_group_map: Final[dict[str, dict[str, Any]]] = {}

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
    result: Final = {}
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

    verbose_proxy_logger.debug("Creating access group: %s with models: %s", data.access_group, data.model_names)

    # Validation: Check if access_group is provided
    if not data.access_group or not data.access_group.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "access_group is required and cannot be empty"},
        )

    # Validation: Check that at least one of model_names or model_ids is provided
    has_model_names: Final = data.model_names and len(data.model_names) > 0
    has_model_ids: Final = data.model_ids and len(data.model_ids) > 0

    if not has_model_names and not has_model_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": "Either model_names or model_ids must be provided and non-empty"},
        )

    # If model_ids is provided, use it (more precise targeting)
    use_model_ids: Final = has_model_ids

    # Validate model_names exist in router (only if using model_names path)
    if not use_model_ids and has_model_names:
        assert data.model_names is not None
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
        existing_access_groups: Final = await get_all_access_groups_from_db(prisma_client=prisma_client)

        if data.access_group in existing_access_groups:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": f"Access group '{data.access_group}' already exists. Use PUT /access_group/{data.access_group}/update to modify it."
                },
            )

        # Update deployments using the appropriate method
        if use_model_ids:
            assert data.model_ids is not None
            updated_pairs = await update_specific_deployments_with_access_group(
                model_ids=data.model_ids,
                access_group=data.access_group,
                prisma_client=prisma_client,
            )
        else:
            assert data.model_names is not None
            updated_pairs = await update_deployments_with_access_group(
                model_names=data.model_names,
                access_group=data.access_group,
                prisma_client=prisma_client,
            )
        models_updated: Final = len(updated_pairs)

        live_before_reload: Final = live_model_ids_snapshot()

        await clear_cache()
        _raise_http_if_reload_degraded_serving(
            before=live_before_reload,
            written_models=updated_pairs,
            access_group=data.access_group,
        )

        verbose_proxy_logger.info(
            "Successfully created access group '%s' with %s models updated", data.access_group, models_updated
        )

        return NewModelGroupResponse(
            access_group=data.access_group,
            model_names=data.model_names,
            model_ids=data.model_ids,
            models_updated=models_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error creating access group '%s': %s", data.access_group, e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to create access group: {e}"},
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
        access_groups_map: Final = await get_all_access_groups_from_db(prisma_client=prisma_client)

        # Sort by access group name
        access_groups_list: Final = sorted(
            access_groups_map.values(),
            key=lambda x: x.access_group,
        )

        return ListAccessGroupsResponse(access_groups=access_groups_list)

    except Exception as e:
        verbose_proxy_logger.exception("Error listing access groups: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to list access groups: {e}"},
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
        access_groups_map: Final = await get_all_access_groups_from_db(prisma_client=prisma_client)

        if access_group not in access_groups_map:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Access group '{access_group}' not found"},
            )

        return access_groups_map[access_group]

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting access group info for '%s': %s", access_group, e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get access group info: {e}"},
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

    verbose_proxy_logger.debug("Updating access group: %s with models: %s", access_group, data.model_names)

    # Validation: Check that at least one of model_names or model_ids is provided
    has_model_names: Final = data.model_names and len(data.model_names) > 0
    has_model_ids: Final = data.model_ids and len(data.model_ids) > 0

    if not has_model_names and not has_model_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": "Either model_names or model_ids must be provided and non-empty"},
        )

    use_model_ids: Final = has_model_ids

    # Validation: Check if access group exists
    try:
        access_groups_map: Final = await get_all_access_groups_from_db(prisma_client=prisma_client)
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
            detail={"error": f"Failed to check access group existence: {e}"},
        )

    # Validation: Check if all new models exist (only if using model_names path)
    if not use_model_ids and has_model_names:
        assert data.model_names is not None
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
        all_deployments: Final = await ModelRepository(prisma_client).table.find_many()

        stripped: Final = [
            await _strip_access_group_from_deployment(
                model_id=deployment.model_id,
                model_info=deployment.model_info,
                access_group=access_group,
                prisma_client=prisma_client,
            )
            for deployment in all_deployments
        ]
        stripped_pairs: Final = tuple(pair for pair in stripped if pair is not None)

        # Step 2: Add access group using the appropriate method
        if use_model_ids:
            assert data.model_ids is not None
            updated_pairs = await update_specific_deployments_with_access_group(
                model_ids=data.model_ids,
                access_group=access_group,
                prisma_client=prisma_client,
            )
        else:
            assert data.model_names is not None
            updated_pairs = await update_deployments_with_access_group(
                model_names=data.model_names,
                access_group=access_group,
                prisma_client=prisma_client,
            )
        models_updated: Final = len(updated_pairs)

        # Clear cache and reload models to pick up the access group changes
        live_before_reload: Final = live_model_ids_snapshot()
        await clear_cache()
        _raise_http_if_reload_degraded_serving(
            before=live_before_reload,
            written_models=list({**dict(stripped_pairs), **dict(updated_pairs)}.items()),
            access_group=access_group,
        )

        verbose_proxy_logger.info(
            "Successfully updated access group '%s' with %s models updated", access_group, models_updated
        )

        return NewModelGroupResponse(
            access_group=access_group,
            model_names=data.model_names,
            model_ids=data.model_ids,
            models_updated=models_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating access group '%s': %s", access_group, e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update access group: {e}"},
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

    verbose_proxy_logger.debug("Deleting access group: %s", access_group)

    # Validation: Check if access group exists
    try:
        access_groups_map: Final = await get_all_access_groups_from_db(prisma_client=prisma_client)
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
            detail={"error": f"Failed to check access group existence: {e}"},
        )

    try:
        # Remove access group from all DB deployments (skip config models)
        all_deployments: Final = await ModelRepository(prisma_client).table.find_many()

        removed: Final = [
            await _strip_access_group_from_deployment(
                model_id=deployment.model_id,
                model_info=deployment.model_info,
                access_group=access_group,
                prisma_client=prisma_client,
            )
            for deployment in all_deployments
        ]
        removed_pairs: Final = tuple(pair for pair in removed if pair is not None)
        models_updated: Final = len(removed_pairs)

        # Clear cache and reload models to pick up the access group changes
        live_before_reload: Final = live_model_ids_snapshot()
        await clear_cache()
        _raise_http_if_reload_degraded_serving(
            before=live_before_reload,
            written_models=removed_pairs,
            access_group=access_group,
        )

        verbose_proxy_logger.info(
            "Successfully deleted access group '%s' from %s deployments", access_group, models_updated
        )

        return DeleteModelGroupResponse(
            access_group=access_group,
            models_updated=models_updated,
            message=f"Access group '{access_group}' deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting access group '%s': %s", access_group, e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to delete access group: {e}"},
        )
