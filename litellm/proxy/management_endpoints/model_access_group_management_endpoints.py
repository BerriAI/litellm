"""
Allow proxy admin to manage model access groups

Endpoints here:
- POST /model_group/new - Create a new access group with multiple model names
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_checks import resolve_nested_groups
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


# ---------------------------------------------------------------------------
# Per-process membership-map cache
# ---------------------------------------------------------------------------
# get_group_memberships_from_db is called on every /v1/models and /model/info
# request via get_available_models_for_user. Without a cache that's one extra
# Prisma roundtrip per request - bad under burst traffic. We cache the map
# in process memory for a short TTL and invalidate explicitly on writes
# (upsert/delete) so the consistency window inside the writing process is
# zero. Across processes, eventual consistency is bounded by the TTL -
# matches today's behavior for llm_router.get_model_access_groups() which is
# also per-process.

_MEMBERSHIPS_CACHE_TTL_SECONDS = 60.0
_MEMBERSHIPS_CACHE: Optional[Tuple[float, Dict[str, List[str]]]] = None


def invalidate_group_memberships_cache() -> None:
    """Drop the in-process membership cache. Call after any write that
    mutates the LiteLLM_AccessGroupMembership table."""
    global _MEMBERSHIPS_CACHE
    _MEMBERSHIPS_CACHE = None


def validate_models_exist(
    model_names: List[str],
    llm_router,
    known_access_groups: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that all requested member names exist as either a router model name
    or a known access group name (for nested groups).

    Returns:
        Tuple[bool, List[str]]: (all_valid, missing_names)
    """
    known_groups = known_access_groups or set()

    if llm_router is None:
        # DB-only deployment: no in-memory router means we cannot validate
        # real model names, but known_access_groups is still authoritative
        # for nested-group composition. Anything not in known_groups is
        # reported as missing (fail-closed).
        missing = [m for m in model_names if m not in known_groups]
        return (len(missing) == 0, missing)

    router_model_names = set(llm_router.get_model_names())
    missing = [
        m for m in model_names if m not in router_model_names and m not in known_groups
    ]
    return (len(missing) == 0, missing)


def _classify_member_names(
    names: List[str],
    router_model_names: Set[str],
    known_access_groups: Set[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of member names into (real_models, child_groups, unknown).

    Names registered both as a router model and as a group are classified
    as a model - matches the existing precedence in `_get_models_from_access_groups`
    where direct model_access_groups expansion runs before any other lookup.
    """
    real_models: List[str] = []
    child_groups: List[str] = []
    unknown: List[str] = []
    for name in names:
        if name in router_model_names:
            real_models.append(name)
        elif name in known_access_groups:
            child_groups.append(name)
        else:
            unknown.append(name)
    return real_models, child_groups, unknown


async def get_group_memberships_from_db(
    prisma_client: PrismaClient,
) -> Dict[str, List[str]]:
    """
    Build parent_group -> [child_groups] map from the membership table.
    Single query, in-memory bucketing - no N+1.

    Resilient by design: any failure to read the membership table (missing
    Prisma model, migration race, transient DB/network error, query timeout)
    degrades to an empty map. The auth path then falls back to today's
    flat-group semantics instead of 500-ing the whole request. We log at
    debug so ops can correlate fallback periods with incidents without
    drowning normal traffic in warnings.
    """
    try:
        rows = await prisma_client.db.litellm_accessgroupmembership.find_many()
    except Exception as e:  # noqa: BLE001 - intentional broad catch on auth path
        verbose_proxy_logger.debug(
            "litellm_accessgroupmembership read failed - "
            "skipping nested group resolution: %s",
            e,
        )
        return {}

    memberships: Dict[str, List[str]] = {}
    for row in rows:
        memberships.setdefault(row.parent_group, []).append(row.child_group)
    return memberships


async def get_cached_group_memberships(
    prisma_client: PrismaClient,
) -> Dict[str, List[str]]:
    """
    TTL-cached wrapper around get_group_memberships_from_db. Hot-path
    callers (model-listing endpoints) should use this; tests and write
    paths that need fresh data can call the underlying helper directly.
    """
    global _MEMBERSHIPS_CACHE
    now = time.monotonic()
    if _MEMBERSHIPS_CACHE is not None:
        cached_at, value = _MEMBERSHIPS_CACHE
        if now - cached_at < _MEMBERSHIPS_CACHE_TTL_SECONDS:
            return value
    fresh = await get_group_memberships_from_db(prisma_client=prisma_client)
    _MEMBERSHIPS_CACHE = (now, fresh)
    return fresh


async def upsert_group_memberships(
    parent_group: str,
    child_groups: List[str],
    prisma_client: PrismaClient,
) -> int:
    """
    Insert parent_group -> child_group edges into LiteLLM_AccessGroupMembership.
    Skips duplicates via the unique constraint. Rejects self-references eagerly.

    Returns:
        int: number of new edges inserted.
    """
    if not child_groups:
        return 0

    if parent_group in child_groups:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    f"Access group '{parent_group}' cannot include itself "
                    "as a member."
                )
            },
        )

    rows = [
        {"parent_group": parent_group, "child_group": child} for child in child_groups
    ]
    result = await prisma_client.db.litellm_accessgroupmembership.create_many(
        data=rows,
        skip_duplicates=True,
    )
    invalidate_group_memberships_cache()
    return result


async def delete_group_membership_edges(
    access_group: str,
    prisma_client: PrismaClient,
) -> int:
    """
    Delete every membership row where `access_group` appears as parent or child.
    Used during group deletion to avoid dangling references.

    Returns:
        int: number of edges deleted.
    """
    result = await prisma_client.db.litellm_accessgroupmembership.delete_many(
        where={
            "OR": [
                {"parent_group": access_group},
                {"child_group": access_group},
            ]
        }
    )
    invalidate_group_memberships_cache()
    return result


async def _clear_access_group_from_all_deployments(
    access_group: str,
    prisma_client: PrismaClient,
) -> int:
    """
    Remove the access_group tag from every deployment that carries it.
    Shared by update_access_group and delete_access_group.

    Returns:
        int: number of deployments touched.
    """
    deployments = await prisma_client.db.litellm_proxymodeltable.find_many()
    touched = 0
    for deployment in deployments:
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
            touched += 1
    return touched


async def _dual_write_group_membership(
    access_group: str,
    member_names: List[str],
    known_access_groups: Set[str],
    llm_router,
    prisma_client: PrismaClient,
) -> int:
    """
    Classify member_names into real models vs child groups and route each to
    the appropriate write path. Shared by create_model_group and
    update_access_group.

    Returns:
        int: total writes performed (deployment tags + membership edges).
    """
    router_model_names = (
        set(llm_router.get_model_names()) if llm_router is not None else set()
    )
    real_models, child_groups, _ = _classify_member_names(
        names=member_names,
        router_model_names=router_model_names,
        known_access_groups=known_access_groups,
    )
    writes = 0
    if real_models:
        writes += await update_deployments_with_access_group(
            model_names=real_models,
            access_group=access_group,
            prisma_client=prisma_client,
        )
    if child_groups:
        writes += await upsert_group_memberships(
            parent_group=access_group,
            child_groups=child_groups,
            prisma_client=prisma_client,
        )
    return writes


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
        verbose_proxy_logger.debug(f"Updating deployments for model_name: {model_name}")

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


async def update_specific_deployments_with_access_group(
    model_ids: List[str],
    access_group: str,
    prisma_client: PrismaClient,
) -> int:
    """
    Update specific deployments (by model_id) to include the access group.

    Unlike update_deployments_with_access_group which tags ALL deployments sharing
    a model_name, this function only tags the specific deployments identified by
    their unique model_id.
    """
    models_updated = 0
    for model_id in model_ids:
        verbose_proxy_logger.debug(f"Updating specific deployment model_id: {model_id}")
        deployment = await prisma_client.db.litellm_proxymodeltable.find_unique(
            where={"model_id": model_id}
        )
        if deployment is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Deployment with model_id '{model_id}' not found in Database."
                },
            )
        model_info = deployment.model_info or {}
        updated_model_info, was_modified = add_access_group_to_deployment(
            model_info=model_info,
            access_group=access_group,
        )
        if was_modified:
            await prisma_client.db.litellm_proxymodeltable.update(
                where={"model_id": model_id},
                data={"model_info": json.dumps(updated_model_info)},
            )
            models_updated += 1
            verbose_proxy_logger.debug(
                f"Updated deployment {model_id} with access group: {access_group}"
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
    Get all access groups from the database, including nested-group structure.

    Builds the direct group -> {models, count} map by scanning deployments,
    then layers in parent/child edges from LiteLLM_AccessGroupMembership.
    Pure-composition groups (those that exist only as a parent in the
    membership table, with no deployment tag) are surfaced too.

    `model_names` is expanded transitively via DFS - cyclic edges are skipped.
    `deployment_count` remains the direct-tag count, not transitive.

    Returns:
        Dict[str, AccessGroupInfo]: name -> info, including parent/child groups.
    """
    # Direct group membership from deployment tags
    deployments = await prisma_client.db.litellm_proxymodeltable.find_many()

    direct_models: Dict[str, Set[str]] = {}
    deployment_count: Dict[str, int] = {}

    for deployment in deployments:
        model_info = deployment.model_info or {}
        access_groups = model_info.get("access_groups", [])
        model_name = deployment.model_name

        for access_group in access_groups:
            direct_models.setdefault(access_group, set()).add(model_name)
            deployment_count[access_group] = deployment_count.get(access_group, 0) + 1

    # Group-to-group edges
    group_memberships = await get_group_memberships_from_db(prisma_client)

    # Pure-composition groups exist only in the membership table.
    # Surface them so they appear in /access_group/list and existence checks.
    all_groups: Set[str] = set(direct_models.keys())
    for parent, children in group_memberships.items():
        all_groups.add(parent)
        all_groups.update(children)

    # Reverse index: child -> [parents]
    parents_of: Dict[str, List[str]] = {}
    for parent, children in group_memberships.items():
        for child in children:
            parents_of.setdefault(child, []).append(parent)

    # Build the flat group -> [direct model names] map for resolution
    flat_group_models: Dict[str, List[str]] = {
        g: sorted(list(direct_models.get(g, set()))) for g in all_groups
    }

    result: Dict[str, AccessGroupInfo] = {}
    for group in all_groups:
        expanded = resolve_nested_groups(
            group_name=group,
            model_access_groups=flat_group_models,
            group_memberships=group_memberships,
            visited=set(),
        )
        # Dedup while preserving order, then sort for stable response shape
        expanded_unique = sorted(set(expanded))

        result[group] = AccessGroupInfo(
            access_group=group,
            model_names=expanded_unique,
            deployment_count=deployment_count.get(group, 0),
            parent_groups=sorted(parents_of.get(group, [])),
            child_groups=sorted(group_memberships.get(group, [])),
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

    # Validation: Check that at least one of model_names or model_ids is provided
    has_model_names = data.model_names and len(data.model_names) > 0
    has_model_ids = data.model_ids and len(data.model_ids) > 0

    if not has_model_names and not has_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Either model_names or model_ids must be provided and non-empty"
            },
        )

    # If model_ids is provided, use it (more precise targeting)
    use_model_ids = has_model_ids

    # Check if database is connected
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected. Cannot create access group."},
        )

    try:
        # Check if access group already exists. Done before validation so we
        # know the set of known groups (needed to accept nested-group members).
        existing_access_groups = await get_all_access_groups_from_db(
            prisma_client=prisma_client
        )

        if data.access_group in existing_access_groups:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": f"Access group '{data.access_group}' already exists. Use PUT /access_group/{data.access_group}/update to modify it."
                },
            )

        # Validate model_names exist in router or as a known access group
        # (only if using model_names path). model_ids targets specific deployments
        # so nesting doesn't apply.
        if not use_model_ids and has_model_names:
            assert data.model_names is not None
            known_groups = set(existing_access_groups.keys())
            all_valid, missing = validate_models_exist(
                model_names=data.model_names,
                llm_router=llm_router,
                known_access_groups=known_groups,
            )

            if not all_valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"Model(s) or access group(s) not found: {', '.join(missing)}"
                    },
                )

        # Write path. model_ids -> existing per-deployment tagging.
        # model_names -> classify into real models vs. nested child groups and
        # route each to the appropriate table.
        if use_model_ids:
            assert data.model_ids is not None
            models_updated = await update_specific_deployments_with_access_group(
                model_ids=data.model_ids,
                access_group=data.access_group,
                prisma_client=prisma_client,
            )
        else:
            assert data.model_names is not None
            models_updated = await _dual_write_group_membership(
                access_group=data.access_group,
                member_names=data.model_names,
                known_access_groups=set(existing_access_groups.keys()),
                llm_router=llm_router,
                prisma_client=prisma_client,
            )

        await clear_cache()

        verbose_proxy_logger.info(
            f"Successfully created access group '{data.access_group}' with {models_updated} writes"
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

    # Validation: Check that at least one of model_names or model_ids is provided
    has_model_names = data.model_names and len(data.model_names) > 0
    has_model_ids = data.model_ids and len(data.model_ids) > 0

    if not has_model_names and not has_model_ids:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Either model_names or model_ids must be provided and non-empty"
            },
        )

    use_model_ids = has_model_ids

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

    # Validation: Check if all new members exist as router models or known groups.
    # model_ids path targets specific deployments, so nesting doesn't apply.
    if not use_model_ids and has_model_names:
        assert data.model_names is not None
        known_groups = set(access_groups_map.keys())
        all_valid, missing = validate_models_exist(
            model_names=data.model_names,
            llm_router=llm_router,
            known_access_groups=known_groups,
        )

        if not all_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Model(s) or access group(s) not found: {', '.join(missing)}"
                },
            )

    try:
        # Step 1a: clear deployment tags carrying this access group
        await _clear_access_group_from_all_deployments(
            access_group=access_group, prisma_client=prisma_client
        )

        # Step 1b: clear parent->child edges ONLY when re-writing via
        # model_names. The model_ids path targets specific deployments and
        # does not own the membership edges — clearing them here without
        # re-adding them would silently destroy nested-group structure.
        if not use_model_ids:
            await prisma_client.db.litellm_accessgroupmembership.delete_many(
                where={"parent_group": access_group}
            )
            invalidate_group_memberships_cache()

        # Step 2: re-add membership using the appropriate write path
        if use_model_ids:
            assert data.model_ids is not None
            models_updated = await update_specific_deployments_with_access_group(
                model_ids=data.model_ids,
                access_group=access_group,
                prisma_client=prisma_client,
            )
        else:
            assert data.model_names is not None
            models_updated = await _dual_write_group_membership(
                access_group=access_group,
                member_names=data.model_names,
                known_access_groups=set(access_groups_map.keys()),
                llm_router=llm_router,
                prisma_client=prisma_client,
            )

        await clear_cache()

        verbose_proxy_logger.info(
            f"Successfully updated access group '{access_group}' with {models_updated} writes"
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
        # Remove tag from all DB deployments (skip config models)
        models_updated = await _clear_access_group_from_all_deployments(
            access_group=access_group, prisma_client=prisma_client
        )

        # Clean up parent/child edges where this group appears on either side
        # to avoid dangling references in the membership table.
        await delete_group_membership_edges(
            access_group=access_group,
            prisma_client=prisma_client,
        )

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
