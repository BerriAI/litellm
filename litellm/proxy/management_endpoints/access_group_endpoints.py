from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LiteLLM_AccessGroupTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _cache_access_object,
    _cache_key_object,
    _cache_team_object,
    _delete_cache_access_object,
    _get_team_object_from_cache,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.types.access_group import (
    AccessGroupCreateRequest,
    AccessGroupResponse,
    AccessGroupUpdateRequest,
)

router = APIRouter(
    tags=["access group management"],
)


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )


def _record_to_response(record) -> AccessGroupResponse:
    return AccessGroupResponse(
        access_group_id=record.access_group_id,
        access_group_name=record.access_group_name,
        description=record.description,
        access_model_names=record.access_model_names,
        access_mcp_server_ids=record.access_mcp_server_ids,
        access_agent_ids=record.access_agent_ids,
        assigned_team_ids=record.assigned_team_ids,
        assigned_key_ids=record.assigned_key_ids,
        created_at=record.created_at,
        created_by=record.created_by,
        updated_at=record.updated_at,
        updated_by=record.updated_by,
    )


def _record_to_access_group_table(record) -> LiteLLM_AccessGroupTable:
    """Convert a Prisma record to a LiteLLM_AccessGroupTable pydantic object for caching."""
    return LiteLLM_AccessGroupTable(**record.dict())


async def _cache_access_group_record(record) -> None:
    """
    Cache an access group Prisma record in the user_api_key_cache.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    access_group_table = _record_to_access_group_table(record)
    await _cache_access_object(
        access_group_id=record.access_group_id,
        access_group_table=access_group_table,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


async def _invalidate_cache_access_group(access_group_id: str) -> None:
    """
    Invalidate (delete) an access group entry from both in-memory and Redis caches.

    Uses a lazy import of user_api_key_cache and proxy_logging_obj from proxy_server
    to avoid circular imports, following the same pattern as key_management_endpoints.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    await _delete_cache_access_object(
        access_group_id=access_group_id,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


@router.post(
    "/v1/access_group",
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_access_group(
    data: AccessGroupCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    existing = await prisma_client.db.litellm_accessgrouptable.find_unique(
        where={"access_group_name": data.access_group_name}
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Access group '{data.access_group_name}' already exists",
        )

    try:
        record = await prisma_client.db.litellm_accessgrouptable.create(
            data={
                "access_group_name": data.access_group_name,
                "description": data.description,
                "access_model_names": data.access_model_names or [],
                "access_mcp_server_ids": data.access_mcp_server_ids or [],
                "access_agent_ids": data.access_agent_ids or [],
                "assigned_team_ids": data.assigned_team_ids or [],
                "assigned_key_ids": data.assigned_key_ids or [],
                "created_by": user_api_key_dict.user_id,
                "updated_by": user_api_key_dict.user_id,
            }
        )
    except Exception as e:
        # Race condition: another request created the same name between find_unique and create.
        # Prisma raises UniqueViolationError (P2002) or similar for unique constraint.
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{data.access_group_name}' already exists",
            )
        raise

    # Cache the newly created access group for read-heavy access patterns
    await _cache_access_group_record(record)

    return _record_to_response(record)


@router.get(
    "/v1/access_group",
    response_model=List[AccessGroupResponse],
)
async def list_access_groups(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[AccessGroupResponse]:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    records = await prisma_client.db.litellm_accessgrouptable.find_many(
        order={"created_at": "desc"}
    )
    return [_record_to_response(r) for r in records]


@router.get(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def get_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    record = await prisma_client.db.litellm_accessgrouptable.find_unique(
        where={"access_group_id": access_group_id}
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )
    return _record_to_response(record)


@router.put(
    "/v1/access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def update_access_group(
    access_group_id: str,
    data: AccessGroupUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    existing = await prisma_client.db.litellm_accessgrouptable.find_unique(
        where={"access_group_id": access_group_id}
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )

    update_data: dict = {"updated_by": user_api_key_dict.user_id}
    for field, value in data.model_dump(exclude_unset=True).items():
        update_data[field] = value

    try:
        record = await prisma_client.db.litellm_accessgrouptable.update(
            where={"access_group_id": access_group_id},
            data=update_data,
        )
    except Exception as e:
        # Unique constraint violation (e.g. access_group_name already exists).
        if "unique constraint" in str(e).lower() or "P2002" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Access group '{update_data.get('access_group_name', '')}' already exists",
            )
        raise

    # Write the updated record into cache (same key, overwrites stale entry)
    await _cache_access_group_record(record)

    return _record_to_response(record)


@router.delete(
    "/v1/access_group/{access_group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> None:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    try:
        # Track affected team IDs and key tokens for cache invalidation
        affected_team_ids: list = []
        affected_key_tokens: list = []

        async with prisma_client.db.tx() as tx:
            existing = await tx.litellm_accessgrouptable.find_unique(
                where={"access_group_id": access_group_id}
            )
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Access group '{access_group_id}' not found",
                )

            # Remove access_group_id from teams and keys that reference it
            teams_with_group = await tx.litellm_teamtable.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            for team in teams_with_group:
                affected_team_ids.append(team.team_id)
                updated_ids = [tid for tid in (team.access_group_ids or []) if tid != access_group_id]
                await tx.litellm_teamtable.update(
                    where={"team_id": team.team_id},
                    data={"access_group_ids": updated_ids},
                )

            keys_with_group = await tx.litellm_verificationtoken.find_many(
                where={"access_group_ids": {"hasSome": [access_group_id]}}
            )
            for key in keys_with_group:
                affected_key_tokens.append(key.token)
                updated_ids = [kid for kid in (key.access_group_ids or []) if kid != access_group_id]
                await tx.litellm_verificationtoken.update(
                    where={"token": key.token},
                    data={"access_group_ids": updated_ids},
                )

            await tx.litellm_accessgrouptable.delete(
                where={"access_group_id": access_group_id}
            )

        # Invalidate the deleted access group from cache
        await _invalidate_cache_access_group(access_group_id)

        # Patch cached team and key objects to remove the deleted access_group_id
        # instead of fully invalidating them (keeps cache warm, avoids DB re-fetch)
        from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

        for team_id in affected_team_ids:
            cached_team = await _get_team_object_from_cache(
                key="team_id:{}".format(team_id),
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=None,
            )
            if cached_team is not None and cached_team.access_group_ids:
                cached_team.access_group_ids = [
                    ag_id for ag_id in cached_team.access_group_ids if ag_id != access_group_id
                ]
                await _cache_team_object(
                    team_id=team_id,
                    team_table=cached_team,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )

        for token in affected_key_tokens:
            cached_key = await user_api_key_cache.async_get_cache(key=token)
            if cached_key is not None:
                if isinstance(cached_key, dict):
                    cached_key = UserAPIKeyAuth(**cached_key)
                if isinstance(cached_key, UserAPIKeyAuth) and cached_key.access_group_ids:
                    cached_key.access_group_ids = [
                        ag_id for ag_id in cached_key.access_group_ids if ag_id != access_group_id
                    ]
                    await _cache_key_object(
                        hashed_token=token,
                        user_api_key_obj=cached_key,
                        user_api_key_cache=user_api_key_cache,
                        proxy_logging_obj=proxy_logging_obj,
                    )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "delete_access_group failed: access_group_id=%s error=%s",
            access_group_id,
            e,
        )
        if PrismaDBExceptionHandler.is_database_connection_error(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CommonProxyErrors.db_not_connected_error.value,
            )
        if "P2025" in str(e) or ("record" in str(e).lower() and "not found" in str(e).lower()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Access group '{access_group_id}' not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete access group. Please try again.",
        )


# Alias routes for /v1/unified_access_group
router.add_api_route(
    "/v1/unified_access_group",
    create_access_group,
    methods=["POST"],
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
router.add_api_route(
    "/v1/unified_access_group",
    list_access_groups,
    methods=["GET"],
    response_model=List[AccessGroupResponse],
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    get_access_group,
    methods=["GET"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    update_access_group,
    methods=["PUT"],
    response_model=AccessGroupResponse,
)
router.add_api_route(
    "/v1/unified_access_group/{access_group_id}",
    delete_access_group,
    methods=["DELETE"],
    status_code=status.HTTP_204_NO_CONTENT,
)
