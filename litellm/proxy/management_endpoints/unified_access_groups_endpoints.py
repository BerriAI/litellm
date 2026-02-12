from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.types.unified_access_group import (
    AccessGroupCreateRequest,
    AccessGroupResponse,
    AccessGroupUpdateRequest,
)

router = APIRouter(
    tags=["access group management"],
    dependencies=[Depends(user_api_key_auth)],
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
        access_model_ids=record.access_model_ids,
        access_mcp_server_ids=record.access_mcp_server_ids,
        access_agent_ids=record.access_agent_ids,
        assigned_team_ids=record.assigned_team_ids,
        assigned_key_ids=record.assigned_key_ids,
        created_at=record.created_at,
        created_by=record.created_by,
        updated_at=record.updated_at,
        updated_by=record.updated_by,
    )


@router.post(
    "/v1/unified_access_group",
    response_model=AccessGroupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_unified_access_group(
    data: AccessGroupCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    existing = await prisma_client.db.litellm_unifiedaccessgroup.find_unique(
        where={"access_group_name": data.access_group_name}
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Access group '{data.access_group_name}' already exists",
        )

    record = await prisma_client.db.litellm_unifiedaccessgroup.create(
        data={
            "access_group_name": data.access_group_name,
            "description": data.description,
            "access_model_ids": data.access_model_ids or [],
            "access_mcp_server_ids": data.access_mcp_server_ids or [],
            "access_agent_ids": data.access_agent_ids or [],
            "assigned_team_ids": data.assigned_team_ids or [],
            "assigned_key_ids": data.assigned_key_ids or [],
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
    )
    return _record_to_response(record)


@router.get(
    "/v1/unified_access_group",
    response_model=List[AccessGroupResponse],
)
async def list_unified_access_groups(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[AccessGroupResponse]:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    records = await prisma_client.db.litellm_unifiedaccessgroup.find_many(
        order={"created_at": "desc"}
    )
    return [_record_to_response(r) for r in records]


@router.get(
    "/v1/unified_access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def get_unified_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    record = await prisma_client.db.litellm_unifiedaccessgroup.find_unique(
        where={"access_group_id": access_group_id}
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )
    return _record_to_response(record)


@router.put(
    "/v1/unified_access_group/{access_group_id}",
    response_model=AccessGroupResponse,
)
async def update_unified_access_group(
    access_group_id: str,
    data: AccessGroupUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AccessGroupResponse:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    existing = await prisma_client.db.litellm_unifiedaccessgroup.find_unique(
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

    record = await prisma_client.db.litellm_unifiedaccessgroup.update(
        where={"access_group_id": access_group_id},
        data=update_data,
    )
    return _record_to_response(record)


@router.delete(
    "/v1/unified_access_group/{access_group_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_unified_access_group(
    access_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> None:
    _require_proxy_admin(user_api_key_dict)
    prisma_client = get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value)

    existing = await prisma_client.db.litellm_unifiedaccessgroup.find_unique(
        where={"access_group_id": access_group_id}
    )
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access group '{access_group_id}' not found",
        )

    await prisma_client.db.litellm_unifiedaccessgroup.delete(
        where={"access_group_id": access_group_id}
    )
