"""Operator surface for `lite login` sessions: list them, and cut one off mid-session."""

from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, status

from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    UserAPIKeyAuth,
    user_api_key_has_admin_view,
)
from litellm.proxy.auth.cli_session_registry import list_cli_sessions, revoke_cli_session
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.types.cli_session import CLISessionListResponse, CLISessionResponse

router: Final = APIRouter()


@router.get("/cli/session/list", response_model=CLISessionListResponse)
async def list_cli_sessions_endpoint(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> CLISessionListResponse:
    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=CommonProxyErrors.not_allowed_access.value,
        )
    return await list_cli_sessions(
        prisma_client=get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value),
        page=page,
        page_size=page_size,
    )


@router.post("/cli/session/{session_id}/revoke", response_model=CLISessionResponse)
async def revoke_cli_session_endpoint(
    session_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CLISessionResponse:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=CommonProxyErrors.not_allowed_access.value,
        )
    from litellm.proxy.proxy_server import user_api_key_cache

    revoked: Final = await revoke_cli_session(
        prisma_client=get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value),
        user_api_key_cache=user_api_key_cache,
        session_id=session_id,
        revoked_by=user_api_key_dict.user_id,
    )
    if revoked is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CLI session not found: {session_id}",
        )
    return revoked
