"""Admin API for inspecting requests currently running on the proxy."""

from datetime import datetime, timezone
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.active_request_registry import ActiveRequestRegistry

router: Final = APIRouter()

USER_API_KEY_AUTH: Final = Depends(user_api_key_auth)
ADMIN_ROLES: Final = frozenset((LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))


def _get_active_request_registry() -> ActiveRequestRegistry | None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    hook: Final = proxy_logging_obj.get_proxy_hook("active_request_registry")
    return hook if isinstance(hook, ActiveRequestRegistry) else None


class ActiveRequestRecord(BaseModel):
    registry_id: str
    request_id: str
    started_at: float
    model: str | None = None
    call_type: str | None = None
    streaming: bool = False
    route: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    end_user_id: str | None = None
    organization_id: str | None = None
    organization_alias: str | None = None
    project_id: str | None = None
    project_alias: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    key_alias: str | None = None
    key_fingerprint: str | None = None
    pod: str | None = None


class ActiveRequestsResponse(BaseModel):
    available: bool
    reason: str | None = None
    items: tuple[ActiveRequestRecord, ...]
    total: int
    page: int
    page_size: int
    truncated: bool = False
    generated_at: datetime


@router.get(
    "/global/active_requests",
    response_model=ActiveRequestsResponse,
    tags=["Active Requests"],  # mutable-ok: FastAPI types tags as a list
    dependencies=[USER_API_KEY_AUTH],  # mutable-ok: FastAPI types dependencies as a list
)
async def get_active_requests(
    response: Response,
    model: str | None = None,
    user_id: str | None = None,
    end_user_id: str | None = None,
    organization_id: str | None = None,
    project_id: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    user_api_key_dict: UserAPIKeyAuth = USER_API_KEY_AUTH,
) -> ActiveRequestsResponse:
    response.headers["Cache-Control"] = "no-store, max-age=0"  # rebind-ok: FastAPI injects the Response
    response.headers["Pragma"] = "no-cache"  # rebind-ok: FastAPI injects the Response
    if user_api_key_dict.user_role not in ADMIN_ROLES:
        raise HTTPException(
            status_code=403,
            detail="Only proxy administrators can view global active requests",
        )

    registry: Final = _get_active_request_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Active request registry is unavailable")

    try:
        result: Final = await registry.list_requests(
            model=model,
            user_id=user_id,
            end_user_id=end_user_id,
            organization_id=organization_id,
            project_id=project_id,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        verbose_proxy_logger.exception("Failed to list active requests")
        raise HTTPException(status_code=503, detail="Unable to read the active request registry") from exc

    return ActiveRequestsResponse(
        **result,
        generated_at=datetime.now(timezone.utc),
    )


class CancelActiveRequestResponse(BaseModel):
    cancelled: bool
    detail: str


@router.post(
    "/global/active_requests/{registry_id}/cancel",
    response_model=CancelActiveRequestResponse,
    tags=["Active Requests"],  # mutable-ok: FastAPI types tags as a list
    dependencies=[USER_API_KEY_AUTH],  # mutable-ok: FastAPI types dependencies as a list
)
async def cancel_active_request(
    registry_id: str,
    user_api_key_dict: UserAPIKeyAuth = USER_API_KEY_AUTH,
) -> CancelActiveRequestResponse:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy administrators can cancel active requests")

    registry: Final = _get_active_request_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Active request registry is unavailable")

    cancelled: Final = await registry.request_cancel(registry_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="That request is no longer running")
    return CancelActiveRequestResponse(
        cancelled=True,
        detail="Cancellation sent to the worker serving the request",
    )
