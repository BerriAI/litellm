"""IN-FLIGHT REQUESTS

GET /in_flight/requests - the requests this worker is serving right now.
POST /in_flight/cancel - cancel one by handle or every running one.

Cancelling closes the task that owns the upstream connection, so the client
sees the stream end and the provider stops generating on disconnect. Entries
are addressed by a server-generated handle (unique per request), not by the
client-supplied litellm_call_id. The registry is per-process; on multi-worker
deployments target the worker serving the request (see
litellm/proxy/in_flight_registry.py)."""

import os
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, Field

from litellm.proxy import in_flight_registry
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter()

_ADMIN_ROLES: Final = (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)


class InFlightCancelRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    handles: tuple[str, ...] | None = Field(
        default=None, description="handles to cancel; omit to cancel none unless all is true"
    )
    all: bool = Field(default=False, description="cancel every running request on this worker")


class InFlightRequestItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    handle: str
    request_id: str
    started_at: float
    model: str
    user_id: str | None = None
    key_alias: str | None = None
    is_streaming: bool = False
    state: Literal["running", "cancelling"] = "running"


class InFlightListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    worker_pid: int
    requests: tuple[InFlightRequestItem, ...]


class InFlightCancelResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    cancelled: tuple[str, ...]


def _require_admin(user_api_key_dict: UserAPIKeyAuth, *, write: bool) -> None:
    if write and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only the proxy admin can cancel in-flight requests")
    if not write and user_api_key_dict.user_role not in _ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Only proxy admin roles can list in-flight requests")


@router.get(
    "/in_flight/requests",
    tags=["in-flight"],
    response_model=InFlightListResponse,
)
async def list_in_flight_requests(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InFlightListResponse:
    """Requests this worker is serving right now, oldest first."""
    _require_admin(user_api_key_dict, write=False)

    return InFlightListResponse(
        worker_pid=os.getpid(),
        requests=tuple(InFlightRequestItem.model_validate(dict(item)) for item in in_flight_registry.snapshot()),
    )


@router.post(
    "/in_flight/cancel",
    tags=["in-flight"],
    response_model=InFlightCancelResponse,
)
async def cancel_in_flight_requests(
    body: InFlightCancelRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> InFlightCancelResponse:
    """Cancel running requests by handle, or all of them when ``all`` is true."""
    _require_admin(user_api_key_dict, write=True)
    if not body.all and not body.handles:
        raise HTTPException(status_code=400, detail="Pass handles or set all=true")
    cancelled: Final = in_flight_registry.cancel(None if body.all else tuple(body.handles or ()))
    return InFlightCancelResponse(cancelled=cancelled)
