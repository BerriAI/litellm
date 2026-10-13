from http import HTTPStatus
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Security

from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, bad_gateway
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient, DataplaneReply, DataplaneUnreachable
from litellm.proxy.control_plane.dataplanes.credentials import Caller, CredentialUnavailable
from litellm.proxy.control_plane.dataplanes.dependencies import (
    dataplane_access,
    get_caller,
    get_dataplane_client,
    get_registry,
)
from litellm.proxy.control_plane.dataplanes.identity import USER_INFO_PATH, UserInfo, parse_user_info
from litellm.proxy.control_plane.dataplanes.registry import Dataplane, DataplaneRegistry, visible_to
from litellm.types.proxy.control_plane.dataplanes import (
    DataplaneHealthResponse,
    DataplaneIdentityResponse,
    DataplaneListResponse,
    DataplaneResponse,
)

router: Final = APIRouter(prefix="/dataplanes", tags=["dataplanes"])

LIVENESS_PATH: Final = "/health/liveliness"


def _to_response(dataplane: Dataplane) -> DataplaneResponse:
    return DataplaneResponse(
        id=dataplane.id, name=dataplane.name, url=dataplane.url, credentials=dataplane.credential_kinds
    )


@router.get("", response_model=DataplaneListResponse)
async def list_dataplanes(
    principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.DATAPLANES_READ.value])],
    registry: Annotated[DataplaneRegistry, Depends(get_registry)],
) -> DataplaneListResponse:
    return DataplaneListResponse(items=tuple(_to_response(dataplane) for dataplane in visible_to(registry, principal)))


@router.get("/{dataplane_id}", response_model=DataplaneResponse)
async def get_dataplane(
    _principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.DATAPLANES_READ.value])],
    dataplane: Annotated[Dataplane, Depends(dataplane_access)],
) -> DataplaneResponse:
    return _to_response(dataplane)


@router.get("/{dataplane_id}/health", response_model=DataplaneHealthResponse)
async def dataplane_health(
    _principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.DATAPLANES_READ.value])],
    dataplane: Annotated[Dataplane, Depends(dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
) -> DataplaneHealthResponse:
    result: Final = await client.get(dataplane, LIVENESS_PATH, caller)
    match result:
        case DataplaneReply(status=status) if HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            return DataplaneHealthResponse(id=dataplane.id, status="ok", http_status=status, detail=None)
        case DataplaneReply(status=status, body=body):
            return DataplaneHealthResponse(id=dataplane.id, status="degraded", http_status=status, detail=str(body))
        case DataplaneUnreachable(detail=detail):
            return DataplaneHealthResponse(id=dataplane.id, status="unreachable", http_status=None, detail=detail)
        case CredentialUnavailable(reason=reason):
            return DataplaneHealthResponse(id=dataplane.id, status="degraded", http_status=None, detail=reason)


@router.get("/{dataplane_id}/whoami", response_model=DataplaneIdentityResponse)
async def dataplane_whoami(
    _principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.DATAPLANES_READ.value])],
    dataplane: Annotated[Dataplane, Depends(dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
) -> DataplaneIdentityResponse:
    """Ask the dataplane who it authenticated, proving the caller's identity survived the hop."""
    result: Final = await client.get(dataplane, USER_INFO_PATH, caller)
    match result:
        case DataplaneReply(status=status, body=body, credential=credential) if status < 400:
            info: Final = parse_user_info(body)
            if info is None:
                raise ControlPlaneException(bad_gateway("Dataplane returned a malformed /user/info body."))
            user: Final = info.user_info or UserInfo()
            return DataplaneIdentityResponse(
                id=dataplane.id,
                credential=credential,
                user_id=info.user_id or user.user_id,
                user_email=user.user_email,
                user_role=user.user_role,
            )
        case DataplaneReply(status=HTTPStatus.NOT_FOUND, credential=credential):
            return DataplaneIdentityResponse(
                id=dataplane.id, credential=credential, user_id=None, user_email=None, user_role=None
            )
        case DataplaneReply(status=status, credential=credential):
            raise ControlPlaneException(
                bad_gateway(f"Dataplane rejected the {credential} credential with HTTP {status}.")
            )
        case DataplaneUnreachable(detail=detail, attempts=attempts):
            raise ControlPlaneException(bad_gateway(f"Dataplane unreachable after {attempts} attempt(s): {detail}"))
        case CredentialUnavailable(reason=reason):
            raise ControlPlaneException(bad_gateway(f"No usable credential for this dataplane: {reason}"))
