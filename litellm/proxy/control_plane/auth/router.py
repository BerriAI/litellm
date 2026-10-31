from typing import Annotated, Final

from fastapi import APIRouter, Depends, Security

from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.login import InvalidCredentials, LoginService, Throttled
from litellm.proxy.control_plane.auth.rbac import Principal
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, too_many_requests, unauthorized
from litellm.types.proxy.control_plane.auth import LoginRequest, PrincipalResponse, SessionTokenResponse

router: Final = APIRouter(prefix="/auth", tags=["control plane: auth"])


def get_login_service() -> LoginService:
    raise RuntimeError("get_login_service is not bound, build the app with create_app")


def principal_response(principal: Principal) -> PrincipalResponse:
    return PrincipalResponse(
        subject=principal.subject,
        email=principal.email,
        roles=tuple(sorted(principal.roles)),
        scopes=tuple(sorted(scope.value for scope in principal.scopes)),
        dataplanes=None if principal.dataplanes.all else tuple(sorted(principal.dataplanes.ids)),
        token_kind=principal.token_kind,
    )


@router.post("/login", response_model=SessionTokenResponse)
async def login(
    body: LoginRequest,
    service: Annotated[LoginService, Depends(get_login_service)],
) -> SessionTokenResponse:
    match service.login(body.username, body.password):
        case SessionTokenResponse() as token:
            return token
        case Throttled(retry_after_seconds=seconds):
            raise ControlPlaneException(too_many_requests(f"Too many failed logins, retry in {seconds} seconds."))
        case InvalidCredentials():
            raise ControlPlaneException(unauthorized("Invalid username or password."))


@router.get("/me", response_model=PrincipalResponse)
async def me(principal: Annotated[Principal, Security(require_scopes)]) -> PrincipalResponse:
    return principal_response(principal)
