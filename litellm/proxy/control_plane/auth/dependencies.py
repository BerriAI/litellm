"""FastAPI glue: bearer header -> Principal, scope checks via Security."""

from typing import Annotated, Final

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes
from starlette.concurrency import run_in_threadpool

from litellm.proxy.control_plane.auth.jwt import Rejected, TokenVerifier
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, forbidden, unauthorized
from litellm.proxy.control_plane.common.tracing import UNTRACED, Telemetry, annotate_principal

bearer_scheme: Final = HTTPBearer(auto_error=False, scheme_name="ControlPlaneBearer")


def get_token_verifier() -> TokenVerifier:
    """Placeholder bound in create_app via dependency_overrides, the FastAPI way to inject a per-process singleton."""
    raise RuntimeError("get_token_verifier is not bound, build the app with create_app")


def get_telemetry() -> Telemetry:
    return UNTRACED


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
    telemetry: Annotated[Telemetry, Depends(get_telemetry)],
) -> Principal:
    if credentials is None:
        raise ControlPlaneException(unauthorized("Missing bearer credentials."))
    verified: Final = await run_in_threadpool(verifier.verify, credentials.credentials)
    if isinstance(verified, Rejected):
        raise ControlPlaneException(unauthorized(verified.reason))
    annotate_principal(verified, telemetry.identity_attributes)
    return verified


async def require_scopes(
    security_scopes: SecurityScopes,
    principal: Annotated[Principal, Depends(get_principal)],
) -> Principal:
    required: Final = frozenset(Scope(scope) for scope in security_scopes.scopes)
    missing: Final = principal.missing_scopes(required)
    if missing:
        raise ControlPlaneException(forbidden(f"Missing scope: {', '.join(sorted(scope.value for scope in missing))}."))
    return principal
