from typing import Any, Optional, Tuple

from fastapi import HTTPException, Request, status

from .authenticators import AuthContext, AuthResult, authenticate
from .authorizer import AuthorizationDenied, authorize
from .context import AuthMethod, RequestAuthContext, set_auth_context
from .end_user import resolve_end_user
from .enforcer import CasbinEnforcer
from .policy_store import load_policy_snapshot
from .principal import Principal, build_principal
from .route_map import is_inference_route, match_route


async def _anonymous_identity(api_key: Optional[str]) -> Any:
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key=api_key)


async def _build_enforcer(principal: Principal, prisma_client: Any) -> CasbinEnforcer:
    """Build the casbin engine for this principal from the current policy snapshot.

    The principal's identity-to-role bridges are added on top of the stored
    groupings. One engine authorizes both the control plane and model calls.
    """
    (
        policies,
        groupings,
        resource_groupings,
        domain_groupings,
    ) = await load_policy_snapshot(prisma_client)
    return CasbinEnforcer(
        policies,
        groupings + principal.groupings,
        resource_groupings,
        domain_groupings,
    )


async def _best_effort_identity(api_key: Optional[str], ctx: AuthContext) -> AuthResult:
    """On loud-open routes, use the real identity if a usable key is present,
    otherwise fall back to an anonymous principal. Never fails the request."""
    if isinstance(api_key, str) and api_key.startswith("sk-"):
        try:
            return await authenticate(api_key, ctx)
        except Exception:
            pass
    identity = await _anonymous_identity(api_key)
    return AuthResult(identity=identity, method=AuthMethod.ANONYMOUS)


def _establish_context(
    request: Request, result: AuthResult, route: str
) -> Tuple[Principal, Any]:
    """Derive the principal and publish the typed auth context for downstream
    stages (budget/limit hooks, end-user resolver, telemetry span)."""
    principal = build_principal(result.identity)
    set_auth_context(
        request,
        RequestAuthContext(
            identity=result.identity,
            principal=principal,
            auth_method=result.method,
            route=route,
        ),
    )
    return principal, result.identity


async def user_api_key_auth_v2(
    request: Request,
    api_key: str = "",
) -> Any:
    """auth_v2 entry point: authenticator chain establishes identity, casbin
    authorizes governed routes. Routes v2 doesn't yet own are loud-open."""
    from litellm.proxy.auth.user_api_key_auth import _get_bearer_token
    from litellm.proxy.auth.auth_utils import get_request_route
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    token = _get_bearer_token(api_key=api_key) if api_key else api_key
    route = get_request_route(request=request)
    ctx = AuthContext(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    rule = match_route(route, request.method)
    if rule is not None:
        # Control plane: RBAC over management resources.
        result = await authenticate(token, ctx)
        request_data = await _read_request_body(request=request)
        principal, identity = _establish_context(request, result, route)
        enforcer = await _build_enforcer(principal, prisma_client)

        try:
            authorize(principal, route, request_data, enforcer, request.method)
        except AuthorizationDenied as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

        identity.request_route = route
        return identity

    if is_inference_route(route):
        # Model calls are a permission like any other: the `call` action on the
        # `model:<id>` object, decided by the same role system. The legacy
        # key.models / access-group mechanism is intentionally not consulted.
        result = await authenticate(token, ctx)
        request_data = await _read_request_body(request=request)
        principal, identity = _establish_context(request, result, route)
        requested_model = (
            request_data.get("model") if isinstance(request_data, dict) else None
        )
        if requested_model:
            enforcer = await _build_enforcer(principal, prisma_client)
            if not enforcer.enforce(
                principal.subject, principal.domain, f"model:{requested_model}", "call"
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"auth_v2: not permitted to call model '{requested_model}'",
                )
        await resolve_end_user(request, request_data, dict(request.headers))
        identity.request_route = route
        return identity

    # Loud-open: route v2 doesn't yet govern. No identity required.
    result = await _best_effort_identity(token, ctx)
    principal, identity = _establish_context(request, result, route)
    authorize(principal, route, None, _DENY_ALL, request.method)
    identity.request_route = route
    return identity


class _DenyAll:
    def enforce(self, *_args: Any) -> bool:
        return False


# Only ever consulted on ungoverned routes, where authorize() short-circuits to
# loud-open before calling enforce(); present so the call signature is uniform.
_DENY_ALL = _DenyAll()
