from typing import Any, Optional

from fastapi import HTTPException, Request, status

from .authenticators import AuthContext, authenticate
from .authorizer import AuthorizationDenied, authorize
from .data_plane import can_call_model
from .enforcer import CasbinEnforcer
from .policy_store import load_policy_snapshot
from .principal import build_principal
from .route_map import is_inference_route, match_route


async def _anonymous_identity(api_key: Optional[str]) -> Any:
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key=api_key)


async def _best_effort_identity(api_key: Optional[str], ctx: AuthContext) -> Any:
    """On loud-open routes, use the real identity if a usable key is present,
    otherwise fall back to an anonymous principal. Never fails the request."""
    if isinstance(api_key, str) and api_key.startswith("sk-"):
        try:
            return await authenticate(api_key, ctx)
        except Exception:
            pass
    return await _anonymous_identity(api_key)


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

    rule = match_route(route)
    if rule is not None:
        # Control plane: RBAC policy rows.
        identity = await authenticate(token, ctx)
        request_data = await _read_request_body(request=request)
        principal = build_principal(identity)

        (
            policies,
            groupings,
            resource_groupings,
            domain_groupings,
        ) = await load_policy_snapshot(prisma_client)
        enforcer = CasbinEnforcer(
            policies,
            groupings + principal.groupings,
            resource_groupings,
            domain_groupings,
        )

        try:
            authorize(principal, route, request_data, enforcer)
        except AuthorizationDenied as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(e)
            )

        identity.request_route = route
        return identity

    if is_inference_route(route):
        # Data plane: casbin ABAC over the principal's allowed-model attribute.
        identity = await authenticate(token, ctx)
        request_data = await _read_request_body(request=request)
        requested_model = (
            request_data.get("model") if isinstance(request_data, dict) else None
        )
        if requested_model and not can_call_model(
            getattr(identity, "models", None), requested_model
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"auth_v2: not permitted to call model '{requested_model}'",
            )
        identity.request_route = route
        return identity

    # Loud-open: route v2 doesn't yet govern. No identity required.
    identity = await _best_effort_identity(token, ctx)
    authorize(build_principal(identity), route, None, _DENY_ALL)
    identity.request_route = route
    return identity


class _DenyAll:
    def enforce(self, *_args: Any) -> bool:
        return False


# Only ever consulted on ungoverned routes, where authorize() short-circuits to
# loud-open before calling enforce(); present so the call signature is uniform.
_DENY_ALL = _DenyAll()
