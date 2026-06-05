from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, cast

from fastapi import HTTPException, Request, status

from litellm.integrations.otel.runtime import seed_request_identity

from .audit import AuthzDecision, Decision, record
from .authn.authenticators import AuthContext, AuthResult, authenticate
from .authz.authorizer import AuthorizationDenied, authorize
from .authz.enforcer import CasbinEnforcer
from .authz.policy_store import load_policy_snapshot
from .authz.route_map import is_inference_route, match_route
from .context import AuthMethod, RequestAuthContext, set_auth_context
from .principal import Principal, build_principal
from .protocols import PolicyDB
from .stages.budgets import enforce_hierarchy_budgets
from .stages.end_user import resolve_end_user
from .stages.enrichment import enrich_identity

if TYPE_CHECKING:
    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LiteLLM_UserTable,
        UserAPIKeyAuth,
    )
    from litellm.proxy.utils import PrismaClient


async def _anonymous_identity(api_key: Optional[str]) -> UserAPIKeyAuth:
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key=api_key)


async def _build_enforcer(
    principal: Principal, prisma_client: Optional[PrismaClient]
) -> CasbinEnforcer:
    """Build the casbin engine for this principal from the current policy snapshot.

    The principal's identity-to-role bridges are added on top of the stored
    groupings. One engine authorizes both the control plane and model calls.
    """
    # The Prisma client's casbin-rule table is generated dynamically; narrow it to
    # the read surface the store needs at this single adapter boundary.
    policy_db = cast(Optional[PolicyDB], prisma_client)
    (
        policies,
        groupings,
        resource_groupings,
        domain_groupings,
    ) = await load_policy_snapshot(policy_db)
    return CasbinEnforcer(
        policies,
        groupings + principal.groupings,
        resource_groupings,
        domain_groupings,
    )


async def _enrich_for_limits(identity: UserAPIKeyAuth, ctx: AuthContext) -> None:
    """Fill user/team budget+limit fields for non-key logins (master/JWT/OAuth) so
    the existing pre-call budget/limit hooks can enforce them. Virtual keys are
    already populated by get_key_object and skip this."""
    from litellm.proxy.auth.auth_checks import get_team_object, get_user_object

    async def load_user(user_id: str) -> Optional[LiteLLM_UserTable]:
        return await get_user_object(
            user_id=user_id,
            prisma_client=ctx.prisma_client,
            user_api_key_cache=ctx.user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=ctx.parent_otel_span,
            proxy_logging_obj=ctx.proxy_logging_obj,
        )

    async def load_team(team_id: str) -> Optional[LiteLLM_TeamTable]:
        return await get_team_object(
            team_id=team_id,
            prisma_client=ctx.prisma_client,
            user_api_key_cache=ctx.user_api_key_cache,
            parent_otel_span=ctx.parent_otel_span,
            proxy_logging_obj=ctx.proxy_logging_obj,
        )

    await enrich_identity(identity, load_user=load_user, load_team=load_team)


async def _enforce_budgets(
    identity: UserAPIKeyAuth, route: str, ctx: AuthContext
) -> None:
    """Enforce team/org/global budgets (reusing v1's functions) and surface a
    breach as the same ProxyException v1 raises."""
    import litellm
    from litellm.proxy._types import ProxyErrorTypes, ProxyException

    try:
        await enforce_hierarchy_budgets(identity, route, ctx)
    except litellm.BudgetExceededError as e:
        raise ProxyException(
            message=e.message,
            type=ProxyErrorTypes.budget_exceeded,
            param=None,
            code=getattr(e, "status_code", status.HTTP_429_TOO_MANY_REQUESTS),
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
) -> Tuple[Principal, UserAPIKeyAuth]:
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
) -> UserAPIKeyAuth:
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
            authorize(
                principal,
                route,
                request_data,
                enforcer,
                request.method,
                auth_method=result.method.value,
            )
        except AuthorizationDenied as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

        seed_request_identity(identity)
        identity.request_route = route
        return identity

    if is_inference_route(route):
        # Model calls are a permission like any other: the `call` action on the
        # `model:<id>` object, decided by the same role system. The legacy
        # key.models / access-group mechanism is intentionally not consulted.
        result = await authenticate(token, ctx)
        request_data = await _read_request_body(request=request)
        if result.method is not AuthMethod.VIRTUAL_KEY:
            await _enrich_for_limits(result.identity, ctx)
        principal, identity = _establish_context(request, result, route)
        requested_model = (
            request_data.get("model") if isinstance(request_data, dict) else None
        )
        if requested_model:
            enforcer = await _build_enforcer(principal, prisma_client)
            obj = f"model:{requested_model}"
            allowed = enforcer.enforce(principal.subject, principal.domain, obj, "call")
            record(
                AuthzDecision(
                    decision=Decision.ALLOW if allowed else Decision.DENY,
                    subject=principal.subject,
                    domain=principal.domain,
                    obj=obj,
                    action="call",
                    route=route,
                    reason="model call",
                    auth_method=result.method.value,
                )
            )
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"auth_v2: not permitted to call model '{requested_model}'",
                )
        await _enforce_budgets(identity, route, ctx)
        await resolve_end_user(request, request_data, dict(request.headers))
        seed_request_identity(identity, model=requested_model)
        identity.request_route = route
        return identity

    # Loud-open: route v2 doesn't yet govern. No identity required.
    result = await _best_effort_identity(token, ctx)
    principal, identity = _establish_context(request, result, route)
    authorize(
        principal,
        route,
        None,
        _DENY_ALL,
        request.method,
        auth_method=result.method.value,
    )
    seed_request_identity(identity)
    identity.request_route = route
    return identity


class _DenyAll:
    def enforce(self, *_args: object) -> bool:
        return False


# Only ever consulted on ungoverned routes, where authorize() short-circuits to
# loud-open before calling enforce(); present so the call signature is uniform.
_DENY_ALL = _DenyAll()
