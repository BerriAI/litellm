from collections.abc import Collection, Iterable, Mapping
from dataclasses import replace
from typing import Final, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, TypeAdapter

from litellm._auth import routes
from litellm.proxy._types import LiteLLM_UserTable, LiteLLMRoutes, UserAPIKeyAuth

_STRINGS: Final = TypeAdapter(tuple[str, ...])


class _TokenFields(BaseModel, frozen=True):
    allowed_routes: object = None
    metadata: Mapping[str, object] | None = Field(default_factory=dict[str, object])
    permissions: Mapping[str, object] | None = None


class _ProxyFields(BaseModel, frozen=True):
    general_settings: Mapping[str, object]
    premium_user: bool


def _allowed_routes(value: object) -> tuple[str, ...] | None:
    return _STRINGS.validate_python(value) if isinstance(value, list) else None


def route_catalog() -> routes.RouteCatalog:
    from litellm.proxy.agent_endpoints.auth.managed_authorization import (
        _MANAGED_LIVE_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _MANAGED_MCP_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _MANAGED_MODEL_PATHS,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _MANAGED_MODEL_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _MANAGED_REALTIME_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
    )
    from litellm.proxy.auth.route_checks import (
        _PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _SELF_SCOPED_CREDENTIAL_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        _UNBILLED_ROUTES,  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        RouteChecks,
    )

    return routes.RouteCatalog(
        groups={name: tuple(member.value) for name, member in LiteLLMRoutes.__members__.items()},
        viewer_blocked=tuple(_PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES),
        viewer_write_blocked=tuple(RouteChecks._ADMIN_VIEWER_BLOCKED_WRITE_ROUTES),  # pyright: ignore[reportPrivateUsage]  # read-only legacy catalog projection
        self_credentials=_SELF_SCOPED_CREDENTIAL_ROUTES,
        managed_models=tuple(_MANAGED_MODEL_ROUTES),
        managed_paths=_MANAGED_MODEL_PATHS,
        managed_realtime=tuple(_MANAGED_REALTIME_ROUTES),
        managed_live=tuple(_MANAGED_LIVE_ROUTES),
        managed_mcp=_MANAGED_MCP_ROUTES,
        unbilled=tuple(_UNBILLED_ROUTES),
    )


def route_context(route: str, method: str | None = None, *, registration: bool = False) -> routes.RouteContext:
    if not registration:
        return routes.RouteContext(route=route, method=method, catalog=route_catalog())
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    registered: Final = InitPassThroughEndpointHelpers.get_registered_pass_through_route(route=route, method=method)
    return routes.RouteContext(
        route=route,
        method=method,
        catalog=route_catalog(),
        registered=InitPassThroughEndpointHelpers.is_registered_pass_through_route(route),
        registered_auth=registered.get("auth") is True if registered is not None else None,
        configured_eu_jev=InitPassThroughEndpointHelpers.has_configured_route_ignoring_trailing_slash(
            "/eu_jev/v1/systemone"
        ),
    )


def passthrough_facts(
    route: str, token: UserAPIKeyAuth, jwt_team_routes: Collection[str] = (), *, include_denied: bool = False
) -> routes.PassThroughFacts:
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    metadata: Final = _TokenFields.model_validate(token, from_attributes=True).metadata or {}
    return routes.PassThroughFacts(
        key_allowed=_STRINGS.validate_python(metadata.get("allowed_passthrough_routes") or ()),
        team_allowed=_STRINGS.validate_python((token.team_metadata or {}).get("allowed_passthrough_routes") or ()),
        denied=(
            *_STRINGS.validate_python(metadata.get("denied_passthrough_routes") or ()),
            *_STRINGS.validate_python((token.team_metadata or {}).get("denied_passthrough_routes") or ()),
        )
        if include_denied
        else (),
        forwarded=InitPassThroughEndpointHelpers.forwarded_routes(route) if include_denied else (),
        jwt_team_routes=tuple(jwt_team_routes),
    )


def raise_route_denial(
    decision: routes.RouteDecision,
    route: str,
    *,
    token: UserAPIKeyAuth | None = None,
    role: str | None = None,
    user: LiteLLM_UserTable | None = None,
    query_user_id: str | None = None,
) -> None:
    from litellm.proxy.auth.route_checks import RouteChecks

    match decision.denial:
        case None:
            return
        case "AdminOnly":
            RouteChecks._raise_admin_only_route_exception(user_obj=user, route=route)  # pyright: ignore[reportPrivateUsage]  # preserve the public exception envelope
        case "PassThrough":
            raise RouteChecks._auth_pass_through_denied_exception(route)  # pyright: ignore[reportPrivateUsage]  # preserve the public exception envelope
        case "DeniedPassThrough":
            raise RouteChecks.passthrough_route_denied_exception(route, decision.detail or "")
        case "VirtualRoute":
            raise HTTPException(
                403,
                f"Virtual key is not allowed to call this route. Only allowed to call routes: {_TokenFields.model_validate(token, from_attributes=True).allowed_routes if token is not None else None}. Tried to call route: {route}",
            )
        case "PasswordReset":
            raise HTTPException(
                403,
                "This account's password must be changed before the session can be used: "
                "it was either found in a known data breach or set by an admin. "
                "Change it via POST /user/password/change (UI: /ui/change-password), then log in again.",
            )
        case "CustomAdmin":
            raise HTTPException(403, f"user not allowed to access this route. Route={route} is an admin only route")
        case "UserInfo":
            raise HTTPException(
                403,
                f"key not allowed to access this user's info. user_id={query_user_id}, key's user_id={token.user_id if token is not None else None}",
            )
        case "ViewerInference":
            raise HTTPException(403, f"user not allowed to access this OpenAI routes, role= {role}")
        case "ViewerWrite":
            raise HTTPException(403, f"user not allowed to access this route, role= {role}. Trying to access: {route}")
        case "ViewerParameter":
            raise HTTPException(
                403,
                f"user not allowed to access this route, role= {role}. Trying to access: {route} and updating invalid param: {decision.detail}. only user_email can be updated",
            )


def read_virtual_route_facts(route: str, token: UserAPIKeyAuth, request: Request | None) -> routes.VirtualRouteFacts:
    from litellm.proxy.auth.route_checks import RouteChecks

    fields: Final = _TokenFields.model_validate(token, from_attributes=True)
    return routes.VirtualRouteFacts(
        context=route_context(route, RouteChecks.get_request_method(request), registration=True),
        allowed_routes=_allowed_routes(fields.allowed_routes),
        passthrough=passthrough_facts(route, token),
        password_reset_required=(fields.metadata or {}).get("password_reset_required") is True,
    )


def check_virtual_route(route: str, token: UserAPIKeyAuth, request: Request | None) -> bool:
    raise_route_denial(routes.virtual_route(read_virtual_route_facts(route, token, request)), route, token=token)
    return True


def check_viewer_route(route: str, role: str, request_data: Mapping[str, object], request: Request | None) -> None:
    from litellm.proxy.auth.route_checks import RouteChecks

    context: Final = route_context(route, RouteChecks.get_request_method(request), registration=True)
    facts: Final = routes.ViewerFacts(
        replace(context, method="") if request is not None and context.method is None else context, tuple(request_data)
    )
    raise_route_denial(routes.viewer_route(facts), route, role=role)


def check_non_admin_route(
    user: LiteLLM_UserTable | None,
    role: str | None,
    route: str,
    request: Request,
    token: UserAPIKeyAuth,
    request_data: Mapping[str, object],
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.route_checks import RouteChecks

    proxy: Final = _ProxyFields.model_validate(vars(proxy_server))
    fields: Final = _TokenFields.model_validate(token, from_attributes=True)
    singular: Final = request_data.get("organization_id")
    plural: Final = request_data.get("organizations")
    organization_ids: Final = (
        *((singular,) if isinstance(singular, str) else ()),
        *(_STRINGS.validate_python(plural) if isinstance(plural, list) else ()),
    )
    context: Final = route_context(route, RouteChecks.get_request_method(request), registration=True)
    facts: Final = routes.NonAdminFacts(
        viewer=routes.ViewerFacts(
            replace(context, method="") if context.method is None else context, tuple(request_data)
        ),
        role=role,
        custom_admin_routes=_STRINGS.validate_python(proxy.general_settings.get("admin_only_routes") or ()),
        premium=proxy.premium_user is True,
        query_user_id=request.query_params.get("user_id") if route == "/user/info" else None,
        token_user_id=token.user_id,
        permissions=tuple(fields.permissions or ()),
        organization_ids=organization_ids,
        admin_organization_ids=tuple(
            membership.organization_id
            for membership in (user.organization_memberships or ())
            if membership.user_role == "org_admin"
        )
        if user is not None
        else (),
        team_service_account=bool(token.is_team_service_account),
        allowed_routes=_allowed_routes(fields.allowed_routes),
        passthrough=passthrough_facts(route, token, RouteChecks._jwt_team_allowed_routes(token), include_denied=True),  # pyright: ignore[reportPrivateUsage]  # project legacy JWT team route metadata
    )
    raise_route_denial(
        routes.non_admin_route(facts), route, token=token, role=role, user=user, query_user_id=facts.query_user_id
    )


def classify(
    route: str,
    kind: Literal["Llm", "Unbilled", "Discovery", "SelfCredential", "Azure", "Managed"],
    method: str | None = None,
) -> bool:
    from litellm._auth.route_enable import llm_route

    context: Final = route_context(route, method)
    if kind != "Llm":
        return routes.classify_route(context, kind)
    decision: Final = llm_route(context)
    if decision == "ReadEuJev":
        return llm_route(context, read_eu_jev_configuration()) == "Allow"
    return decision == "Allow"


def read_eu_jev_configuration() -> bool:
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    return InitPassThroughEndpointHelpers.has_configured_route_ignoring_trailing_slash("/eu_jev/v1/systemone")


def denied_passthrough(route: str, metadata_sources: Iterable[Mapping[str, object] | None]) -> str | None:
    from itertools import chain

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import InitPassThroughEndpointHelpers

    facts: Final = routes.PassThroughFacts(
        denied=tuple(
            chain.from_iterable(
                _STRINGS.validate_python((source or {}).get("denied_passthrough_routes") or ())
                for source in metadata_sources
            )
        ),
        forwarded=InitPassThroughEndpointHelpers.forwarded_routes(route),
    )
    return routes.denied_passthrough_route(route, facts).detail
