from collections.abc import Iterable
from itertools import chain
from typing import Final

from fastapi import Request
from starlette.routing import compile_path

from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_UserTable, LiteLLMRoutes, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks


def effective_user_role(user_role: str | None) -> LitellmUserRoles:
    try:
        return LitellmUserRoles(user_role)
    except ValueError:
        return LitellmUserRoles.INTERNAL_USER


def get_user_role(user_obj: LiteLLM_UserTable | None) -> LitellmUserRoles | None:
    return None if user_obj is None else effective_user_role(user_obj.user_role)


def is_user_proxy_admin(user_obj: LiteLLM_UserTable | None) -> bool:
    return user_obj is not None and user_obj.user_role is not None and user_obj.user_role == "proxy_admin"


def is_api_route_allowed(
    route: str, request: Request, request_data: dict[str, object], valid_token: UserAPIKeyAuth | None,  # mutable-ok: preserve the original request identity at the shared route effect
    user_obj: LiteLLM_UserTable | None = None,
) -> bool:
    role: Final = get_user_role(user_obj)
    if valid_token is None:
        raise Exception("Invalid proxy server token passed. valid_token=None.")
    if not is_user_proxy_admin(user_obj):
        RouteChecks.non_proxy_admin_allowed_routes_check(  # pyright: ignore[reportUnknownMemberType]  # Phase 1 public effect signature
            user_obj=user_obj, _user_role=role, route=route, request=request,
            request_data=request_data, valid_token=valid_token,
        )
    return True


def allowed_routes_list_check(user_route: str, allowed_routes: Iterable[object]) -> bool:
    for allowed in allowed_routes:
        if allowed in LiteLLMRoutes.__members__:
            for template in LiteLLMRoutes[allowed].value:  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportUnknownMemberType]  # reference intentionally preserves the untyped Phase 1 input contract
                regex, _, _ = compile_path(template)  # pyright: ignore[reportUnknownArgumentType]  # enum values are legacy route collections
                if regex.match(user_route):
                    return True
        elif allowed.endswith("*"):  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # invalid legacy entries must raise their original AttributeError
            if user_route.startswith(allowed[:-1]):  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]  # preserve the reference's raw entry operations
                return True
        elif user_route == allowed:
            return True
    return False


def allowed_routes_check(user_role: LitellmUserRoles, user_route: str, litellm_proxy_roles: LiteLLM_JWTAuth) -> bool:
    if user_role == LitellmUserRoles.PROXY_ADMIN:
        return allowed_routes_list_check(user_route, litellm_proxy_roles.admin_allowed_routes)
    if user_role == LitellmUserRoles.TEAM:
        configured: Final = litellm_proxy_roles.team_allowed_routes
        return allowed_routes_list_check(
            user_route, ("openai_routes", "info_routes") if configured is None else configured  # pyright: ignore[reportUnnecessaryComparison]  # retain Phase 1 constructed None config behavior
        )
    return False


def allowed_route_check_inside_route(auth: UserAPIKeyAuth, requested_user_id: str | None) -> bool:
    return auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY) or (
        requested_user_id is not None and auth.user_id is not None and auth.user_id == requested_user_id
    )


def get_actual_routes(allowed_routes: Iterable[object]) -> list[object]:  # mutable-ok: public Phase 1 projection
    def expanded(raw: object) -> tuple[object, ...]:
        try:
            return tuple(LiteLLMRoutes[raw].value)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType, reportUnknownMemberType]  # preserve enum indexing and iterable values from Phase 1
        except KeyError:
            return (raw,)

    return list(chain.from_iterable(expanded(raw) for raw in allowed_routes))
