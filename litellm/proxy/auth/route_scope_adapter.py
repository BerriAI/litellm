from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Final, assert_never

from fastapi import HTTPException
from pydantic import BaseModel, TypeAdapter

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.route_scope import RouteScopeFacts, RouteScopeKind, RouteValues, route_scope
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors

_OBJECTS: Final = TypeAdapter(Iterable[object])


class _Settings(BaseModel, frozen=True):
    general_settings: Mapping[str, object]
    premium_user: object


def route_values(value: object) -> RouteValues:
    if isinstance(value, str):
        return RouteValues(text=value)
    if isinstance(value, (bytes, bytearray)):
        return RouteValues(failure="a bytes-like object is required, not 'str'")
    if isinstance(value, Iterable):
        return RouteValues(names=tuple(item for item in _OBJECTS.validate_python(value) if isinstance(item, str)))
    return RouteValues(failure=f"argument of type '{type(value).__name__}' is not iterable")


def read_route_scope_facts(route: str, kind: RouteScopeKind) -> RouteScopeFacts:
    from litellm.proxy import proxy_server

    settings: Final = _Settings.model_validate(vars(proxy_server))
    setting: Final = "allowed_routes" if kind == "Proxy" else "admin_only_routes"
    initial: Final = RouteScopeFacts(kind, route, setting in settings.general_settings, settings.premium_user is True)

    def advance(facts: RouteScopeFacts) -> RouteScopeFacts:
        decision: Final = route_scope(facts)
        match decision.kind:
            case "Allow" | "Deny":
                return facts
            case "Warn":
                verbose_proxy_logger.error(
                    "Trying to set allowed_routes. This is an Enterprise feature. %s"
                    if kind == "Proxy"
                    else "Trying to use 'admin_only_routes' this is an Enterprise only feature. %s",
                    CommonProxyErrors.not_premium_user.value,
                )
                return advance(replace(facts, warned=True))
            case "ReadValues":
                return advance(replace(facts, values=route_values(settings.general_settings[setting])))
            case "Failure":
                raise TypeError(decision.message)
            case _:
                assert_never(decision.kind)

    return advance(initial)


def is_allowed_on_proxy(route: str) -> bool:
    return route_scope(read_route_scope_facts(route, "Proxy")).kind == "Allow"


def custom_admin_only_route_check(route: str) -> None:
    decision: Final = route_scope(read_route_scope_facts(route, "Admin"))
    match decision.kind:
        case "Allow":
            return
        case "Deny":
            raise HTTPException(403, f"user not allowed to access this route. Route={route} is an admin only route")
        case _:
            raise NativeAuthRuntimeError("incomplete custom admin route decision")
