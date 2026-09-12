"""Starlette matches routes in registration order, so the routes that take the most traffic go first."""

from collections.abc import Sequence
from typing import Final

from starlette.routing import BaseRoute, Route

HOT_ROUTE_PATHS: Final[frozenset[str]] = frozenset(
    (
        "/health/liveliness",
        "/health/liveness",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
    )
)


def _is_hot(route: BaseRoute) -> bool:
    return isinstance(route, Route) and route.path in HOT_ROUTE_PATHS


def hot_routes_first(routes: Sequence[BaseRoute]) -> list[BaseRoute]:  # mutable-ok: assigned to Router.routes, a list
    return sorted(routes, key=lambda route: not _is_hot(route))
