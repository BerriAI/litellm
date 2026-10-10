"""Starlette matches routes in registration order, so the routes that take the most traffic go first."""

import os
import re
from collections.abc import Collection, Sequence
from itertools import chain
from typing import Final

from starlette.routing import BaseRoute, Route

from litellm.types.passthrough_endpoints.pass_through_endpoints import LITELLM_PASS_THROUGH_ENDPOINT_MARKER

HOT_ROUTE_PATHS: Final[frozenset[str]] = frozenset(
    (
        "/health/liveliness",
        "/health/liveness",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
    )
)


def builtin_pass_through_routes_first() -> bool:
    return os.getenv("LITELLM_BUILTIN_PASS_THROUGH_ROUTES_FIRST", "false").lower() == "true"


def _is_hot(route: BaseRoute) -> bool:
    return isinstance(route, Route) and route.path in HOT_ROUTE_PATHS


def hot_routes_first(routes: Sequence[BaseRoute]) -> list[BaseRoute]:  # mutable-ok: assigned to Router.routes, a list
    return sorted(routes, key=lambda route: not _is_hot(route))


BUILTIN_PREFIX_CLAIM: Final[re.Pattern[str]] = re.compile(r"^(/[^{}]+/)\{[A-Za-z_][A-Za-z0-9_]*:path\}$")


def _claim_prefix(route: BaseRoute) -> str | None:
    match: Final = BUILTIN_PREFIX_CLAIM.match(getattr(route, "path", ""))
    return match.group(1) if match else None


def _prefix_candidates(path: str) -> tuple[str, ...]:
    return tuple(path[: k + 1] for k in range(1, len(path)) if path[k] == "/")


def shadowed_by_builtin_claim(path: str, builtin_routes: Collection[BaseRoute]) -> bool:
    candidates: Final = frozenset(_prefix_candidates(path))
    return any((prefix := _claim_prefix(route)) is not None and prefix in candidates for route in builtin_routes)


def configured_pass_through_routes_first(
    routes: Sequence[BaseRoute], builtin_routes: Collection[BaseRoute]
) -> list[BaseRoute]:  # mutable-ok: assigned to Router.routes, a list
    builtin_ids: Final = frozenset(id(route) for route in builtin_routes)

    def is_configured(route: BaseRoute) -> bool:
        return isinstance(route, Route) and getattr(route.endpoint, LITELLM_PASS_THROUGH_ENDPOINT_MARKER, False) is True

    def claim_prefix(index: int) -> str | None:
        if id(routes[index]) not in builtin_ids:
            return None
        return _claim_prefix(routes[index])

    claimed: Final = tuple(
        (index, prefix) for index in range(len(routes) - 1, -1, -1) if (prefix := claim_prefix(index)) is not None
    )
    earliest_claim: Final = {prefix: index for index, prefix in claimed}

    def hoist_target(index: int) -> int | None:
        configured_path: Final = getattr(routes[index], "path", "")
        hits: Final = tuple(
            earliest_claim[candidate]
            for candidate in _prefix_candidates(configured_path)
            if candidate in earliest_claim
        )
        target: Final = min(hits) if hits else len(routes)
        return target if target < index else None

    targets: Final = {index: hoist_target(index) for index in range(len(routes)) if is_configured(routes[index])}
    moved: Final = frozenset(index for index, target in targets.items() if target is not None)
    return list(
        chain.from_iterable(
            (
                *tuple(routes[j] for j in targets if targets[j] == index),
                *(() if index in moved else (route,)),
            )
            for index, route in enumerate(routes)
        )
    )
