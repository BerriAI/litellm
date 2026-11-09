from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException, Request

from litellm._auth import routes
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.route_policy_adapter import route_context
from tests.unit._auth.support.route_reference import RouteChecks as ReferenceRoutes
from tests.unit._auth.support.route_reference import managed_agent_route_allowed


@dataclass(frozen=True, slots=True)
class RouteResult:
    allowed: bool
    status: int | None = None
    detail: object = None


def outcome(call: Callable[[], object]) -> RouteResult:
    try:
        result: Final = call()
        return RouteResult(result is not False)
    except HTTPException as error:
        return RouteResult(False, error.status_code, error.detail)


_ROUTES: Final = tuple(
    dict.fromkeys(
        (
            "/unknown",
            "/chat/completions",
            "/v1/mcp/server",
            "/v1/mcp/server/id",
            "/v1/mcp/server/id/more",
            "/auto_router/session",
            "/models",
            "/v1/responses/id",
            "/agents",
            "/v1/agents/id",
            "/key/id/regenerate",
            "/user/update",
            "/user/password/change",
            "/session/logout",
            "/user/info",
            "/organization/new",
            "/team/new",
            "/openai/deployments/model/versions/chat/completions",
            "/engines/model/chat/completions",
            "/foo/count_tokens",
            "/v1beta/models/model:countTokens",
            "/unknown\n",
            "/key/generate",
            "/key/list",
            "/session/list",
        )
    )
)


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("method", (None, "GET", "POST", "DELETE", "HEAD", "OPTIONS"))
def test_route_classifications_match_reference(route: str, method: str | None) -> None:
    request: Final = Request({"type": "http", "method": method or "GET", "path": route, "headers": []})
    assert RouteChecks.is_llm_api_route(route) == ReferenceRoutes.is_llm_api_route(route)
    assert RouteChecks.is_unbilled_route(route) == ReferenceRoutes.is_unbilled_route(route)
    assert RouteChecks.is_self_scoped_credential_route(
        route, request
    ) == ReferenceRoutes.is_self_scoped_credential_route(route, request)
    assert RouteChecks._is_get_mcp_server_discovery_route(  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection
        route, request
    ) == ReferenceRoutes._is_get_mcp_server_discovery_route(route, request)  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "allowed",
    (
        None,
        (),
        ("llm_api_routes",),
        ("llm_api_routes_without_mcp",),
        ("management_routes",),
        ("/v1/*",),
        ("/user",),
        ("*",),
        ("/no",),
    ),
)
@pytest.mark.parametrize("method", ("GET", "POST"))
def test_virtual_route_allow_and_denial_envelope(route: str, allowed: tuple[str, ...] | None, method: str) -> None:
    request: Final = Request({"type": "http", "method": method, "path": route, "headers": []})
    token: Final = UserAPIKeyAuth(allowed_routes=list(allowed) if allowed is not None else None)
    assert outcome(lambda: RouteChecks.is_virtual_key_allowed_to_call_route(route, token, request)) == outcome(
        lambda: ReferenceRoutes.is_virtual_key_allowed_to_call_route(route, token, request)
    )


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("method", ("GET", "POST", "PATCH", "HEAD"))
@pytest.mark.parametrize("keys", ((), ("user_email",), ("user_id", "user_email")))
def test_viewer_role_denial_order_and_envelope(route: str, method: str, keys: tuple[str, ...]) -> None:
    request: Final = Request({"type": "http", "method": method, "path": route, "headers": []})
    data: Final = {key: "value" for key in keys}
    assert outcome(
        lambda: RouteChecks._check_proxy_admin_viewer_access(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # compare the preserved private route behavior against native projection
            route=route, _user_role="proxy_admin_viewer", request_data=data, request=request
        )
    ) == outcome(
        lambda: ReferenceRoutes._check_proxy_admin_viewer_access(  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection
            route=route, _user_role="proxy_admin_viewer", request_data=data, request=request
        )
    )


@pytest.mark.parametrize("group", tuple(LiteLLMRoutes))
def test_full_catalog_classification(group: LiteLLMRoutes) -> None:
    assert tuple(RouteChecks.is_llm_api_route(route) for route in group.value) == tuple(
        ReferenceRoutes.is_llm_api_route(route) for route in group.value
    )


@pytest.mark.parametrize(
    ("route", "pattern"),
    (
        ("/a/id", "/a/{id}"),
        ("/a/x/y", "/a/{id:path}"),
        ("/a/x/y:action", "/a/{id:path}:action"),
        ("/a/x\ny", "/a/{id:path}"),
        ("/a/id\n", "/a/{id}"),
        ("/a/b", "/a/*"),
        ("/ab", "/a"),
        ("/a", ""),
        ("/a", "/"),
        ("/a/id:action", "/a/{id}:action"),
        ("/a/id", "/a/{id:path}"),
    ),
)
def test_route_pattern_modes(route: str, pattern: str) -> None:
    assert RouteChecks._route_matches_pattern(route, pattern) == ReferenceRoutes._route_matches_pattern(route, pattern)  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection
    assert RouteChecks._route_matches_allowed_route(route, pattern) == ReferenceRoutes._route_matches_allowed_route(  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection
        route, pattern
    )
    assert RouteChecks.route_matches_wildcard_pattern(route, pattern) == ReferenceRoutes.route_matches_wildcard_pattern(
        route, pattern
    )
    assert RouteChecks._route_matches_denied_route(route, pattern) == ReferenceRoutes._route_matches_denied_route(  # pyright: ignore[reportPrivateUsage]  # compare the preserved private route behavior against native projection
        route, pattern
    )


@pytest.mark.parametrize(
    "route",
    (
        "/agents",
        "/v1/agents",
        "/chat/completions",
        "/v1/chat/completions",
        "/v1/responses/id",
        "/realtime",
        "/v1/realtime",
        "/v1/mcp/server/id",
        "/mcp/foo",
        "/user/info",
        "/v1beta/models/x:generateContent",
    ),
)
@pytest.mark.parametrize("method", (None, "GET", "HEAD", "POST", "DELETE", "get"))
def test_managed_actor_route_methods(route: str, method: str | None) -> None:
    assert routes.classify_route(route_context(route, method), "Managed") == managed_agent_route_allowed(route, method)


@pytest.mark.parametrize(
    ("key", "team", "jwt", "denied", "expected"),
    (
        (("/p",), (), (), (), True),
        ((), ("/p",), (), (), True),
        (("/q",), ("/p",), (), (), False),
        ((), (), ("/p/*",), (), True),
        ((), (), ("*",), (), False),
        (("/p",), (), (), ("/p",), False),
    ),
)
def test_passthrough_precedence(
    key: tuple[str, ...], team: tuple[str, ...], jwt: tuple[str, ...], denied: tuple[str, ...], expected: bool
) -> None:
    facts: Final = routes.PassThroughFacts(key, team, denied, ("/p/resource",), jwt)
    assert routes.passthrough_route(routes.RouteContext("/p/resource"), facts, required=True).allowed == expected
