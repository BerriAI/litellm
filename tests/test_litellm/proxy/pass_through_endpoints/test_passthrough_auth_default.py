"""
Regression tests for the pass-through endpoint auth-default fix
(GHSA-7h34-mmrh-6g58).

Two failures the fix closes:

1. ``PassThroughGenericEndpoint.auth`` defaulted to ``False`` — an
   admin who added a pass-through to ``general_settings`` without
   explicitly setting ``auth: true`` shipped an unauthenticated
   forwarder.
2. Setting ``auth: true`` was rejected at startup unless the operator
   had a LiteLLM Enterprise license, leaving OSS deployments with no
   safe configuration.

The fix flips the default to ``True`` (safe-by-default) and removes
the enterprise gate so OSS operators can register an authenticated
pass-through. The runtime check in ``user_api_key_auth.py`` also now
defaults to ``True`` so a config dict (raw, not Pydantic) without an
``auth`` key still requires authentication.
"""

from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute

from litellm.proxy._types import LiteLLMRoutes, PassThroughGenericEndpoint
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import (
    check_api_key_for_custom_headers_or_pass_through_endpoints,
    user_api_key_auth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _register_pass_through_endpoint,
)
from litellm.proxy.utils import ProxyLogging


def test_passthrough_auth_defaults_to_true():
    # Regression: an admin who configures a pass-through without setting
    # auth explicitly used to ship an unauthenticated forwarder. The
    # default is now safe.
    endpoint = PassThroughGenericEndpoint(
        path="/canary-forwarder",
        target="https://postman-echo.com/get",
    )
    assert endpoint.auth is True


def test_passthrough_auth_can_still_be_explicitly_disabled():
    # Operators who genuinely need an unauthenticated forwarder (e.g.
    # public webhook receiver) can opt in explicitly.
    endpoint = PassThroughGenericEndpoint(
        path="/public-webhook",
        target="https://example.com/webhook",
        auth=False,
    )
    assert endpoint.auth is False


@pytest.mark.asyncio
async def test_register_passthrough_with_auth_true_works_for_oss(monkeypatch):
    # Regression: setting ``auth: true`` used to raise at startup
    # unless ``premium_user`` was True, leaving OSS with no safe
    # configuration.
    app = MagicMock(spec=FastAPI)
    visited: set = set()

    endpoint = PassThroughGenericEndpoint(
        path="/forwarder",
        target="https://example.com",
        auth=True,
    )

    # Should not raise; OSS premium_user=False is allowed to use auth=True.
    await _register_pass_through_endpoint(
        endpoint=endpoint,
        app=app,
        premium_user=False,
        visited_endpoints=visited,
    )


@pytest.mark.asyncio
async def test_runtime_check_treats_missing_auth_key_as_authenticated():
    # The runtime dispatch in user_api_key_auth pulls
    # pass_through_endpoints from general_settings as raw dicts (the
    # Pydantic default never applies). A dict without an ``auth`` key
    # must default to "authenticated" — without this, the previous
    # behaviour (``endpoint.get("auth") is not True`` -> True -> empty
    # auth) ships an unauthenticated forwarder.
    request = MagicMock()
    request.headers = {}
    raw_endpoint_no_auth_key = {
        "path": "/forwarder",
        "target": "https://example.com",
        # ``auth`` deliberately omitted
    }

    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/forwarder",
        pass_through_endpoints=[raw_endpoint_no_auth_key],
        api_key="sk-1234",
    )

    # Result is the api_key string (auth is REQUIRED for this endpoint
    # — flow continues to normal key validation), NOT an empty
    # ``UserAPIKeyAuth()`` (which was the unauthenticated-forwarder
    # bug).
    assert result == "sk-1234"


@pytest.mark.asyncio
async def test_runtime_check_explicit_auth_false_still_skips_validation():
    # Operators who explicitly set ``auth: False`` get the legacy
    # behaviour — an empty UserAPIKeyAuth, no key required.
    from litellm.proxy._types import UserAPIKeyAuth

    request = MagicMock()
    request.headers = {}
    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/public-webhook",
        pass_through_endpoints=[raw_endpoint_auth_false],
        api_key="",
    )

    assert isinstance(result, UserAPIKeyAuth)


def _route_level_dependencies(app: FastAPI, path: str) -> list[object]:
    matching: Final = [route for route in app.routes if isinstance(route, APIRoute) and route.path == path]
    return [dep.dependency for dep in chain.from_iterable(route.dependencies for route in matching)]


def _is_proxy_only_error(proxy_log: ProxyLogging, original_exception: Exception, route: str) -> bool:
    is_proxy_only: Final = (
        proxy_log._is_proxy_only_llm_api_error  # pyright: ignore[reportPrivateUsage]  # only observable as a DB row
    )
    return is_proxy_only(original_exception=original_exception, route=route)


_CONFIG_OMITTED: Final = {
    "id": "cfg-omitted",
    "path": "/cfg-omitted-pt",
    "target": "http://upstream.invalid",
    "include_subpath": True,
}
_CONFIG_AUTH_FALSE: Final = {
    "id": "cfg-authfalse",
    "path": "/cfg-authfalse-pt",
    "target": "http://upstream.invalid",
    "include_subpath": True,
    "auth": False,
}
_DB_DEFAULT: Final = PassThroughGenericEndpoint(
    id="db-default",
    path="/db-default-pt",
    target="http://upstream.invalid",
    include_subpath=True,
)
_DB_AUTH_FALSE: Final = PassThroughGenericEndpoint(
    id="db-authfalse",
    path="/db-authfalse-pt",
    target="http://upstream.invalid",
    include_subpath=True,
    auth=False,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "endpoint_id", "path", "enforced", "route"),
    [
        pytest.param(
            _CONFIG_OMITTED,
            "cfg-omitted",
            "/cfg-omitted-pt",
            True,
            "/cfg-omitted-pt",
            id="config-omitted-exact",
        ),
        pytest.param(
            _CONFIG_OMITTED,
            "cfg-omitted",
            "/cfg-omitted-pt",
            True,
            "/cfg-omitted-pt/sub",
            id="config-omitted-sub",
        ),
        pytest.param(
            _CONFIG_AUTH_FALSE,
            "cfg-authfalse",
            "/cfg-authfalse-pt",
            False,
            "/cfg-authfalse-pt",
            id="config-authfalse-exact",
        ),
        pytest.param(
            _CONFIG_AUTH_FALSE,
            "cfg-authfalse",
            "/cfg-authfalse-pt",
            False,
            "/cfg-authfalse-pt/sub",
            id="config-authfalse-sub",
        ),
        pytest.param(_DB_DEFAULT, "db-default", "/db-default-pt", True, "/db-default-pt", id="db-default-exact"),
        pytest.param(_DB_DEFAULT, "db-default", "/db-default-pt", True, "/db-default-pt/sub", id="db-default-sub"),
        pytest.param(
            _DB_AUTH_FALSE,
            "db-authfalse",
            "/db-authfalse-pt",
            False,
            "/db-authfalse-pt",
            id="db-authfalse-exact",
        ),
        pytest.param(
            _DB_AUTH_FALSE,
            "db-authfalse",
            "/db-authfalse-pt",
            False,
            "/db-authfalse-pt/sub",
            id="db-authfalse-sub",
        ),
    ],
)
async def test_pass_through_registration_auth_enforcement(
    endpoint: dict[str, object] | PassThroughGenericEndpoint,
    endpoint_id: str,
    path: str,
    enforced: bool,
    route: str,
):
    app: Final = FastAPI()
    try:
        await _register_pass_through_endpoint(
            endpoint=endpoint,
            app=app,
            premium_user=False,
            visited_endpoints=set(),
        )
        upstream_error: Final = HTTPException(
            status_code=403,
            detail="Upstream passthrough request failed with status 403",
        )
        proxy_logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
        exact_dependencies: Final = _route_level_dependencies(app, path)
        subpath_dependencies: Final = _route_level_dependencies(app, f"{path}/{{subpath:path}}")

        assert (path in LiteLLMRoutes.openai_routes.value) == enforced
        assert (f"{path}/*" in LiteLLMRoutes.openai_routes.value) == enforced
        assert RouteChecks.is_llm_api_route(route=route) == enforced
        assert RouteChecks.is_auth_enforced_pass_through_route(route=route, method="POST") == enforced
        assert _is_proxy_only_error(proxy_logging, upstream_error, route) == enforced
        assert (user_api_key_auth in exact_dependencies) == enforced
        assert (user_api_key_auth in subpath_dependencies) == enforced
    finally:
        InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_id)
