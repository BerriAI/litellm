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

from collections.abc import Iterator
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
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
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


@pytest.fixture
def pass_through_route_globals() -> Iterator[None]:
    openai_routes = list(LiteLLMRoutes.openai_routes.value)
    registered = dict(
        pass_through_endpoints._registered_pass_through_routes  # pyright: ignore[reportPrivateUsage]  # registry under test
    )
    yield
    LiteLLMRoutes.openai_routes.value[:] = openai_routes
    pass_through_endpoints._registered_pass_through_routes.clear()  # pyright: ignore[reportPrivateUsage]  # registry under test
    pass_through_endpoints._registered_pass_through_routes.update(  # pyright: ignore[reportPrivateUsage]  # registry under test
        registered
    )


def _registered_app_route_calls(app: FastAPI, *paths: str) -> list[object]:
    return [
        dep.dependency
        for app_route in app.routes
        if isinstance(app_route, APIRoute) and app_route.path in paths
        for dep in app_route.dependencies
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/omitted-auth-pt", "/omitted-auth-pt/sub"])
async def test_config_dict_without_auth_registers_auth_enforced_route(route: str, pass_through_route_globals: None):
    app = FastAPI()
    await _register_pass_through_endpoint(
        endpoint={
            "path": "/omitted-auth-pt",
            "target": "http://upstream.invalid",
            "include_subpath": True,
        },
        app=app,
        premium_user=False,
        visited_endpoints=set(),
    )

    assert "/omitted-auth-pt" in LiteLLMRoutes.openai_routes.value
    assert "/omitted-auth-pt/*" in LiteLLMRoutes.openai_routes.value
    assert RouteChecks.is_llm_api_route(route=route)
    assert RouteChecks.is_auth_enforced_pass_through_route(route=route, method="POST")
    upstream_error = HTTPException(
        status_code=403,
        detail="Upstream passthrough request failed with status 403",
    )
    assert ProxyLogging(user_api_key_cache=UserApiKeyCache())._is_proxy_only_llm_api_error(  # pyright: ignore[reportPrivateUsage]  # asserts the failure-spend gate
        original_exception=upstream_error,
        route=route,
    )
    assert user_api_key_auth in _registered_app_route_calls(app, "/omitted-auth-pt", "/omitted-auth-pt/{subpath:path}")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/omitted-auth-pt", "/omitted-auth-pt/sub"])
async def test_config_dict_with_auth_false_registers_unenforced_route(route: str, pass_through_route_globals: None):
    app = FastAPI()
    await _register_pass_through_endpoint(
        endpoint={
            "path": "/omitted-auth-pt",
            "target": "http://upstream.invalid",
            "include_subpath": True,
            "auth": False,
        },
        app=app,
        premium_user=False,
        visited_endpoints=set(),
    )

    assert "/omitted-auth-pt" not in LiteLLMRoutes.openai_routes.value
    assert "/omitted-auth-pt/*" not in LiteLLMRoutes.openai_routes.value
    assert not RouteChecks.is_llm_api_route(route=route)
    assert not RouteChecks.is_auth_enforced_pass_through_route(route=route, method="POST")
    upstream_error = HTTPException(
        status_code=403,
        detail="Upstream passthrough request failed with status 403",
    )
    assert not ProxyLogging(user_api_key_cache=UserApiKeyCache())._is_proxy_only_llm_api_error(  # pyright: ignore[reportPrivateUsage]  # asserts the failure-spend gate
        original_exception=upstream_error,
        route=route,
    )
    assert user_api_key_auth not in _registered_app_route_calls(
        app, "/omitted-auth-pt", "/omitted-auth-pt/{subpath:path}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/db-default-pt", "/db-default-pt/sub"])
async def test_db_endpoint_default_auth_registers_auth_enforced_route(route: str, pass_through_route_globals: None):
    app = FastAPI()
    await _register_pass_through_endpoint(
        endpoint=PassThroughGenericEndpoint(
            path="/db-default-pt",
            target="http://upstream.invalid",
            include_subpath=True,
        ),
        app=app,
        premium_user=False,
        visited_endpoints=set(),
    )

    assert "/db-default-pt" in LiteLLMRoutes.openai_routes.value
    assert "/db-default-pt/*" in LiteLLMRoutes.openai_routes.value
    assert RouteChecks.is_llm_api_route(route=route)
    assert RouteChecks.is_auth_enforced_pass_through_route(route=route, method="POST")
    upstream_error = HTTPException(
        status_code=403,
        detail="Upstream passthrough request failed with status 403",
    )
    assert ProxyLogging(user_api_key_cache=UserApiKeyCache())._is_proxy_only_llm_api_error(  # pyright: ignore[reportPrivateUsage]  # asserts the failure-spend gate
        original_exception=upstream_error,
        route=route,
    )
    assert user_api_key_auth in _registered_app_route_calls(app, "/db-default-pt", "/db-default-pt/{subpath:path}")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/db-authfalse-pt", "/db-authfalse-pt/sub"])
async def test_db_endpoint_auth_false_registers_unenforced_route(route: str, pass_through_route_globals: None):
    app = FastAPI()
    await _register_pass_through_endpoint(
        endpoint=PassThroughGenericEndpoint(
            path="/db-authfalse-pt",
            target="http://upstream.invalid",
            include_subpath=True,
            auth=False,
        ),
        app=app,
        premium_user=False,
        visited_endpoints=set(),
    )

    assert "/db-authfalse-pt" not in LiteLLMRoutes.openai_routes.value
    assert "/db-authfalse-pt/*" not in LiteLLMRoutes.openai_routes.value
    assert not RouteChecks.is_llm_api_route(route=route)
    assert not RouteChecks.is_auth_enforced_pass_through_route(route=route, method="POST")
    upstream_error = HTTPException(
        status_code=403,
        detail="Upstream passthrough request failed with status 403",
    )
    assert not ProxyLogging(user_api_key_cache=UserApiKeyCache())._is_proxy_only_llm_api_error(  # pyright: ignore[reportPrivateUsage]  # asserts the failure-spend gate
        original_exception=upstream_error,
        route=route,
    )
    assert user_api_key_auth not in _registered_app_route_calls(
        app, "/db-authfalse-pt", "/db-authfalse-pt/{subpath:path}"
    )
