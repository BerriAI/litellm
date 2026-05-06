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

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import PassThroughGenericEndpoint
from litellm.proxy.auth.user_api_key_auth import (
    check_api_key_for_custom_headers_or_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    SafeRouteAdder,
    _register_pass_through_endpoint,
)


@pytest.fixture(autouse=True)
def reset_passthrough_route_state(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_server_root_path",
        lambda: "/",
    )
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    yield
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


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
    request.method = "POST"
    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    app = FastAPI()
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    try:
        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/public-webhook",
            target="https://example.com",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="public-webhook",
            methods=["POST"],
        )

        result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=request,
            route="/public-webhook",
            pass_through_endpoints=[raw_endpoint_auth_false],
            api_key="",
        )
    finally:
        InitPassThroughEndpointHelpers.clear_all_pass_through_routes()

    assert isinstance(result, UserAPIKeyAuth)


def test_colliding_passthrough_route_does_not_register_metadata():
    app = FastAPI()

    @app.post("/customer/block")
    async def existing_management_route():
        return {"ok": True}

    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    try:
        route_registered = InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/customer/block",
            target="https://example.com",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="colliding-forwarder",
            methods=["POST"],
        )

        assert route_registered is False
        assert (
            InitPassThroughEndpointHelpers.get_registered_pass_through_route(
                route="/customer/block", method="POST"
            )
            is None
        )
    finally:
        InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


@pytest.mark.asyncio
async def test_unregistered_auth_false_passthrough_does_not_skip_validation():
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    raw_endpoint_auth_false = {
        "path": "/customer/block",
        "target": "https://example.com",
        "auth": False,
    }

    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/customer/block",
        pass_through_endpoints=[raw_endpoint_auth_false],
        api_key="sk-1234",
    )

    assert result == "sk-1234"


def test_existing_passthrough_route_metadata_can_be_updated():
    app = FastAPI()
    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    try:
        first_registration = InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/forwarder",
            target="https://example.com/old",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="forwarder",
            methods=["POST"],
        )
        second_registration = InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/forwarder",
            target="https://example.com/new",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="forwarder",
            methods=["POST"],
        )

        route_info = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            route="/forwarder", method="POST"
        )

        assert first_registration is True
        assert second_registration is True
        assert route_info is not None
        assert route_info["passthrough_params"]["target"] == "https://example.com/new"
    finally:
        InitPassThroughEndpointHelpers.clear_all_pass_through_routes()


def test_passthrough_route_detection_is_method_aware_for_split_paths():
    app = FastAPI()

    @app.get("/split-route")
    async def existing_get_route():
        return {"ok": True}

    InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
    try:
        route_registered = InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/split-route",
            target="https://example.com",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="split-route-forwarder",
            methods=["POST"],
        )

        assert route_registered is True
        assert (
            SafeRouteAdder.is_registered_route_pass_through(
                app=app, path="/split-route", methods=["POST"]
            )
            is True
        )
        assert (
            SafeRouteAdder.is_registered_route_pass_through(
                app=app, path="/split-route", methods=["GET", "POST"]
            )
            is False
        )
    finally:
        InitPassThroughEndpointHelpers.clear_all_pass_through_routes()
