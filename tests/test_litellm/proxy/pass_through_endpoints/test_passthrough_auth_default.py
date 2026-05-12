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
    _register_pass_through_endpoint,
)


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
async def test_runtime_check_treats_missing_auth_key_as_authenticated(monkeypatch):
    # The runtime dispatch in user_api_key_auth pulls
    # pass_through_endpoints from general_settings as raw dicts (the
    # Pydantic default never applies). A dict without an ``auth`` key
    # must default to "authenticated" — without this, the previous
    # behaviour (``endpoint.get("auth") is not True`` -> True -> empty
    # auth) ships an unauthenticated forwarder.
    # SERVER_ROOT_PATH is set as a module-level side-effect in another
    # proxy test file; strip it so the path lookup matches the raw route.
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    raw_endpoint_no_auth_key = {
        "path": "/forwarder",
        "target": "https://example.com",
        # ``auth`` deliberately omitted
    }

    # The auth helper now consults a method-aware registry; register
    # the forwarder so the gate is satisfied for /forwarder POST.
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _registered_pass_through_routes,
    )

    _registered_pass_through_routes.clear()
    _registered_pass_through_routes["test-fwd:exact:/forwarder:POST"] = {
        "endpoint_id": "test-fwd",
        "path": "/forwarder",
        "type": "exact",
        "methods": ["POST"],
        "passthrough_params": {},
    }

    try:
        result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=request,
            route="/forwarder",
            pass_through_endpoints=[raw_endpoint_no_auth_key],
            api_key="sk-1234",
        )
    finally:
        _registered_pass_through_routes.clear()

    # Result is the api_key string (auth is REQUIRED for this endpoint
    # — flow continues to normal key validation), NOT an empty
    # ``UserAPIKeyAuth()`` (which was the unauthenticated-forwarder
    # bug).
    assert result == "sk-1234"


@pytest.mark.asyncio
async def test_runtime_check_explicit_auth_false_still_skips_validation(monkeypatch):
    # Operators who explicitly set ``auth: False`` get the legacy
    # behaviour — an empty UserAPIKeyAuth, no key required.
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _registered_pass_through_routes,
    )

    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    request = MagicMock()
    request.headers = {}
    request.method = "POST"
    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    # Mirror what ``_register_pass_through_endpoint`` does for a
    # non-colliding path. The auth helper now consults this registry
    # so a colliding config entry (registry-absent) does not silently
    # bypass auth on a built-in handler it would shadow.
    _registered_pass_through_routes.clear()
    _registered_pass_through_routes["test-webhook:exact:/public-webhook:POST"] = {
        "endpoint_id": "test-webhook",
        "path": "/public-webhook",
        "type": "exact",
        "methods": ["POST"],
        "passthrough_params": {},
    }

    try:
        result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=request,
            route="/public-webhook",
            pass_through_endpoints=[raw_endpoint_auth_false],
            api_key="",
        )

        assert isinstance(result, UserAPIKeyAuth)
    finally:
        _registered_pass_through_routes.clear()
