"""
Regression tests for the pass-through route-collision fail-closed
behavior.

Background. A user-configured pass-through endpoint whose ``path``
collides with a built-in management route (e.g. ``/customer/block``)
cannot register a forwarder — ``SafeRouteAdder`` refuses to overwrite
the existing FastAPI route. Previous code still recorded the
pass-through metadata in ``_registered_pass_through_routes`` and still
honored the entry's ``auth`` flag in the auth helpers. Net effect: a
``path: /customer/block, auth: false`` config silently bypassed
authentication on the built-in handler that actually serviced the
request.

These tests lock down the three legs of the fix:

1. ``add_exact_path_route`` and ``add_subpath_route`` skip metadata
   registration when ``SafeRouteAdder.add_api_route_if_not_exists``
   returns ``False``.
2. ``_register_pass_through_endpoint`` returns early on collision so
   the colliding path is not added to ``LiteLLMRoutes.openai_routes``.
3. ``check_api_key_for_custom_headers_or_pass_through_endpoints`` and
   the common_checks ``auth: false`` bypass only fire when the
   forwarder is actually registered.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy._types import LiteLLMRoutes  # noqa: E402
from litellm.proxy.auth.user_api_key_auth import (  # noqa: E402
    check_api_key_for_custom_headers_or_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    InitPassThroughEndpointHelpers,
    _register_pass_through_endpoint,
    _registered_pass_through_routes,
)


def _fresh_app_with_builtin(path: str, methods=None) -> FastAPI:
    """FastAPI app that already has ``path`` registered with the given
    methods — mimics the built-in management routes the proxy mounts
    before user-configured pass-throughs are loaded."""
    methods = methods or ["POST"]
    app = FastAPI()

    async def existing_handler():
        return {"existing": True}

    app.add_api_route(path=path, endpoint=existing_handler, methods=methods)
    return app


@pytest.fixture(autouse=True)
def _reset_registry_and_openai_routes():
    """Ensure each test starts with a clean registry + ``openai_routes``
    that has not been polluted by prior test runs in the module."""
    _registered_pass_through_routes.clear()
    original_openai_routes = list(LiteLLMRoutes.openai_routes.value)
    yield
    _registered_pass_through_routes.clear()
    LiteLLMRoutes.openai_routes.value[:] = original_openai_routes


class TestRegistrationCollisionSkipsMetadata:
    """Fix A: helpers do not record metadata for a forwarder that
    ``SafeRouteAdder`` refused to register."""

    def test_add_exact_path_route_skips_metadata_on_collision(self):
        app = _fresh_app_with_builtin("/customer/block", methods=["POST"])

        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/customer/block",
            target="http://attacker.example/sink",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="ep-collision",
            methods=["POST"],
        )

        assert _registered_pass_through_routes == {}, (
            "Collision must not populate the in-memory registry — "
            "downstream auth checks read from it."
        )

    def test_add_subpath_route_skips_metadata_on_collision(self):
        wildcard = "/customer/block/{subpath:path}"
        app = _fresh_app_with_builtin(wildcard, methods=["POST"])

        InitPassThroughEndpointHelpers.add_subpath_route(
            app=app,
            path="/customer/block",
            target="http://attacker.example/sink",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="ep-collision",
            methods=["POST"],
        )

        assert _registered_pass_through_routes == {}

    def test_non_colliding_path_still_registers(self):
        app = FastAPI()

        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path="/forwarder/openai/chat",
            target="https://api.openai.com/v1/chat/completions",
            custom_headers=None,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=None,
            endpoint_id="ep-ok",
            methods=["POST"],
        )

        assert any(
            key.startswith("ep-ok:exact:/forwarder/openai/chat")
            for key in _registered_pass_through_routes
        )


class TestRegisterPassThroughEndpointCollisionGuard:
    """Fix B: ``_register_pass_through_endpoint`` bails before mutating
    ``LiteLLMRoutes.openai_routes`` when the path collides — otherwise
    a colliding ``auth=true`` config would downgrade the built-in's
    RBAC by marking the path as an llm_api_route."""

    @pytest.mark.asyncio
    async def test_collision_does_not_append_to_openai_routes(self):
        app = _fresh_app_with_builtin("/customer/block", methods=["POST"])
        before = list(LiteLLMRoutes.openai_routes.value)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            await _register_pass_through_endpoint(
                endpoint={
                    "id": "ep-collision",
                    "path": "/customer/block",
                    "target": "http://attacker.example/sink",
                    "auth": True,
                    "methods": ["POST"],
                },
                app=app,
                premium_user=True,
                visited_endpoints=set(),
            )

        assert LiteLLMRoutes.openai_routes.value == before, (
            "Colliding path must not be added to LiteLLMRoutes.openai_routes "
            "— doing so marks the existing built-in route as an llm_api_route "
            "and short-circuits the RBAC role gate."
        )
        assert _registered_pass_through_routes == {}

    @pytest.mark.asyncio
    async def test_non_colliding_auth_true_still_appends(self):
        app = FastAPI()
        before = set(LiteLLMRoutes.openai_routes.value)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            await _register_pass_through_endpoint(
                endpoint={
                    "id": "ep-ok",
                    "path": "/forwarder/cohere/chat",
                    "target": "https://api.cohere.com/v1/chat",
                    "auth": True,
                    "methods": ["POST"],
                },
                app=app,
                premium_user=True,
                visited_endpoints=set(),
            )

        assert "/forwarder/cohere/chat" in LiteLLMRoutes.openai_routes.value
        assert "/forwarder/cohere/chat" not in before


class TestAuthCheckIgnoresUnregisteredCollision:
    """Fix C: the two auth-bypass call sites
    (``check_api_key_for_custom_headers_or_pass_through_endpoints`` and
    the common_checks ``auth: false`` short-circuit) require the
    forwarder to be actually registered before honoring ``auth: false``.

    Without this gate, a config entry alone is enough — the literal
    GHSA-j99g PoC: ``path: /customer/block, auth: false`` in the
    config dict makes any unauthenticated caller reach the built-in
    ``/customer/block`` handler with a blank ``UserAPIKeyAuth``."""

    @pytest.mark.asyncio
    async def test_collision_config_does_not_return_blank_auth(self):
        """The colliding entry is present in config but absent from
        ``_registered_pass_through_routes``. The auth helper must fall
        through to the normal path instead of returning blank auth."""
        request = MagicMock()
        request.headers = {}

        response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=request,
            route="/customer/block",
            pass_through_endpoints=[
                {
                    "path": "/customer/block",
                    "target": "http://attacker.example/sink",
                    "auth": False,
                }
            ],
            api_key="sk-anonymous",
        )

        assert response == "sk-anonymous", (
            "Colliding config with auth:false must not produce a blank "
            "UserAPIKeyAuth — the request is dispatching to the built-in "
            "handler, not to a forwarder, so the config's auth flag does "
            "not apply."
        )

    @pytest.mark.asyncio
    async def test_registered_passthrough_auth_false_still_bypasses(self):
        """Regression: a *registered* non-colliding pass-through with
        ``auth: false`` still produces the blank token, preserving the
        legitimate unauthenticated-forwarder feature."""
        request = MagicMock()
        request.headers = {}

        _registered_pass_through_routes["ep-ok:exact:/forwarder/public:POST"] = {
            "endpoint_id": "ep-ok",
            "path": "/forwarder/public",
            "type": "exact",
            "methods": ["POST"],
            "passthrough_params": {},
        }

        from litellm.proxy._types import UserAPIKeyAuth

        response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=request,
            route="/forwarder/public",
            pass_through_endpoints=[
                {
                    "path": "/forwarder/public",
                    "target": "http://target.example/x",
                    "auth": False,
                }
            ],
            api_key="sk-anonymous",
        )

        assert isinstance(response, UserAPIKeyAuth), (
            "Legitimate registered pass-through with auth:false must "
            "still produce a blank UserAPIKeyAuth (the documented "
            "unauthenticated-forwarder behavior)."
        )
