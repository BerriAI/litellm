"""
Regression tests: a user-configured pass-through whose ``path``
collides with an existing route must not record metadata, must not
pollute ``LiteLLMRoutes.openai_routes``, and its ``auth`` flag must
not apply to the request that lands on the shadowed built-in.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth  # noqa: E402
from litellm.proxy.auth.user_api_key_auth import (  # noqa: E402
    check_api_key_for_custom_headers_or_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: E402
    InitPassThroughEndpointHelpers,
    _register_pass_through_endpoint,
    _registered_pass_through_routes,
)

_HELPER_KWARGS = dict(
    target="http://attacker.example/sink",
    custom_headers=None,
    forward_headers=None,
    merge_query_params=None,
    dependencies=None,
    cost_per_request=None,
    endpoint_id="ep",
    methods=["POST"],
)


def _app_with_route(path: str, methods=None) -> FastAPI:
    app = FastAPI()

    async def _existing():
        return {}

    app.add_api_route(path=path, endpoint=_existing, methods=methods or ["POST"])
    return app


def _register_passthrough(endpoint_id: str, path: str, methods=None) -> None:
    methods = methods or ["POST"]
    key = f"{endpoint_id}:exact:{path}:{','.join(sorted(methods))}"
    _registered_pass_through_routes[key] = {
        "endpoint_id": endpoint_id,
        "path": path,
        "type": "exact",
        "methods": methods,
        "passthrough_params": {},
    }


def _mock_request(method: str = "POST") -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.method = method
    return request


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    # SERVER_ROOT_PATH leaks from test_custom_proxy.py under xdist;
    # strip it so registered-path lookups match raw test routes.
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    _registered_pass_through_routes.clear()
    original_openai_routes = list(LiteLLMRoutes.openai_routes.value)
    yield
    _registered_pass_through_routes.clear()
    LiteLLMRoutes.openai_routes.value[:] = original_openai_routes


@pytest.mark.parametrize(
    "helper,collide_path",
    [
        (
            InitPassThroughEndpointHelpers.add_exact_path_route,
            "/customer/block",
        ),
        (
            InitPassThroughEndpointHelpers.add_subpath_route,
            "/customer/block/{subpath:path}",
        ),
    ],
    ids=["exact", "subpath"],
)
def test_registration_collision_skips_metadata(helper, collide_path):
    app = _app_with_route(collide_path)
    helper(app=app, path="/customer/block", **_HELPER_KWARGS)
    assert _registered_pass_through_routes == {}


def test_non_colliding_path_registers():
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=FastAPI(),
        path="/forwarder/openai/chat",
        **{**_HELPER_KWARGS, "endpoint_id": "ep-ok"},
    )
    assert any(
        k.startswith("ep-ok:exact:/forwarder/openai/chat")
        for k in _registered_pass_through_routes
    )


def test_subpath_route_base_path_collision_skips_metadata():
    # Veria HIGH: ``add_subpath_route`` only checks the generated
    # ``/{subpath:path}`` route for collisions. But
    # ``get_registered_pass_through_route()`` returns a subpath entry
    # whose ``path`` matches the request route exactly — so an
    # ``include_subpath: true, auth: false`` entry whose base path
    # shadows a built-in route lets the auth gate honor the ``auth``
    # flag against the built-in. Refuse the metadata.
    app = _app_with_route("/customer/block")

    InitPassThroughEndpointHelpers.add_subpath_route(
        app=app,
        path="/customer/block",
        **{**_HELPER_KWARGS, "endpoint_id": "ep-shadowing-subpath"},
    )

    assert _registered_pass_through_routes == {}


def test_subpath_route_with_builtin_child_skips_metadata():
    # Veria HIGH (variant): a subpath registration at ``/v1/chat`` does
    # NOT collide with the wildcard or the base path on the app — but
    # ``get_registered_pass_through_route`` classifies every route under
    # ``/v1/chat/...`` as a match. If a built-in child route already
    # exists (e.g. ``/v1/chat/completions``), FastAPI dispatches the
    # request to the built-in but the auth gate would still find the
    # pass-through metadata for that ``(route, method)`` and honor its
    # ``auth: false`` flag. Refuse the registration.
    app = FastAPI()

    async def _existing_child():
        return {}

    app.add_api_route(
        path="/v1/chat/completions",
        endpoint=_existing_child,
        methods=["POST"],
    )

    InitPassThroughEndpointHelpers.add_subpath_route(
        app=app,
        path="/v1/chat",
        **{**_HELPER_KWARGS, "endpoint_id": "ep-shadowing-child"},
    )

    assert _registered_pass_through_routes == {}


def test_subpath_route_self_rereg_preserves_metadata_with_child_passthrough():
    # Greptile follow-up: ``_registered_pass_through_routes`` is not
    # cleared between config reloads. A subpath re-registration (same
    # endpoint_id, same path) must NOT be silently dropped when there's
    # a child exact pass-through route already on the app from the
    # previous reload — otherwise operator config changes (e.g. new
    # target URL) silently fail to apply.
    app = FastAPI()

    # First reload: pretend a sibling pass-through child route was
    # registered (this would normally be added via add_exact_path_route).
    async def _child_passthrough_endpoint():
        return {}

    app.add_api_route(
        path="/v1/chat/completions",
        endpoint=_child_passthrough_endpoint,
        methods=["POST"],
    )

    # Also pre-populate the registry with the subpath entry that this
    # reload re-registers.
    _registered_pass_through_routes["ep-subpath:subpath:/v1/chat:POST"] = {
        "endpoint_id": "ep-subpath",
        "path": "/v1/chat",
        "type": "subpath",
        "methods": ["POST"],
        "passthrough_params": {"target": "http://old.example/sink"},
    }

    InitPassThroughEndpointHelpers.add_subpath_route(
        app=app,
        path="/v1/chat",
        **{**_HELPER_KWARGS, "endpoint_id": "ep-subpath"},
    )

    # Metadata must remain — the self-rereg should not be dropped.
    assert "ep-subpath:subpath:/v1/chat:POST" in _registered_pass_through_routes


def test_parameterized_route_collision_skips_metadata():
    # A built-in such as ``/credentials/{credential_name:path}`` consumes
    # ``/credentials/foo`` via FastAPI's first-match dispatch. The pass-through
    # entry would never receive traffic but its ``auth`` flag could still
    # leak into the built-in's authorization path. Treat parameterized
    # consumption as a collision.
    app = FastAPI()

    async def _existing(credential_name: str):
        return {}

    app.add_api_route(
        path="/credentials/{credential_name:path}",
        endpoint=_existing,
        methods=["POST", "DELETE"],
    )

    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=app,
        path="/credentials/foo",
        **{**_HELPER_KWARGS, "endpoint_id": "ep-shadowed"},
    )

    assert _registered_pass_through_routes == {}


class TestRegisterPassThroughEndpointCollisionGuard:
    """``_register_pass_through_endpoint`` must bail before any mutation
    of ``LiteLLMRoutes.openai_routes`` — an ``auth=true`` colliding
    entry would otherwise mark the built-in as llm_api_route and
    short-circuit RBAC."""

    @pytest.mark.asyncio
    async def test_collision_does_not_append_to_openai_routes(self):
        app = _app_with_route("/customer/block")
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
        assert LiteLLMRoutes.openai_routes.value == before
        assert _registered_pass_through_routes == {}

    @pytest.mark.asyncio
    async def test_include_subpath_builtin_child_does_not_append_to_openai_routes(self):
        # Veria HIGH (order-of-operations variant): if the subpath
        # collision guard fires INSIDE add_subpath_route, the
        # ``auth=true`` branch above it has already appended the base
        # ``path`` to LiteLLMRoutes.openai_routes — leaving builtin
        # child routes (which my fix correctly prevents from getting
        # pass-through metadata) silently downgraded to llm_api_route.
        # _register_pass_through_endpoint must refuse the entry up
        # front, BEFORE any openai_routes mutation.
        app = FastAPI()

        async def _existing_child():
            return {}

        app.add_api_route(
            path="/v1/chat/completions",
            endpoint=_existing_child,
            methods=["POST"],
        )
        before = list(LiteLLMRoutes.openai_routes.value)

        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            await _register_pass_through_endpoint(
                endpoint={
                    "id": "ep-include-subpath-shadow",
                    "path": "/v1/chat",
                    "target": "http://attacker.example/sink",
                    "auth": True,
                    "include_subpath": True,
                    "methods": ["POST"],
                },
                app=app,
                premium_user=True,
                visited_endpoints=set(),
            )

        # openai_routes must not contain ``/v1/chat`` — appending it
        # would mark every method at every child route as llm_api_route
        # and silently downgrade RBAC on /v1/chat/completions.
        assert LiteLLMRoutes.openai_routes.value == before
        assert _registered_pass_through_routes == {}

    @pytest.mark.asyncio
    async def test_method_confused_new_method_treated_as_collision(self):
        # Existing GET /customer/block pass-through must not give a
        # later POST /customer/block entry a free pass — the POST
        # registration is NEW for that (path, method) and the built-in
        # POST handler still services the request.
        app = _app_with_route("/customer/block", methods=["POST"])
        _register_passthrough("ep-get-only", "/customer/block", methods=["GET"])
        before = list(LiteLLMRoutes.openai_routes.value)
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            await _register_pass_through_endpoint(
                endpoint={
                    "id": "ep-collision-post",
                    "path": "/customer/block",
                    "target": "http://attacker.example/sink",
                    "auth": False,
                    "methods": ["POST"],
                },
                app=app,
                premium_user=True,
                visited_endpoints=set(),
            )
        # POST entry must not have been registered.
        assert all(
            "ep-collision-post" not in k for k in _registered_pass_through_routes
        )
        # openai_routes must remain untouched.
        assert LiteLLMRoutes.openai_routes.value == before

    @pytest.mark.asyncio
    async def test_auth_true_other_method_builtin_does_not_pollute_openai_routes(self):
        # ``LiteLLMRoutes.openai_routes`` is method-blind: adding a path
        # there marks every method on that path as llm_api_route.
        # Refuse the openai_routes append when a built-in handler exists
        # at the same path on any other method.
        app = _app_with_route("/customer/block", methods=["POST"])
        before = list(LiteLLMRoutes.openai_routes.value)
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            await _register_pass_through_endpoint(
                endpoint={
                    "id": "ep-get-only",
                    "path": "/customer/block",
                    "target": "https://example.com/fwd",
                    "auth": True,
                    "methods": ["GET"],
                },
                app=app,
                premium_user=True,
                visited_endpoints=set(),
            )
        assert LiteLLMRoutes.openai_routes.value == before

    @pytest.mark.asyncio
    async def test_reregistration_does_not_warn_or_drop_metadata(self):
        # On config reload ``initialize_pass_through_endpoints`` re-invokes
        # this function with the same path; the route already exists in
        # ``app.routes`` from the previous load. That's not a collision
        # with a built-in — the deeper helpers must update metadata in
        # place rather than emit a "collision" warning + early return.
        app = FastAPI()
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header",
            new=AsyncMock(return_value=None),
        ):
            for _ in range(2):
                await _register_pass_through_endpoint(
                    endpoint={
                        "id": "ep-reload",
                        "path": "/forwarder/reload",
                        "target": "https://example.com",
                        "auth": True,
                        "methods": ["POST"],
                    },
                    app=app,
                    premium_user=True,
                    visited_endpoints=set(),
                )
        assert any(
            k.startswith("ep-reload:exact:/forwarder/reload")
            for k in _registered_pass_through_routes
        )

    @pytest.mark.asyncio
    async def test_non_colliding_auth_true_still_appends(self):
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
                app=FastAPI(),
                premium_user=True,
                visited_endpoints=set(),
            )
        assert "/forwarder/cohere/chat" in LiteLLMRoutes.openai_routes.value
        assert "/forwarder/cohere/chat" not in before


class TestAuthCheckIgnoresUnregisteredCollision:
    """Auth-bypass call sites must verify a forwarder is registered for
    the request's (route, method) before honoring ``auth: false``."""

    @pytest.mark.asyncio
    async def test_collision_config_does_not_return_blank_auth(self):
        # The literal GHSA-j99g PoC: config present, forwarder unregistered.
        response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=_mock_request(),
            route="/customer/block",
            pass_through_endpoints=[
                {"path": "/customer/block", "target": "x", "auth": False}
            ],
            api_key="sk-anonymous",
        )
        assert response == "sk-anonymous"

    @pytest.mark.asyncio
    async def test_registered_passthrough_auth_false_still_bypasses(self):
        _register_passthrough("ep-ok", "/forwarder/public", methods=["POST"])
        response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=_mock_request(method="POST"),
            route="/forwarder/public",
            pass_through_endpoints=[
                {"path": "/forwarder/public", "target": "x", "auth": False}
            ],
            api_key="sk-anonymous",
        )
        assert isinstance(response, UserAPIKeyAuth)

    @pytest.mark.asyncio
    async def test_method_aware_lookup_blocks_wrong_method_bypass(self):
        # Registered GET forwarder must not leak auth:false onto a POST
        # that hits the shadowed built-in handler.
        _register_passthrough("ep-get-only", "/customer/block", methods=["GET"])
        response = await check_api_key_for_custom_headers_or_pass_through_endpoints(
            request=_mock_request(method="POST"),
            route="/customer/block",
            pass_through_endpoints=[
                {
                    "path": "/customer/block",
                    "target": "x",
                    "auth": False,
                    "methods": ["GET"],
                }
            ],
            api_key="sk-anonymous",
        )
        assert response == "sk-anonymous"
