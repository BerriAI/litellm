"""
Tests for the provider-pinned standard routes module
(litellm/proxy/pinned_provider_routes.py).

Covers, per the 2026-07-21 provider-pinned-routes design:
- pin-tag injection on both dialects, including the router-visible metadata
  field (metadata vs litellm_metadata per LITELLM_METADATA_ROUTES);
- client-supplied tags are unioned with the pin, never replaced by it and
  never able to replace it;
- route precedence over the four provider pass-through catch-alls
  (/gemini, /bedrock, /azure, /vertex_ai {endpoint:path}) plus resolution
  for fireworks/baseten which have no catch-all (the R4 regression guard);
- unknown providers register nothing;
- disabled by default: no general_settings entry -> no routes;
- the REAL auth seam: every pinned path is an LLM API route and passes
  non_proxy_admin_allowed_routes_check for a plain non-admin key (round-1 P1:
  /fireworks and /baseten were 403-dead for non-admin keys);
- unmatched pin surfaces fast as a 400 naming the model and the pin tag
  (round-1 P2: it was a retried, seconds-late 401);
- openai/anthropic are denylisted as pinned prefixes (round-1 P2);
- messages-dialect registration schedules the heavy anthropic_endpoints
  import off the event loop, and re-init removes stale routes (round-1 P3s).
"""

import asyncio
import importlib
import os
import sys
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Match

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.pinned_provider_routes import (
    PINNED_PROVIDER_ROUTES_SETTING,
    get_pin_tag,
    initialize_pinned_provider_routes,
)

PINNED_PROVIDERS = [
    "azure",
    "azure_ai",
    "bedrock",
    "vertex_ai",
    "gemini",
    "fireworks",
    "baseten",
]

# All 14 pinned literal paths (7 providers x 2 dialects).
ALL_PINNED_PATHS = [
    f"/{provider}{suffix}" for provider in PINNED_PROVIDERS for suffix in ("/v1/chat/completions", "/v1/messages")
]

# Providers whose pass-through catch-all (/{p}/{endpoint:path}) overlaps the
# pinned literal paths — precedence over these is load-bearing (design R4).
CATCH_ALL_PROVIDERS = ["azure", "bedrock", "gemini", "vertex_ai"]

EXPECTED_PIN_TAGS = {
    "azure": "pin:azure",
    "azure_ai": "pin:azure_ai",
    "bedrock": "pin:bedrock",
    "vertex_ai": "pin:vertex_ai",
    "gemini": "pin:vertex_ai",  # alias: /gemini pins the vertex_ai backend
    "fireworks": "pin:fireworks",
    "baseten": "pin:baseten",
}

CHAT_STUB_SENTINEL = {"handler": "pinned-chat-stub"}
MESSAGES_STUB_SENTINEL = {"handler": "pinned-messages-stub"}


def _remove_pinned_routes(app) -> None:
    app.router.routes[:] = [
        r
        for r in app.router.routes
        if not (isinstance(r, APIRoute) and getattr(r.endpoint, "_pinned_provider_route", None) is not None)
    ]


@pytest.fixture
def pinned_app():
    """The real proxy app with all seven pinned providers registered and
    auth overridden; pinned routes and overrides are removed on teardown."""
    from litellm.proxy.proxy_server import app

    _remove_pinned_routes(app)  # tolerate leftovers from an aborted run
    registered = initialize_pinned_provider_routes(
        app=app,
        general_settings={PINNED_PROVIDER_ROUTES_SETTING: list(PINNED_PROVIDERS)},
    )
    assert len(registered) == 2 * len(PINNED_PROVIDERS)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="hashed-test-key")
    try:
        yield app
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)
        _remove_pinned_routes(app)


@pytest.fixture
def capture(monkeypatch):
    """Stub both delegated endpoint functions; capture the (post-injection)
    parsed request body each one receives."""
    import litellm.proxy.anthropic_endpoints.endpoints as anthropic_endpoints_module
    import litellm.proxy.proxy_server as proxy_server_module

    captured = {}

    async def chat_stub(request, fastapi_response, model=None, user_api_key_dict=None):
        captured["body"] = await _read_request_body(request=request)
        captured["path"] = request.scope["path"]
        return CHAT_STUB_SENTINEL

    async def messages_stub(fastapi_response=None, request=None, user_api_key_dict=None):
        captured["body"] = await _read_request_body(request=request)
        captured["path"] = request.scope["path"]
        return MESSAGES_STUB_SENTINEL

    monkeypatch.setattr(proxy_server_module, "chat_completion", chat_stub)
    monkeypatch.setattr(anthropic_endpoints_module, "anthropic_response", messages_stub)
    return captured


def _post(app, path: str, body: dict):
    client = TestClient(app)
    return client.post(path, json=body, headers={"Authorization": "Bearer sk-test"})


def _router_visible_tags(body: dict, path: str, expected_field: str):
    """Run the captured (post-injection) body through the real
    add_litellm_data_to_request and return the tag list the router sees."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.scope = {"path": path, "root_path": ""}
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"content-type": "application/json"}
    request_mock.url = MagicMock()
    request_mock.url.path = path
    request_mock.url.__str__.return_value = f"http://localhost{path}"
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    updated = asyncio.run(
        add_litellm_data_to_request(
            data=dict(body),
            request=request_mock,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-test-key"),
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )
    )
    other_field = "litellm_metadata" if expected_field == "metadata" else "metadata"
    assert not (updated.get(other_field) or {}).get("tags"), f"tags leaked into {other_field} for path {path}"
    return (updated.get(expected_field) or {}).get("tags")


class TestTagInjectionChatDialect:
    @pytest.mark.parametrize("provider", PINNED_PROVIDERS)
    def test_pin_appended_to_client_tags(self, pinned_app, capture, provider):
        resp = _post(
            pinned_app,
            f"/{provider}/v1/chat/completions",
            {"model": "some-model", "messages": [{"role": "user", "content": "hi"}], "tags": ["client-tag"]},
        )
        assert resp.status_code == 200
        assert resp.json() == CHAT_STUB_SENTINEL
        assert capture["body"]["tags"] == ["client-tag", EXPECTED_PIN_TAGS[provider]]

    def test_pin_alone_when_no_client_tags(self, pinned_app, capture):
        resp = _post(
            pinned_app,
            "/fireworks/v1/chat/completions",
            {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert capture["body"]["tags"] == ["pin:fireworks"]

    def test_tag_reaches_router_visible_metadata_plain_route(self, pinned_app, capture):
        """/azure/... does not match LITELLM_METADATA_ROUTES -> tags must land
        in `metadata` (the field tag-based routing reads for chat routes)."""
        path = "/azure/v1/chat/completions"
        resp = _post(pinned_app, path, {"model": "gpt-5.4-nano", "messages": [], "tags": ["client-tag"]})
        assert resp.status_code == 200
        tags = _router_visible_tags(capture["body"], path, expected_field="metadata")
        assert tags is not None and "pin:azure" in tags and "client-tag" in tags

    def test_tag_reaches_router_visible_metadata_litellm_metadata_route(self, pinned_app, capture):
        """/bedrock/... contains "bedrock", a LITELLM_METADATA_ROUTES entry ->
        tags must land in `litellm_metadata`."""
        path = "/bedrock/v1/chat/completions"
        resp = _post(pinned_app, path, {"model": "claude-haiku-4-5", "messages": []})
        assert resp.status_code == 200
        tags = _router_visible_tags(capture["body"], path, expected_field="litellm_metadata")
        assert tags is not None and "pin:bedrock" in tags


class TestTagInjectionMessagesDialect:
    @pytest.mark.parametrize("provider", PINNED_PROVIDERS)
    def test_pin_appended_to_client_tags(self, pinned_app, capture, provider):
        resp = _post(
            pinned_app,
            f"/{provider}/v1/messages",
            {
                "model": "some-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "tags": ["client-tag"],
            },
        )
        assert resp.status_code == 200
        assert resp.json() == MESSAGES_STUB_SENTINEL
        assert capture["body"]["tags"] == ["client-tag", EXPECTED_PIN_TAGS[provider]]

    def test_tag_reaches_litellm_metadata(self, pinned_app, capture):
        """Every pinned messages path contains "/v1/messages", a
        LITELLM_METADATA_ROUTES entry -> tags land in `litellm_metadata`."""
        path = "/azure/v1/messages"
        resp = _post(pinned_app, path, {"model": "gpt-5.4-nano", "messages": [], "max_tokens": 16})
        assert resp.status_code == 200
        tags = _router_visible_tags(capture["body"], path, expected_field="litellm_metadata")
        assert tags is not None and "pin:azure" in tags

    def test_messages_paths_match_litellm_metadata_routes(self, pinned_app):
        from litellm.proxy.litellm_pre_call_utils import LITELLM_METADATA_ROUTES

        for provider in PINNED_PROVIDERS:
            path = f"/{provider}/v1/messages"
            assert any(route in path for route in LITELLM_METADATA_ROUTES)


class TestClientTagsCannotReplacePin:
    def test_client_pin_tag_for_other_provider_is_unioned(self, pinned_app, capture):
        """A client sending its own pin: tag cannot REPLACE the server-side
        pin — both remain (union), which under subset matching can only
        narrow eligibility (worst case: matches nothing, fail-loud 4xx)."""
        resp = _post(
            pinned_app,
            "/bedrock/v1/chat/completions",
            {"model": "claude-haiku-4-5", "messages": [], "tags": ["pin:azure"]},
        )
        assert resp.status_code == 200
        assert capture["body"]["tags"] == ["pin:azure", "pin:bedrock"]

    def test_duplicate_pin_not_doubled(self, pinned_app, capture):
        resp = _post(
            pinned_app,
            "/bedrock/v1/chat/completions",
            {"model": "claude-haiku-4-5", "messages": [], "tags": ["pin:bedrock"]},
        )
        assert resp.status_code == 200
        assert capture["body"]["tags"] == ["pin:bedrock"]

    def test_non_list_tags_replaced_by_pin(self, pinned_app, capture):
        """Malformed (non-list) client tags are ignored downstream anyway;
        the wrapper must still guarantee the pin applies."""
        resp = _post(
            pinned_app,
            "/bedrock/v1/chat/completions",
            {"model": "claude-haiku-4-5", "messages": [], "tags": "not-a-list"},
        )
        assert resp.status_code == 200
        assert capture["body"]["tags"] == ["pin:bedrock"]


def _first_full_match_route(app, path: str):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "root_path": "",
        "headers": [],
        "query_string": b"",
        "path_params": {},
    }
    for route in app.router.routes:
        try:
            match, _ = route.matches(scope)
        except Exception:
            continue
        if match is Match.FULL:
            return route
    return None


class TestRoutePrecedence:
    """The R4 regression guard: the pinned literal routes must win over the
    provider pass-through catch-alls in FastAPI's in-order route matching."""

    @pytest.mark.parametrize("provider", PINNED_PROVIDERS)
    @pytest.mark.parametrize("suffix", ["/v1/chat/completions", "/v1/messages"])
    def test_route_table_resolves_to_pinned_handler(self, pinned_app, provider, suffix):
        path = f"/{provider}{suffix}"
        route = _first_full_match_route(pinned_app, path)
        assert route is not None, f"no route resolves {path}"
        assert getattr(route.endpoint, "_pinned_provider_route", None) == provider, (
            f"{path} resolves to {route.path!r} (endpoint {route.name!r}), not the pinned handler"
        )
        assert route.path == path  # literal, not {endpoint:path}

    @pytest.mark.parametrize("provider", CATCH_ALL_PROVIDERS)
    def test_pinned_routes_precede_the_catch_all(self, pinned_app, provider):
        """The catch-all must still exist AND sit after the pinned routes —
        proves the pinned routes actually beat a live overlapping route."""
        routes = pinned_app.router.routes
        catch_all_path = f"/{provider}/{{endpoint:path}}"
        catch_all_idx = [i for i, r in enumerate(routes) if getattr(r, "path", None) == catch_all_path]
        assert catch_all_idx, f"expected the {catch_all_path} pass-through catch-all to exist"
        pinned_idx = [
            i
            for i, r in enumerate(routes)
            if isinstance(r, APIRoute) and getattr(r.endpoint, "_pinned_provider_route", None) == provider
        ]
        assert pinned_idx, f"pinned routes for {provider} missing"
        assert max(pinned_idx) < min(catch_all_idx), (
            f"pinned routes for {provider} are registered AFTER its catch-all — "
            "they would be swallowed by the pass-through route"
        )

    @pytest.mark.parametrize("provider", PINNED_PROVIDERS)
    def test_request_level_resolution(self, pinned_app, capture, provider):
        """Request-level proof (covers fireworks/baseten too, which have no
        catch-all): a POST reaches the pinned handler, not a pass-through."""
        chat = _post(pinned_app, f"/{provider}/v1/chat/completions", {"model": "m", "messages": []})
        assert chat.status_code == 200
        assert chat.json() == CHAT_STUB_SENTINEL
        messages = _post(pinned_app, f"/{provider}/v1/messages", {"model": "m", "messages": [], "max_tokens": 16})
        assert messages.status_code == 200
        assert messages.json() == MESSAGES_STUB_SENTINEL


class TestRegistrationGating:
    def test_unknown_provider_not_registered(self):
        app = FastAPI()
        registered = initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock", "definitely_not_a_provider"]},
        )
        assert registered == ["/bedrock/v1/chat/completions", "/bedrock/v1/messages"]
        assert not any("definitely_not_a_provider" in getattr(r, "path", "") for r in app.router.routes)

    def test_disabled_by_default_no_general_settings_entry(self):
        app = FastAPI()
        n_routes = len(app.router.routes)
        assert initialize_pinned_provider_routes(app=app, general_settings={}) == []
        assert initialize_pinned_provider_routes(app=app, general_settings=None) == []
        assert initialize_pinned_provider_routes(app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: []}) == []
        assert len(app.router.routes) == n_routes

    def test_disabled_by_default_on_the_real_proxy_app(self):
        """The real app, as imported (no pinned_provider_routes config), must
        not carry any pinned route."""
        from litellm.proxy.proxy_server import app

        assert not any(
            isinstance(r, APIRoute) and getattr(r.endpoint, "_pinned_provider_route", None) is not None
            for r in app.router.routes
        )

    def test_non_list_setting_ignored(self):
        app = FastAPI()
        assert (
            initialize_pinned_provider_routes(app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: "bedrock"})
            == []
        )

    def test_reinitialize_is_idempotent(self):
        app = FastAPI()
        first = initialize_pinned_provider_routes(
            app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]}
        )
        assert len(first) == 2
        n_routes = len(app.router.routes)
        second = initialize_pinned_provider_routes(
            app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]}
        )
        assert second == []
        assert len(app.router.routes) == n_routes

    def test_gemini_alias_maps_to_vertex_ai_tag(self):
        assert get_pin_tag("gemini") == "pin:vertex_ai"
        assert get_pin_tag("bedrock") == "pin:bedrock"
        assert get_pin_tag("fireworks") == "pin:fireworks"


@pytest.mark.asyncio
async def test_load_config_seam_registers_pinned_routes(tmp_path, monkeypatch):
    """The ProxyConfig.load_config hook is the only production caller of
    initialize_pinned_provider_routes — guard the seam so an upstream sync
    cannot silently drop it (per-stage-green/feature-dead prevention)."""
    from litellm.proxy.proxy_server import ProxyConfig, app

    f = tmp_path / "c.yaml"
    f.write_text("model_list: []\nlitellm_settings: {}\ngeneral_settings:\n  pinned_provider_routes: [baseten]\n")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    _remove_pinned_routes(app)
    try:
        await ProxyConfig().load_config(router=None, config_file_path=str(f))
        pinned_paths = {
            r.path
            for r in app.router.routes
            if isinstance(r, APIRoute) and getattr(r.endpoint, "_pinned_provider_route", None) is not None
        }
        assert pinned_paths == {"/baseten/v1/chat/completions", "/baseten/v1/messages"}
    finally:
        _remove_pinned_routes(app)


class TestAuthSeamNonAdminAccess:
    """Round-1 P1 regression guard, exercised at the REAL auth seam (no
    dependency_overrides): every pinned path must be recognized as an LLM API
    route, and non_proxy_admin_allowed_routes_check must not raise for a
    plain non-admin key. Before the fix, /fireworks/* and /baseten/* were
    missing from LiteLLMRoutes.mapped_pass_through_routes, so both checks
    failed and non-admin keys got the admin-only 403."""

    @pytest.mark.parametrize("path", ALL_PINNED_PATHS)
    def test_is_llm_api_route(self, path):
        from litellm.proxy.auth.route_checks import RouteChecks

        assert RouteChecks.is_llm_api_route(route=path) is True

    @pytest.mark.parametrize("path", ALL_PINNED_PATHS)
    def test_non_admin_key_passes_route_check(self, path):
        from litellm.proxy.auth.route_checks import RouteChecks

        request = MagicMock(spec=Request)
        request.method = "POST"
        request.query_params = {}

        # Must not raise for a plain, role-less, non-admin key.
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=None,
            route=path,
            request=request,
            valid_token=UserAPIKeyAuth(api_key="hashed-non-admin-key"),
            request_data={},
        )


class TestUnmatchedPinFailsFastAs400:
    """Round-1 P2 regression guard: a pin tag that matches no deployment must
    surface immediately (no retry burn) as a 400 naming the model and the
    pin tag — not a seconds-late 401."""

    @staticmethod
    def _unmatched_router() -> "litellm.Router":
        return litellm.Router(
            model_list=[
                {
                    "model_name": "deepseek-v4",
                    "litellm_params": {
                        "model": "openai/deepseek-v4",
                        "api_key": "sk-fake",
                        "api_base": "http://127.0.0.1:1/nope",
                        "tags": ["pin:baseten"],
                    },
                }
            ],
            enable_tag_filtering=True,
            tag_filtering_match_any=False,
            num_retries=2,
        )

    @pytest.mark.asyncio
    async def test_router_raises_non_retryable_400_fast(self):
        """Reviewer repro: num_retries=2, tag filtering on, no matching
        deployment. Before the fix this was a plain ValueError retried for
        ~5s; it must now be a BadRequestError(400) in well under a second
        (loose <2s bound to stay CI-safe while still failing the retried
        path, which cannot finish that fast)."""
        router = self._unmatched_router()
        start = time.perf_counter()
        with pytest.raises(litellm.BadRequestError) as exc_info:
            await router.acompletion(
                model="deepseek-v4",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["pin:fireworks"]},
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"unmatched pin took {elapsed:.2f}s to surface — retries are burning again"
        assert exc_info.value.status_code == 400
        assert "deepseek-v4" in str(exc_info.value)
        assert "pin:fireworks" in str(exc_info.value)

    def test_proxy_exception_maps_tag_routing_to_400_not_401(self):
        from litellm.proxy._types import ProxyException
        from litellm.types.router import RouterErrors

        exc = ProxyException(
            message=f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model=m and tags=['pin:x']",
            type="bad_request_error",
            param=None,
            code=400,
        )
        assert exc.code == "400"

    def test_proxy_surfaces_400_with_model_and_pin_tag(self, pinned_app, monkeypatch):
        """Proxy-level: POST to a pinned route whose pin matches no
        deployment must return HTTP 400 (not 401), and the body must name
        the model and the pin tag."""
        import litellm.proxy.proxy_server as proxy_server_module

        monkeypatch.setattr(proxy_server_module, "llm_router", self._unmatched_router())
        start = time.perf_counter()
        resp = _post(
            pinned_app,
            "/fireworks/v1/chat/completions",
            {"model": "deepseek-v4", "messages": [{"role": "user", "content": "hi"}]},
        )
        elapsed = time.perf_counter() - start
        assert resp.status_code == 400, f"expected 400, got {resp.status_code}: {resp.text}"
        assert "deepseek-v4" in resp.text
        assert "pin:fireworks" in resp.text
        assert elapsed < 2.0, f"unmatched pin took {elapsed:.2f}s to surface at the proxy"


class TestPinnedPrefixDenylist:
    """Round-1 P2 regression guard: openai/anthropic are real litellm
    provider names, but /openai/* and /anthropic/* are live credentialed
    pass-through surfaces — pinning them would shadow those trees, so they
    are warn+skipped even when configured."""

    @pytest.mark.parametrize("denied", ["openai", "anthropic"])
    def test_denylisted_prefix_skipped(self, denied):
        app = FastAPI()
        registered = initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: [denied, "bedrock"]},
        )
        assert registered == ["/bedrock/v1/chat/completions", "/bedrock/v1/messages"]
        assert not any(getattr(r, "path", "").startswith(f"/{denied}/") for r in app.router.routes)


class TestMessagesDialectWarmup:
    """Round-1 P3(a) regression guard: registering any messages-dialect
    route must schedule the heavy anthropic_endpoints import on the default
    executor (off the event loop), mirroring _lazy_features._force_load."""

    def test_registration_schedules_executor_warmup(self, monkeypatch):
        fake_loop = MagicMock()
        monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

        app = FastAPI()
        registered = initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]},
        )
        assert "/bedrock/v1/messages" in registered
        fake_loop.run_in_executor.assert_called_once_with(
            None, importlib.import_module, "litellm.proxy.anthropic_endpoints.endpoints"
        )

    def test_no_warmup_when_nothing_registered(self, monkeypatch):
        fake_loop = MagicMock()
        monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

        app = FastAPI()
        assert initialize_pinned_provider_routes(app=app, general_settings={}) == []
        fake_loop.run_in_executor.assert_not_called()


class TestReinitializeRemovesStaleRoutes:
    """Round-1 P3(b) regression guard: a config reload that drops a provider
    (or the whole setting) must remove its previously-registered routes."""

    @staticmethod
    def _pinned_paths(app) -> set:
        return {
            r.path
            for r in app.router.routes
            if isinstance(r, APIRoute) and getattr(r.endpoint, "_pinned_provider_route", None) is not None
        }

    def test_add_then_remove_then_reload_drops_stale_route(self):
        app = FastAPI()
        first = initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock", "baseten"]},
        )
        assert len(first) == 4

        second = initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]},
        )
        assert second == []  # bedrock already registered; nothing new
        assert self._pinned_paths(app) == {"/bedrock/v1/chat/completions", "/bedrock/v1/messages"}

    def test_reload_without_setting_drops_all_pinned_routes(self):
        app = FastAPI()
        initialize_pinned_provider_routes(
            app=app,
            general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]},
        )
        assert self._pinned_paths(app)
        assert initialize_pinned_provider_routes(app=app, general_settings={}) == []
        assert self._pinned_paths(app) == set()

    def test_reload_can_reregister_a_removed_provider(self):
        app = FastAPI()
        initialize_pinned_provider_routes(
            app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock", "baseten"]}
        )
        initialize_pinned_provider_routes(app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock"]})
        third = initialize_pinned_provider_routes(
            app=app, general_settings={PINNED_PROVIDER_ROUTES_SETTING: ["bedrock", "baseten"]}
        )
        assert third == ["/baseten/v1/chat/completions", "/baseten/v1/messages"]
        assert self._pinned_paths(app) == {
            "/bedrock/v1/chat/completions",
            "/bedrock/v1/messages",
            "/baseten/v1/chat/completions",
            "/baseten/v1/messages",
        }
