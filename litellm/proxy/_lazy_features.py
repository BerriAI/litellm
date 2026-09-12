"""
Lazy registration for optional feature routers. Each LAZY_FEATURES entry
imports its module only on the first request matching its path prefix,
saving ~700 MB at idle for deployments that don't use these features.
First hit pays the import cost (1-3 s for heavy modules); /openapi.json
omits each feature's routes until the feature is warmed.
"""

import asyncio
import importlib
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from starlette.routing import BaseRoute, Match
from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.proxy.route_priority import hot_routes_first

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI


def _include_router(attr_name: str = "router") -> Callable[["FastAPI", object], None]:
    def _register(app: "FastAPI", module: object) -> None:
        app.include_router(getattr(module, attr_name))

    return _register


def _mount_app(prefix: str, attr_name: str = "app") -> Callable[["FastAPI", object], None]:
    def _register(app: "FastAPI", module: object) -> None:
        app.mount(path=prefix, app=getattr(module, attr_name))

    return _register


@dataclass(frozen=True)
class LazyFeature:
    name: str
    module_path: str
    path_prefixes: tuple[str, ...]
    register_fn: Callable[["FastAPI", object], None] = field(default_factory=lambda: _include_router("router"))
    # For routes whose path has a leading parameter (e.g. /{server}/authorize)
    # — startswith can't match those, so the matcher also checks endswith.
    path_suffixes: tuple[str, ...] = ()
    # Keep the stub injected even after load — for mounted ASGI sub-apps
    # whose routes don't appear in the parent app's openapi spec.
    persistent_swagger_stub: bool = False

    def matches(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.path_prefixes) or any(path.endswith(s) for s in self.path_suffixes)


LAZY_FEATURES: Final[tuple[LazyFeature, ...]] = (
    LazyFeature(
        name="guardrails",
        module_path="litellm.proxy.guardrails.guardrail_endpoints",
        path_prefixes=(
            "/guardrails",
            "/v2/guardrails",
            "/apply_guardrail",
            "/policies/usage",
        ),
    ),
    LazyFeature(
        name="policies",
        module_path="litellm.proxy.management_endpoints.policy_endpoints",
        # Trailing slash to avoid matching /policies/... (policy_engine).
        path_prefixes=("/policy/", "/utils/test_policies_and_guardrails"),
    ),
    LazyFeature(
        name="policy_engine",
        module_path="litellm.proxy.policy_engine.policy_endpoints",
        path_prefixes=("/policies",),
    ),
    LazyFeature(
        name="policy_resolve",
        module_path="litellm.proxy.policy_engine.policy_resolve_endpoints",
        path_prefixes=("/policies/resolve", "/policies/attachments/estimate-impact"),
    ),
    LazyFeature(
        name="agents",
        module_path="litellm.proxy.agent_endpoints.endpoints",
        path_prefixes=("/v1/agents", "/agents", "/agent/"),
    ),
    LazyFeature(
        name="gemini_agents",
        module_path="litellm.proxy.google_endpoints.agents_endpoints",
        path_prefixes=("/v1beta/agents",),
    ),
    LazyFeature(
        name="a2a",
        module_path="litellm.proxy.agent_endpoints.a2a_endpoints",
        # ``/v1/a2a/{agent_id}/message/send`` is caught via the suffix so the
        # ``/v1/a2a`` prefix doesn't subsume the discover prefix below.
        path_prefixes=("/a2a",),
        path_suffixes=("/message/send",),
    ),
    LazyFeature(
        name="a2a_registration",
        module_path="litellm.proxy.a2a.endpoints",
        path_prefixes=("/v1/a2a/discover",),
    ),
    LazyFeature(
        name="vector_stores",
        module_path="litellm.proxy.vector_store_endpoints.endpoints",
        path_prefixes=("/v1/vector_stores", "/vector_stores", "/v1/indexes"),
    ),
    LazyFeature(
        name="vector_store_management",
        module_path="litellm.proxy.vector_store_endpoints.management_endpoints",
        # Trailing slash to avoid matching /vector_stores/... (vector_stores).
        path_prefixes=("/vector_store/", "/v1/vector_store/"),
    ),
    LazyFeature(
        name="vector_store_files",
        # Routes appear under both /v1/vector_stores/{id}/files and the
        # un-versioned form, so both prefixes must trigger the load.
        module_path="litellm.proxy.vector_store_files_endpoints.endpoints",
        path_prefixes=("/v1/vector_stores", "/vector_stores"),
    ),
    LazyFeature(
        name="tools",
        module_path="litellm.proxy.management_endpoints.tool_management_endpoints",
        path_prefixes=("/v1/tool", "/tool"),
    ),
    LazyFeature(
        name="search_tools",
        module_path="litellm.proxy.search_endpoints.search_tool_management",
        path_prefixes=("/search_tools",),
    ),
    # mcp_management owns most /v1/mcp/* admin routes; mcp_app is the mounted
    # streaming sub-app at /mcp.
    LazyFeature(
        name="mcp_management",
        module_path="litellm.proxy.management_endpoints.mcp_management_endpoints",
        path_prefixes=("/v1/mcp/",),
    ),
    LazyFeature(
        # Also serves /.well-known/oauth-* (OAuth metadata discovery).
        # No /mcp/oauth prefix here: the mounted /mcp sub-app would
        # shadow it, and there are no actual routes there anyway.
        name="mcp_byok_oauth",
        module_path="litellm.proxy._experimental.mcp_server.byok_oauth_endpoints",
        path_prefixes=("/v1/mcp/oauth", "/.well-known/oauth-"),
    ),
    LazyFeature(
        # Serves OAuth dance endpoints (/authorize, /token, /callback,
        # /register) plus several /.well-known/ discovery URLs at the proxy
        # root — needed for MCP-over-OAuth flows even before /mcp is hit.
        name="mcp_discoverable",
        module_path="litellm.proxy._experimental.mcp_server.discoverable_endpoints",
        path_prefixes=(
            "/.well-known/oauth-",
            "/.well-known/openid-configuration",
            "/.well-known/jwks.json",
            "/.well-known/litellm-cli-auth",
            "/authorize",
            "/token",
            "/callback",
            "/register",
            "/revoke",
            "/introspect",
        ),
        # Catches the /{mcp_server_name}/authorize|token|register variants.
        path_suffixes=("/authorize", "/token", "/register"),
    ),
    LazyFeature(
        name="mcp_rest",
        module_path="litellm.proxy._experimental.mcp_server.rest_endpoints",
        path_prefixes=("/mcp-rest",),
    ),
    LazyFeature(
        # Hardcoded /mcp matches BASE_MCP_ROUTE; importing the constant
        # here would defeat lazy loading.
        name="mcp_app",
        module_path="litellm.proxy._experimental.mcp_server.server",
        path_prefixes=("/mcp",),
        register_fn=_mount_app("/mcp", attr_name="app"),
        persistent_swagger_stub=True,
    ),
    LazyFeature(
        name="config_overrides",
        module_path="litellm.proxy.management_endpoints.config_override_endpoints",
        path_prefixes=("/config_overrides",),
    ),
    LazyFeature(
        name="llm_passthrough",
        module_path="litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        path_prefixes=(
            "/anthropic/",
            "/assemblyai/",
            "/azure/",
            "/azure_ai/",
            "/bedrock/",
            "/cohere/",
            "/comprehendmedical",
            "/cursor/",
            "/eu.assemblyai/",
            "/gemini/",
            "/gigachat/",
            "/milvus/",
            "/mistral/",
            "/openai/",
            "/openai_passthrough/",
            "/vertex-ai/",
            "/vertex_ai/",
            "/vllm/",
            "/watsonx/",
        ),
    ),
    LazyFeature(
        name="realtime",
        module_path="litellm.proxy.realtime_endpoints.endpoints",
        path_prefixes=("/openai/v1/realtime", "/v1/realtime", "/realtime"),
    ),
    LazyFeature(
        name="anthropic_passthrough",
        module_path="litellm.proxy.anthropic_endpoints.endpoints",
        path_prefixes=("/v1/messages", "/anthropic", "/api/event_logging"),
    ),
    LazyFeature(
        name="anthropic_skills",
        module_path="litellm.proxy.anthropic_endpoints.skills_endpoints",
        path_prefixes=("/v1/skills", "/skills"),
    ),
    LazyFeature(
        name="langfuse_passthrough",
        module_path="litellm.proxy.vertex_ai_endpoints.langfuse_endpoints",
        path_prefixes=("/langfuse",),
    ),
    LazyFeature(
        name="evals",
        module_path="litellm.proxy.openai_evals_endpoints.endpoints",
        path_prefixes=("/v1/evals", "/evals"),
    ),
    LazyFeature(
        name="claude_code_marketplace",
        module_path="litellm.proxy.anthropic_endpoints.claude_code_endpoints",
        path_prefixes=("/claude-code",),
        register_fn=_include_router("claude_code_marketplace_router"),
    ),
    LazyFeature(
        name="scim",
        module_path="litellm.proxy.management_endpoints.scim.scim_v2",
        path_prefixes=("/scim",),
        register_fn=_include_router("scim_router"),
    ),
    LazyFeature(
        name="cloudzero",
        module_path="litellm.proxy.spend_tracking.cloudzero_endpoints",
        path_prefixes=("/cloudzero",),
    ),
    LazyFeature(
        name="vantage",
        module_path="litellm.proxy.spend_tracking.vantage_endpoints",
        path_prefixes=("/vantage",),
    ),
    LazyFeature(
        name="usage_ai",
        module_path="litellm.proxy.management_endpoints.usage_endpoints",
        path_prefixes=("/usage/ai",),
    ),
    LazyFeature(
        name="prompts",
        module_path="litellm.proxy.prompts.prompt_endpoints",
        path_prefixes=("/prompts", "/utils/dotprompt_json_converter"),
    ),
    LazyFeature(
        name="jwt_mappings",
        module_path="litellm.proxy.management_endpoints.jwt_key_mapping_endpoints",
        path_prefixes=("/jwt/key/mapping",),
    ),
    LazyFeature(
        name="compliance",
        module_path="litellm.proxy.management_endpoints.compliance_endpoints",
        path_prefixes=("/compliance",),
    ),
    LazyFeature(
        name="access_groups",
        module_path="litellm.proxy.management_endpoints.access_group_endpoints",
        path_prefixes=("/access_group", "/v1/access_group", "/v1/unified_access_group"),
    ),
)


class LazyFeatureMiddleware:
    """ASGI middleware that imports + registers a feature router on first
    matching request. Idempotent; once loaded, subsequent requests skip."""

    def __init__(
        self,
        app,
        fastapi_app: "FastAPI",
        features: tuple[LazyFeature, ...] = LAZY_FEATURES,
    ):
        self.app = app
        self._fastapi_app = fastapi_app
        self._features = features
        # SERVER_ROOT_PATH is a process-startup env var, cache the normalized
        # form once instead of recomputing per request. Lazy import to avoid
        # pulling proxy.utils into this module's import graph at startup
        # (proxy_server imports both).
        from litellm.proxy.utils import get_server_root_path

        self._root_path = get_server_root_path().rstrip("/")
        # Loaded set / per-feature locks live on app.state so the warm endpoint
        # and the middleware share them — preventing duplicate registrations
        # when both paths fire for the same feature.
        if not hasattr(fastapi_app.state, "lazy_loaded"):
            fastapi_app.state.lazy_loaded = set()
            fastapi_app.state.lazy_locks = {}

    @property
    def _loaded(self) -> set:
        return self._fastapi_app.state.lazy_loaded

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Short-circuit once every feature has loaded.
        if scope["type"] in ("http", "websocket") and len(self._loaded) < len(self._features):
            path = scope.get("path", "")
            # Strip the request's root_path so prefix matching works under a
            # server root path. Without this, requests like /api/v1/policies/...
            # never match the registered prefixes (/policies/...) and lazy
            # features stay unloaded — every endpoint under them returns 404.
            # scope["root_path"] wins over the cached env scalar: FastAPI
            # stamps SERVER_ROOT_PATH there, and PerRequestRootPathMiddleware
            # resolves SERVER_ROOT_PATHS prefixes there per request. The
            # `+ "/"` boundary prevents false-positive matches (e.g. /apiv2
            # against root /api); a pre-stripped path is left alone.
            root_path: Final = str(scope.get("root_path", "")).rstrip("/") or self._root_path
            if root_path and path.startswith(root_path + "/"):
                path = path[len(root_path) :]  # rebind-ok: local strip after the boundary check above
            for feat in self._features:
                if feat.module_path in self._loaded or not feat.matches(path):
                    continue
                if _eager_route_wins(self._fastapi_app, feat, scope):
                    continue
                await _force_load(self._fastapi_app, feat, self._features)
        await self.app(scope, receive, send)


def _lazy_slots(app: "FastAPI") -> Mapping[str, BaseRoute | None]:
    return app.state.lazy_slots if hasattr(app.state, "lazy_slots") else MappingProxyType({})


def reserve_lazy_slot(app: "FastAPI", name: str, features: tuple[LazyFeature, ...] = LAZY_FEATURES) -> None:
    """Record the route the feature's router used to be included after, so its routes
    are spliced back in there once it loads and keep the same precedence. Anchoring on
    the route rather than its index survives later reordering of the table."""
    feat: Final = next(f for f in features if f.name == name)
    anchor: Final = app.router.routes[-1] if app.router.routes else None
    app.state.lazy_slots = MappingProxyType({**_lazy_slots(app), feat.module_path: anchor})


def _slot_index(routes: Sequence[BaseRoute], anchor: BaseRoute | None) -> int:
    if anchor is None:
        return 0
    return next((i + 1 for i, route in enumerate(routes) if route is anchor), len(routes))


def _eager_route_wins(app: "FastAPI", feat: LazyFeature, scope: Scope) -> bool:
    """Routes ahead of a feature's reserved slot beat its routes in Starlette's scan,
    so a request one of them fully matches never needs the feature loaded."""
    slots: Final = _lazy_slots(app)
    if feat.module_path not in slots:
        return False
    ahead: Final = app.router.routes[: _slot_index(app.router.routes, slots[feat.module_path])]
    return any(route.matches(scope)[0] is Match.FULL for route in ahead)


def _in_registry_order(
    routes: Sequence[BaseRoute],
    lazy_routes: Mapping[str, tuple[BaseRoute, ...]],
    features: tuple[LazyFeature, ...],
    slots: Mapping[str, BaseRoute | None],
) -> tuple[BaseRoute, ...]:
    """Lazy routers land in registry order, not first-request order, so overlapping
    paths (/openai/{endpoint:path} vs /openai/v1/realtime/calls) resolve the same
    way no matter which feature a deployment happens to hit first. Features with a
    reserved slot go back where they were eagerly included; the rest follow every
    eager route."""
    rank: Final = MappingProxyType({f.module_path: i for i, f in enumerate(features)})
    modules: Final = tuple(sorted(lazy_routes, key=lambda m: rank.get(m, len(rank))))
    lazy_ids: Final = frozenset(id(route) for module_path in modules for route in lazy_routes[module_path])
    eager: Final = tuple(route for route in routes if id(route) not in lazy_ids)

    def slot_of(module_path: str) -> int:
        return _slot_index(eager, slots[module_path]) if module_path in slots else len(eager)

    return tuple(
        route
        for index in range(len(eager) + 1)
        for route in (
            *(r for module_path in modules if slot_of(module_path) == index for r in lazy_routes[module_path]),
            *eager[index : index + 1],
        )
    )


async def _force_load(app: "FastAPI", feat: LazyFeature, features: tuple[LazyFeature, ...] = LAZY_FEATURES) -> bool:
    """Import + register a lazy feature exactly once per (app, module).
    Shared by the middleware and the /lazy/warm endpoint."""
    if not hasattr(app.state, "lazy_loaded"):
        app.state.lazy_loaded = set()
        app.state.lazy_locks = {}
    lock: Final = app.state.lazy_locks.setdefault(feat.module_path, asyncio.Lock())
    async with lock:
        if feat.module_path in app.state.lazy_loaded:
            return False
        try:
            # Import on a thread (heavy modules take 1-3 s). register_fn
            # mutates app.router.routes, so it stays on the loop thread.
            loop: Final = asyncio.get_running_loop()
            module: Final = await loop.run_in_executor(None, importlib.import_module, feat.module_path)
            before: Final = len(app.router.routes)
            feat.register_fn(app, module)
            previous: Final[Mapping[str, tuple[BaseRoute, ...]]] = (
                app.state.lazy_routes if hasattr(app.state, "lazy_routes") else MappingProxyType({})
            )
            lazy_routes: Final[Mapping[str, tuple[BaseRoute, ...]]] = MappingProxyType(
                {**previous, feat.module_path: tuple(app.router.routes[before:])}
            )
            app.state.lazy_routes = lazy_routes  # rebind-ok: the app owns the record of which routes each feature added
            app.router.routes[:] = hot_routes_first(  # rebind-ok: the app owns its route table
                _in_registry_order(app.router.routes, lazy_routes, features, _lazy_slots(app))
            )
            app.state.lazy_loaded.add(feat.module_path)
            app.openapi_schema = None
            verbose_proxy_logger.info(
                "Lazy-loaded optional feature %r (module: %s)",
                feat.name,
                feat.module_path,
            )
            return True
        except Exception as exc:
            # Mark loaded anyway so we don't retry on every request.
            app.state.lazy_loaded.add(feat.module_path)
            verbose_proxy_logger.warning(
                "Failed to lazy-load optional feature %r (module: %s): %s. "
                "This feature's endpoints will return 404 until restart.",
                feat.name,
                feat.module_path,
                exc,
            )
            return False


def attach_lazy_features(app: "FastAPI") -> None:
    app.include_router(_make_warmup_router(app))
    app.add_middleware(LazyFeatureMiddleware, fastapi_app=app)


def _make_warmup_router(app: "FastAPI") -> "APIRouter":
    """POST /lazy/warm/{name}: load a feature and return its partial openapi
    so the Swagger plugin can merge in-place without a full /openapi.json refetch.
    Requires auth — anyone who can hit the proxy can already trigger the same
    imports by sending a real request to a feature's prefix, but gating this
    debug endpoint avoids unauthenticated callers forcing the import chain."""
    from fastapi import APIRouter, Depends, HTTPException
    from fastapi.openapi.utils import get_openapi

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    router: Final = APIRouter()

    @router.post(
        "/lazy/warm/{name}",
        include_in_schema=False,
        dependencies=[Depends(user_api_key_auth)],
    )
    async def warm(name: str):
        feat: Final = next((f for f in LAZY_FEATURES if f.name == name), None)
        if feat is None:
            raise HTTPException(404, f"unknown lazy feature: {name}")
        if feat.persistent_swagger_stub:
            return {"stub_path": None, "paths": {}, "components": {"schemas": {}}}

        await _force_load(app, feat)

        feat_routes: Final = [r for r in app.routes if feat.matches(getattr(r, "path", ""))]
        full: Final = get_openapi(title=app.title, version=app.version, routes=feat_routes)
        # Force all operations under one tag so they group under a single Swagger
        # section — many lazy modules tag routes inconsistently.
        for path_ops in full.get("paths", {}).values():
            for op in path_ops.values():
                if isinstance(op, dict):
                    op["tags"] = [feat.name]
        return {
            "stub_path": feat.path_prefixes[0],
            "paths": full.get("paths", {}),
            "components": {"schemas": full.get("components", {}).get("schemas", {})},
        }

    return router


def loaded_lazy_modules(app: "FastAPI") -> frozenset[str]:
    """The set of lazy feature modules whose routers are actually registered
    on this app (tracked by _force_load), empty before the middleware ever ran.
    sys.modules is the wrong signal: boot code imports several feature modules
    (mcp_management, cloudzero, vantage, config_overrides) without mounting
    their routers, and their stubs must still be injected."""
    loaded: Final = getattr(app.state, "lazy_loaded", None)
    if not isinstance(loaded, set):
        return frozenset()
    return frozenset(m for m in loaded if isinstance(m, str))


def inject_lazy_stubs(
    schema: dict,
    loaded_modules: AbstractSet[str],
    features: tuple[LazyFeature, ...] = LAZY_FEATURES,
) -> dict:
    """Inject openapi entries for features not in loaded_modules. Uses the
    snapshot file when available (full route info), otherwise falls back to a
    single placeholder per feature. Any failure logs and returns the schema
    unchanged so /openapi.json never 500s on a cosmetic injection bug."""
    try:
        from litellm.proxy._lazy_openapi_snapshot import load_snapshot

        snapshot: Final = load_snapshot()
        paths: Final = schema.setdefault("paths", {})
        schemas: Final = schema.setdefault("components", {}).setdefault("schemas", {})

        for feat in features:
            if feat.module_path in loaded_modules and not feat.persistent_swagger_stub:
                continue

            fragment = (snapshot or {}).get(feat.name)
            if fragment:
                for p, ops in fragment.get("paths", {}).items():
                    paths.setdefault(p, ops)
                for name, sch in fragment.get("components", {}).get("schemas", {}).items():
                    schemas.setdefault(name, sch)
                continue

            prefix = feat.path_prefixes[0]
            if prefix in paths:
                continue
            paths[prefix] = {
                "get": {
                    "tags": [feat.name],
                    "summary": feat.name,
                    "responses": {"200": {"description": "OK"}},
                }
            }
    except Exception as exc:
        verbose_proxy_logger.warning("inject_lazy_stubs failed: %s", exc)
    return schema


def lazy_tag_to_prefix() -> dict[str, str]:
    """feature.name -> first prefix, used by the Swagger warmup JS plugin.
    Returns empty when the snapshot is loaded — the plugin is unnecessary
    because /openapi.json already has full route info."""
    from litellm.proxy._lazy_openapi_snapshot import load_snapshot

    if load_snapshot():
        return {}
    return {feat.name: feat.path_prefixes[0] for feat in LAZY_FEATURES if not feat.persistent_swagger_stub}
