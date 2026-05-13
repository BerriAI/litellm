"""
Lazy registration for optional feature routers. Each LAZY_FEATURES entry
imports its module only on the first request matching its path prefix,
saving ~700 MB at idle for deployments that don't use these features.
First hit pays the import cost (1-3 s for heavy modules); /openapi.json
omits each feature's routes until the feature is warmed.
"""

import asyncio
import importlib
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Tuple

from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from fastapi import APIRouter, FastAPI


def _include_router(attr_name: str = "router") -> Callable[["FastAPI", object], None]:
    def _register(app: "FastAPI", module: object) -> None:
        app.include_router(getattr(module, attr_name))

    return _register


def _mount_app(
    prefix: str, attr_name: str = "app"
) -> Callable[["FastAPI", object], None]:
    def _register(app: "FastAPI", module: object) -> None:
        app.mount(path=prefix, app=getattr(module, attr_name))

    return _register


@dataclass(frozen=True)
class LazyFeature:
    name: str
    module_path: str
    path_prefixes: Tuple[str, ...]
    register_fn: Callable[["FastAPI", object], None] = field(
        default_factory=lambda: _include_router("router")
    )
    # For routes whose path has a leading parameter (e.g. /{server}/authorize)
    # — startswith can't match those, so the matcher also checks endswith.
    path_suffixes: Tuple[str, ...] = ()
    # Keep the stub injected even after load — for mounted ASGI sub-apps
    # whose routes don't appear in the parent app's openapi spec.
    persistent_swagger_stub: bool = False


LAZY_FEATURES: Tuple[LazyFeature, ...] = (
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
        name="a2a",
        module_path="litellm.proxy.agent_endpoints.a2a_endpoints",
        path_prefixes=("/a2a", "/v1/a2a"),
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
            "/authorize",
            "/token",
            "/callback",
            "/register",
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
        features: Tuple[LazyFeature, ...] = LAZY_FEATURES,
    ):
        self.app = app
        self._fastapi_app = fastapi_app
        self._features = features
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
        if scope["type"] in ("http", "websocket") and len(self._loaded) < len(
            self._features
        ):
            # Lazy import to avoid pulling proxy.utils into this module's
            # import graph (proxy_server imports both).
            from litellm.proxy.utils import get_server_root_path

            path = scope.get("path", "")
            # Strip SERVER_ROOT_PATH so prefix matching works under a server
            # root path. Without this, requests like /api/v1/policies/... never
            # match the registered prefixes (/policies/...) and lazy features
            # stay unloaded — every endpoint under them returns 404. The
            # `+ "/"` boundary prevents false-positive matches (e.g. /apiv2
            # against root /api). If the path doesn't start with the prefix
            # (e.g. a reverse proxy already stripped it), we leave it alone.
            root_path = get_server_root_path().rstrip("/")
            if root_path and path.startswith(root_path + "/"):
                path = path[len(root_path) :]
            for feat in self._features:
                if feat.module_path in self._loaded:
                    continue
                if any(path.startswith(p) for p in feat.path_prefixes) or any(
                    path.endswith(s) for s in feat.path_suffixes
                ):
                    await _force_load(self._fastapi_app, feat)
        await self.app(scope, receive, send)


async def _force_load(app: "FastAPI", feat: LazyFeature) -> bool:
    """Import + register a lazy feature exactly once per (app, module).
    Shared by the middleware and the /lazy/warm endpoint."""
    if not hasattr(app.state, "lazy_loaded"):
        app.state.lazy_loaded = set()
        app.state.lazy_locks = {}
    lock = app.state.lazy_locks.setdefault(feat.module_path, asyncio.Lock())
    async with lock:
        if feat.module_path in app.state.lazy_loaded:
            return False
        try:
            # Import on a thread (heavy modules take 1-3 s). register_fn
            # mutates app.router.routes, so it stays on the loop thread.
            loop = asyncio.get_running_loop()
            module = await loop.run_in_executor(
                None, importlib.import_module, feat.module_path
            )
            feat.register_fn(app, module)
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

    router = APIRouter()

    @router.post(
        "/lazy/warm/{name}",
        include_in_schema=False,
        dependencies=[Depends(user_api_key_auth)],
    )
    async def warm(name: str):
        feat = next((f for f in LAZY_FEATURES if f.name == name), None)
        if feat is None:
            raise HTTPException(404, f"unknown lazy feature: {name}")
        if feat.persistent_swagger_stub:
            return {"stub_path": None, "paths": {}, "components": {"schemas": {}}}

        await _force_load(app, feat)

        feat_routes = [
            r
            for r in app.routes
            if any(getattr(r, "path", "").startswith(p) for p in feat.path_prefixes)
        ]
        full = get_openapi(title=app.title, version=app.version, routes=feat_routes)
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


def inject_lazy_stubs(schema: Dict) -> Dict:
    """Inject openapi entries for unloaded features. Uses the snapshot file
    when available (full route info), otherwise falls back to a single
    placeholder per feature. Any failure logs and returns the schema unchanged
    so /openapi.json never 500s on a cosmetic injection bug."""
    try:
        from litellm.proxy._lazy_openapi_snapshot import load_snapshot

        snapshot = load_snapshot()
        paths = schema.setdefault("paths", {})
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})

        for feat in LAZY_FEATURES:
            if feat.module_path in sys.modules and not feat.persistent_swagger_stub:
                continue

            fragment = (snapshot or {}).get(feat.name)
            if fragment:
                for p, ops in fragment.get("paths", {}).items():
                    paths.setdefault(p, ops)
                for name, sch in (
                    fragment.get("components", {}).get("schemas", {}).items()
                ):
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


def lazy_tag_to_prefix() -> Dict[str, str]:
    """feature.name -> first prefix, used by the Swagger warmup JS plugin.
    Returns empty when the snapshot is loaded — the plugin is unnecessary
    because /openapi.json already has full route info."""
    from litellm.proxy._lazy_openapi_snapshot import load_snapshot

    if load_snapshot():
        return {}
    return {
        feat.name: feat.path_prefixes[0]
        for feat in LAZY_FEATURES
        if not feat.persistent_swagger_stub
    }
