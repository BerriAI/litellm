"""Coverage test for the gateway / backend component allowlists.

The componentization scaffold splits the proxy FastAPI app into two runtime
components by trimming the route table inside a wrapped lifespan context:

  gateway.main  -> only paths matched by gateway/routes/allowlist.py
  backend.main  -> only paths matched by backend/routes/allowlist.py

If either allowlist drops a path that was reachable on the monolithic app,
clients hitting that path on the corresponding pod get a 404. This test
guarantees that the union of the two trimmed route sets equals the full set
of routes on the proxy app — i.e. no endpoint is dropped on the floor.

The union-coverage test reproduces the same predicate that ``gateway/main.py``
and ``backend/main.py`` use, without importing them. The component modules wrap
the shared ``app.router.lifespan_context``; importing them in the test process
would chain wrappers and corrupt the snapshot. The gateway Mount tests below
import the real ``gateway.main._is_gateway_route`` instead, undoing both of the
module's import-time side effects: the lifespan wrapper is restored right after
the import, and the DATABASE_* env vars are popped for its duration because
``gateway.main`` runs ``DatabaseURLSettings.from_env().apply_to_env()`` at
import (which raises on a non-postgres ``DATABASE_URL`` scheme and can mint an
RDS IAM token when ``IAM_TOKEN_DB_AUTH`` is set).
"""

import json
import os
import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import partial
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI, HTTPException
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient
from starlette.types import Lifespan

# Importing ``litellm.proxy.proxy_server`` runs its module-level setup, which
# reads ``DATABASE_URL`` (Prisma) and ``LITELLM_MASTER_KEY``. Tier-zero CI
# runners don't set these. We pin throwaway values before the import so the
# test never depends on a live database or master key, then restore the prior
# environment so the throwaway values don't leak into sibling tests sharing the
# xdist worker (a leaked non-postgres ``DATABASE_URL`` makes DB-backed tests
# treat a phantom database as available instead of skipping).
_THROWAWAY_ENV = {
    "DATABASE_URL": "sqlite:///:memory:",
    "LITELLM_MASTER_KEY": "sk-test-component-allowlist",
}
_PRE_EXISTING_ENV = {key: os.environ.get(key) for key in _THROWAWAY_ENV}
for _key, _value in _THROWAWAY_ENV.items():
    os.environ.setdefault(_key, _value)

from prometheus_client import make_asgi_app

# gateway/ and backend/ live at the repo root, not inside litellm/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.routes.allowlist import BACKEND_MOUNT_PATHS
from gateway.routes.allowlist import GATEWAY_MOUNT_PATHS
from litellm.proxy import tracing_endpoints
from litellm.proxy._lazy_features import LazyFeature, attach_lazy_features
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization_dependencies import get_log_team_lookup
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.rust_bridge.trace.storage import ClickHouseStorage
from litellm.tracing import Tenant, TraceReceiver
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

for _key, _previous in _PRE_EXISTING_ENV.items():
    if _previous is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _previous

_DB_ENV_KEYS = (
    "DATABASE_URL",
    "DIRECT_URL",
    "DATABASE_URL_READ_REPLICA",
    "DATABASE_HOST",
    "DATABASE_HOST_READ_REPLICA",
    "DATABASE_PASSWORD",
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
)
_PRE_DB_ENV = {_key: os.environ.pop(_key, None) for _key in _DB_ENV_KEYS}
_PRE_COMPONENT_LIFESPAN = app.router.lifespan_context
from gateway.main import _gateway_lifespan, _is_gateway_route

app.router.lifespan_context = _PRE_COMPONENT_LIFESPAN
from backend.main import _backend_lifespan

app.router.lifespan_context = _PRE_COMPONENT_LIFESPAN
for _key, _previous in _PRE_DB_ENV.items():
    if _previous is not None:
        os.environ[_key] = _previous


_COVERAGE_PROBE: Final = """
import json, os, sys
sys.path.insert(0, os.environ["LITELLM_COMPONENT_ALLOWLIST_REPO_ROOT"])
from starlette.routing import Mount
from backend.routes.allowlist import BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES
from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES
from litellm.proxy._lazy_features import loaded_lazy_modules
from litellm.proxy.proxy_server import app

all_paths = {
    r.path for r in app.router.routes
    if not isinstance(r, Mount) and getattr(r, "path", None) is not None
}


def covered(exact, prefixes):
    return {p for p in all_paths if p in exact or any(p.startswith(x) for x in prefixes)}


json.dump({
    "lazy_loaded": sorted(loaded_lazy_modules(app)),
    "route_count": len(all_paths),
    "uncovered": sorted(all_paths - (
        covered(GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES)
        | covered(BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES)
    )),
}, sys.stdout)
"""


@pytest.mark.parametrize(
    "component_lifespan", (None, _gateway_lifespan, _backend_lifespan), ids=("proxy", "gateway", "backend")
)
@pytest.mark.parametrize("eager", (False, True), ids=("lazy", "eager"))
@pytest.mark.parametrize("state_kind", ("enabled", "disabled", "stateless"))
def test_composed_lifespan_preserves_request_state_and_teardown(
    monkeypatch: pytest.MonkeyPatch,
    component_lifespan: Lifespan[Starlette] | None,
    eager: bool,
    state_kind: Literal["enabled", "disabled", "stateless"],
) -> None:
    monkeypatch.setenv("LITELLM_DISABLE_LAZY_ROUTES", str(eager).lower())
    receiver: Final = object()
    resource: Final = object()
    state: Final[Mapping[str, object]] = {
        "tracing_receiver": receiver if state_kind == "enabled" else None,
        "other_resource": resource,
    }
    events: Final[list[str]] = []  # mutable-ok: observe startup, requests and teardown across the ASGI boundary

    async def trace_state(request: Request) -> JSONResponse:
        events.append("request")
        assert events[0] == "startup" and "shutdown" not in events
        assert getattr(request.state, "other_resource", None) is (resource if state_kind != "stateless" else None)
        assert getattr(request.state, "tracing_receiver", None) is (receiver if state_kind == "enabled" else None)
        return JSONResponse({"keys": sorted(request.scope["state"])})

    def register_trace_route(application: Starlette, module: object) -> None:
        application.router.routes.append(Route("/v1/traces", trace_state))

    @asynccontextmanager
    async def stateful_lifespan(application: Starlette) -> AsyncGenerator[Mapping[str, object], None]:
        events.append("startup")
        application.router.routes.append(Route("/not-a-component-route", trace_state))
        try:
            yield state
        finally:
            events.append("shutdown")

    @asynccontextmanager
    async def stateless_lifespan(application: Starlette) -> AsyncGenerator[None, None]:
        async with stateful_lifespan(application):
            yield

    application: Final = type(app)(lifespan=stateless_lifespan if state_kind == "stateless" else stateful_lifespan)
    feature: Final = LazyFeature("traces", __name__, ("/v1/traces",), register_fn=register_trace_route)
    attach_lazy_features(application, (feature,))
    if component_lifespan is not None:
        application.router.lifespan_context = partial(component_lifespan, lifespan=application.router.lifespan_context)

    with TestClient(application) as client:
        response: Final = client.get("/v1/traces")
        assert response.status_code == 200, response.text
        assert response.json() == {"keys": [] if state_kind == "stateless" else sorted(state)}
        filtered: Final = client.get("/not-a-component-route")
        assert filtered.status_code == (200 if component_lifespan is None else 404), filtered.text
        assert events == (["startup", "request", "request"] if component_lifespan is None else ["startup", "request"])
    assert events == (
        ["startup", "request", "request", "shutdown"] if component_lifespan is None else ["startup", "request", "shutdown"]
    )


@pytest.mark.parametrize(
    "component_lifespan", (None, _gateway_lifespan, _backend_lifespan), ids=("proxy", "gateway", "backend")
)
@pytest.mark.parametrize("eager", (False, True), ids=("lazy", "eager"))
@pytest.mark.parametrize("phase", ("startup", "shutdown"))
def test_composed_lifespan_propagates_lifecycle_failures(
    monkeypatch: pytest.MonkeyPatch, component_lifespan: Lifespan[Starlette] | None, eager: bool, phase: str
) -> None:
    monkeypatch.setenv("LITELLM_DISABLE_LAZY_ROUTES", str(eager).lower())
    failure: Final = RuntimeError(f"{phase} failed")
    events: Final[list[str]] = []  # mutable-ok: observe lifecycle events across the ASGI boundary

    @asynccontextmanager
    async def inner_lifespan(application: Starlette) -> AsyncGenerator[Mapping[str, object], None]:
        events.append("startup")
        if phase == "startup":
            raise failure
        yield {}
        events.append("shutdown")
        raise failure

    application: Final = type(app)(lifespan=inner_lifespan)
    attach_lazy_features(application, ())
    if component_lifespan is not None:
        application.router.lifespan_context = partial(component_lifespan, lifespan=application.router.lifespan_context)

    with pytest.raises(RuntimeError) as caught:
        with TestClient(application):
            events.append("serving")
    assert caught.value is failure
    assert events == (["startup"] if phase == "startup" else ["startup", "serving", "shutdown"])


@pytest.mark.parametrize(
    "component_lifespan", (_gateway_lifespan, _backend_lifespan), ids=("gateway", "backend")
)
@pytest.mark.parametrize("endpoint", ("/v1/traces", "/v1/logs"), ids=("traces", "logs"))
def test_otlp_ingest_routes_authenticate_and_isolate_tenants_on_each_component(
    component_lifespan: Lifespan[Starlette], endpoint: str
) -> None:
    application: Final = FastAPI()
    application.include_router(tracing_endpoints.router)
    storage: Final = MagicMock(spec=ClickHouseStorage)
    storage.ingest = AsyncMock(return_value=1)
    application.dependency_overrides[tracing_endpoints.provide_receiver] = lambda: TraceReceiver(storage)

    async def lookup(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    application.dependency_overrides[get_log_team_lookup] = lambda: lookup

    def authenticate(request: Request) -> UserAPIKeyAuth:
        match request.headers.get("Authorization"):
            case "Bearer team-a-key":
                return UserAPIKeyAuth(
                    user_id="user-a",
                    token="hashed-a",
                    team_id="team-a",
                    org_id="org-a",
                    user_role=LitellmUserRoles.INTERNAL_USER,
                )
            case "Bearer team-b-key":
                return UserAPIKeyAuth(
                    user_id="user-b",
                    token="hashed-b",
                    team_id="team-b",
                    org_id="org-b",
                    user_role=LitellmUserRoles.INTERNAL_USER,
                )
            case _:
                raise HTTPException(status_code=401, detail="Invalid API key")

    application.dependency_overrides[user_api_key_auth] = authenticate
    application.router.lifespan_context = partial(
        component_lifespan, lifespan=application.router.lifespan_context
    )

    body: Final = b'{"resourceLogs": []}'
    content_type: Final = "application/json"
    with TestClient(application) as client:
        unauthenticated: Final = client.post(endpoint, content=body, headers={"content-type": content_type})
        assert unauthenticated.status_code == 401, unauthenticated.text

        team_a: Final = client.post(
            endpoint,
            content=body,
            headers={"Authorization": "Bearer team-a-key", "content-type": content_type},
        )
        assert team_a.status_code == 200, team_a.text

        team_b: Final = client.post(
            endpoint,
            content=body,
            headers={"Authorization": "Bearer team-b-key", "content-type": content_type},
        )
        assert team_b.status_code == 200, team_b.text

    tenant_a: Final = Tenant(team_id="team-a", api_key_hash="hashed-a", org_id="org-a", user_id="user-a")
    tenant_b: Final = Tenant(team_id="team-b", api_key_hash="hashed-b", org_id="org-b", user_id="user-b")
    assert storage.ingest.await_args_list == [
        call(body, content_type, tenant_a, endpoint == "/v1/logs"),
        call(body, content_type, tenant_b, endpoint == "/v1/logs"),
    ]


def test_gateway_plus_backend_covers_full_app():
    """Every route on the proxy app must be served by gateway or backend.

    ``gateway.main`` and ``backend.main`` trim the route table once, inside the
    lifespan, so the set this has to cover is the one registered at startup. A
    lazy feature appends its router on demand, after that trim, and whether a
    sibling test in the same xdist worker has triggered one is not something
    this test can control. Measuring in a fresh interpreter is what makes the
    route table deterministic; nothing is subtracted, so every route the trim
    will actually see stays in the assertion.
    """
    env: Final = {**os.environ, "LITELLM_COMPONENT_ALLOWLIST_REPO_ROOT": _REPO_ROOT}
    for key, value in _THROWAWAY_ENV.items():
        env.setdefault(key, value)

    probe: Final = run_child_interpreter(_COVERAGE_PROBE, env=env, timeout=90)
    assert probe.returncode == 0, f"route probe failed:\n{probe.stderr}"
    report: Final = json.loads(probe.stdout)

    assert not report["lazy_loaded"], (
        "route probe was not pristine; it loaded lazy features "
        f"{report['lazy_loaded']}, so its route table is not the startup one"
    )
    assert report["route_count"] > 100, (
        f"route probe only saw {report['route_count']} routes, so an empty "
        "uncovered set would not mean anything"
    )

    uncovered: Final = report["uncovered"]
    assert not uncovered, (
        f"{len(uncovered)} route(s) are not exposed on either component. "
        f"Update gateway/routes/allowlist.py or backend/routes/allowlist.py to cover:\n  "
        + "\n  ".join(uncovered)
    )


def test_backend_mount_paths_defined():
    """BACKEND_MOUNT_PATHS constant must exist and be a frozenset."""
    assert isinstance(BACKEND_MOUNT_PATHS, frozenset), \
        f"BACKEND_MOUNT_PATHS must be a frozenset, got {type(BACKEND_MOUNT_PATHS)}"
    assert len(BACKEND_MOUNT_PATHS) > 0, \
        "BACKEND_MOUNT_PATHS must contain at least one Mount path"


def test_swagger_mount_in_backend_allowlist():
    """The /swagger Mount must be in BACKEND_MOUNT_PATHS."""
    assert "/swagger" in BACKEND_MOUNT_PATHS, \
        "/swagger Mount path must be in BACKEND_MOUNT_PATHS"


def test_backend_keeps_swagger_mount():
    """Verify that Mounts in BACKEND_MOUNT_PATHS are kept on the backend."""
    backend_mounts = {
        getattr(r, "path")
        for r in app.router.routes
        if isinstance(r, Mount) and getattr(r, "path", None) in BACKEND_MOUNT_PATHS
    }
    assert "/swagger" in backend_mounts, \
        "/swagger Mount is expected on the proxy app and should be in BACKEND_MOUNT_PATHS"


def test_backend_drops_non_allowlisted_mounts():
    """Verify that Mounts NOT in BACKEND_MOUNT_PATHS would be dropped from backend."""
    all_mounts = {
        getattr(r, "path")
        for r in app.router.routes
        if isinstance(r, Mount) and getattr(r, "path", None) is not None
    }
    non_backend_mounts = all_mounts - BACKEND_MOUNT_PATHS

    assert len(non_backend_mounts) > 0, \
        "Expected at least one non-backend Mount (e.g., /ui, /_next) to verify filtering logic"
    for mount_path in non_backend_mounts:
        assert mount_path not in BACKEND_MOUNT_PATHS, \
            f"Mount {mount_path} should not be in BACKEND_MOUNT_PATHS"


def test_gateway_mount_paths_defined():
    """GATEWAY_MOUNT_PATHS constant must exist and expose /metrics."""
    assert isinstance(GATEWAY_MOUNT_PATHS, frozenset), \
        f"GATEWAY_MOUNT_PATHS must be a frozenset, got {type(GATEWAY_MOUNT_PATHS)}"
    assert "/metrics" in GATEWAY_MOUNT_PATHS, \
        "/metrics Mount path must be in GATEWAY_MOUNT_PATHS"


def test_gateway_trim_keeps_metrics_mount():
    """The Prometheus /metrics Mount must survive the gateway route trim.

    Regression test for https://github.com/BerriAI/litellm/issues/30291:
    ``_is_gateway_route`` used to reject every Mount before the allowlist
    check, so the /metrics Mount registered by
    ``PrometheusLogger._mount_metrics_endpoint()`` was dropped at startup and
    the gateway returned 404 on /metrics.
    """
    metrics_mount = Mount("/metrics", app=make_asgi_app())
    routes = [*app.router.routes, metrics_mount]
    trimmed = [r for r in routes if _is_gateway_route(r)]
    assert metrics_mount in trimmed, \
        "/metrics Mount must survive the gateway route trim"


def test_gateway_drops_ui_and_swagger_mounts():
    """UI static and swagger Mounts must still be trimmed from the gateway."""
    for path in ("/ui", "/_next", "/litellm-asset-prefix/_next", "/swagger"):
        assert not _is_gateway_route(Mount(path, app=make_asgi_app())), \
            f"Mount {path} must not be served by the gateway"


def test_gateway_keeps_memory_summary_and_trims_the_other_debug_routes():
    """The gateway serves /debug/memory/summary, since the RSS that matters is the
    serving worker's and the memory regression e2e test reads it on every gateway
    replica; the heavier and mutating /debug/memory routes stay on the backend."""
    debug_memory_routes = {
        getattr(r, "path"): r for r in app.router.routes if str(getattr(r, "path", "")).startswith("/debug/memory/")
    }
    assert {"/debug/memory/summary", "/debug/memory/details", "/debug/memory/gc/configure"} <= set(debug_memory_routes)
    assert _is_gateway_route(debug_memory_routes["/debug/memory/summary"]), \
        "/debug/memory/summary must survive the gateway route trim"
    for path in ("/debug/memory/details", "/debug/memory/gc/configure"):
        assert not _is_gateway_route(debug_memory_routes[path]), f"{path} must not be served by the gateway"


def test_every_app_mount_is_assigned_to_a_component():
    """Every Mount on the proxy app must be consciously assigned to a component.

    A Mount must be kept by the gateway (GATEWAY_MOUNT_PATHS), kept by the
    backend (BACKEND_MOUNT_PATHS), or be a static mount served by the
    dedicated UI container. A Mount matching none of these is unreachable in
    a componentized deployment, which is exactly how the /metrics Mount was
    silently dropped.
    """
    ui_served_prefixes = ("/ui", "/_next", "/litellm-asset-prefix")
    mounts = [*app.router.routes, Mount("/metrics", app=make_asgi_app())]
    unassigned = {
        path
        for r in mounts
        if isinstance(r, Mount)
        and (path := getattr(r, "path", None)) is not None
        and path not in GATEWAY_MOUNT_PATHS
        and path not in BACKEND_MOUNT_PATHS
        and not path.startswith(ui_served_prefixes)
    }
    assert not unassigned, (
        f"{len(unassigned)} Mount(s) are not exposed on any component. "
        f"Add them to GATEWAY_MOUNT_PATHS, BACKEND_MOUNT_PATHS, or serve them "
        f"from the UI container:\n  " + "\n  ".join(sorted(unassigned))
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize(
    "component_lifespan", (None, _gateway_lifespan, _backend_lifespan), ids=("proxy", "gateway", "backend")
)
@pytest.mark.parametrize("enabled", (False, True))
def test_admin_mcp_survives_only_management_component_lifespans(
    monkeypatch: pytest.MonkeyPatch, component_lifespan: Lifespan[Starlette] | None, enabled: bool
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.admin_mcp import admin_mcp_lifespan

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", str(enabled).lower())
    for name in (
        "PROXY_BASE_URL", "LITELLM_MCP_PUBLIC_URL", "LITELLM_ADMIN_TOOLS", "LITELLM_ADMIN_READ_ONLY",
        "LITELLM_ADMIN_RESPONSE_VIEW", "LITELLM_ADMIN_SCHEMA_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[Mapping[str, object], None]:
        async with admin_mcp_lifespan(application):
            yield {"tracing_receiver": None}

    application: Final = FastAPI(lifespan=lifespan)

    @application.get("/user/info")
    async def user_info() -> dict[str, object]:
        return {"user_id": "admin", "user_info": {"user_id": "admin", "user_role": "proxy_admin"}}

    if component_lifespan is not None:
        application.router.lifespan_context = partial(component_lifespan, lifespan=application.router.lifespan_context)
    for _ in range(2):
        with TestClient(application, base_url="http://localhost:4000") as client:
            response: Final = client.post(
                "/admin/mcp",
                headers={"Authorization": "Bearer admin", "Accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26", "capabilities": {},
                        "clientInfo": {"name": "component-test", "version": "1"},
                    },
                },
            )
            expected_status: Final = 200 if enabled and component_lifespan is not _gateway_lifespan else 404
            assert response.status_code == expected_status, response.text
            if response.status_code == 200:
                assert response.json()["result"]["serverInfo"]["name"] == "litellm-admin-mcp"
