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

import os
import sys

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

from fastapi.routing import Mount
from prometheus_client import make_asgi_app

# gateway/ and backend/ live at the repo root, not inside litellm/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.routers import ADMIN_PREFIX
from backend.routes.allowlist import (
    BACKEND_EXACT_PATHS,
    BACKEND_MOUNT_PATHS,
    BACKEND_PATH_PREFIXES,
)
from gateway.routes.allowlist import (
    GATEWAY_EXACT_PATHS,
    GATEWAY_MOUNT_PATHS,
    GATEWAY_PATH_PREFIXES,
)
from litellm.proxy.proxy_server import app

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
)
_PRE_DB_ENV = {_key: os.environ.pop(_key, None) for _key in _DB_ENV_KEYS}
_PRE_COMPONENT_LIFESPAN = app.router.lifespan_context
from gateway.main import _is_gateway_route

app.router.lifespan_context = _PRE_COMPONENT_LIFESPAN
for _key, _previous in _PRE_DB_ENV.items():
    if _previous is not None:
        os.environ[_key] = _previous


def _component_paths(routes, exact_paths, path_prefixes, keep_prefixes=()) -> set[str]:
    """Reproduce ``gateway.main._is_gateway_route`` / ``backend.main._is_backend_route``."""
    out: set[str] = set()
    for route in routes:
        if isinstance(route, Mount):
            continue
        path = getattr(route, "path", None)
        if path is None:
            continue
        if any(path == p or path.startswith(p + "/") for p in keep_prefixes):
            out.add(path)
        elif path in exact_paths or any(path.startswith(p) for p in path_prefixes):
            out.add(path)
    return out


def test_gateway_plus_backend_covers_full_app():
    """Every route on the proxy app must be served by gateway or backend."""
    all_paths = {
        getattr(r, "path")
        for r in app.router.routes
        if not isinstance(r, Mount) and getattr(r, "path", None) is not None
    }
    gateway_paths = _component_paths(
        app.router.routes, GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES
    )
    backend_paths = _component_paths(
        app.router.routes,
        BACKEND_EXACT_PATHS,
        BACKEND_PATH_PREFIXES,
        keep_prefixes=(ADMIN_PREFIX,),
    )

    uncovered = all_paths - (gateway_paths | backend_paths)

    assert not uncovered, (
        f"{len(uncovered)} route(s) are not exposed on either component. "
        f"Update gateway/routes/allowlist.py or backend/routes/allowlist.py to cover:\n  "
        + "\n  ".join(sorted(uncovered))
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
