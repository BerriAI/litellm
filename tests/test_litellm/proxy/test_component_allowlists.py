"""Coverage test for the gateway / backend component allowlists.

The componentization scaffold splits the proxy FastAPI app into two runtime
components by trimming the route table inside a wrapped lifespan context:

  gateway.main  -> only paths matched by gateway/routes/allowlist.py
  backend.main  -> only paths matched by backend/routes/allowlist.py

If either allowlist drops a path that was reachable on the monolithic app,
clients hitting that path on the corresponding pod get a 404. This test
guarantees that the union of the two trimmed route sets equals the full set
of routes on the proxy app — i.e. no endpoint is dropped on the floor.

The test reproduces the same predicate that ``gateway/main.py`` and
``backend/main.py`` use, without importing them. The component modules wrap
the shared ``app.router.lifespan_context``; importing them in the test process
would chain wrappers and corrupt the snapshot.
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

# gateway/ and backend/ live at the repo root, not inside litellm/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.routes.allowlist import (
    BACKEND_EXACT_PATHS,
    BACKEND_MOUNT_PATHS,
    BACKEND_PATH_PREFIXES,
)
from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES
from litellm.proxy.proxy_server import app

for _key, _previous in _PRE_EXISTING_ENV.items():
    if _previous is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _previous


def _component_paths(routes, exact_paths, path_prefixes) -> set[str]:
    """Reproduce ``gateway.main._is_gateway_route`` / ``backend.main._is_backend_route``."""
    out: set[str] = set()
    for route in routes:
        if isinstance(route, Mount):
            continue
        path = getattr(route, "path", None)
        if path is None:
            continue
        if path in exact_paths or any(path.startswith(p) for p in path_prefixes):
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
        app.router.routes, BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES
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
