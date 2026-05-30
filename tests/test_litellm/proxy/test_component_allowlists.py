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
# test never depends on a live database or master key.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("LITELLM_MASTER_KEY", "sk-test-component-allowlist")

from fastapi.routing import Mount

# gateway/ and backend/ live at the repo root, not inside litellm/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.routes.allowlist import BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES
from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES
from litellm.proxy.proxy_server import app


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
