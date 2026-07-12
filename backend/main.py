"""UI backend entrypoint.

Reuses the existing FastAPI app from `litellm.proxy.proxy_server` and trims its
route table to just the management/admin surface used by the dashboard. Purely
additive — no existing module is modified.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 4001
"""

from contextlib import asynccontextmanager

from fastapi.routing import Mount

# See gateway/main.py for why we assemble DATABASE_URL(s) here before
# importing proxy_server.
from litellm.proxy.db.db_url_settings import DatabaseURLSettings

DatabaseURLSettings.from_env().apply_to_env()

from litellm.proxy.proxy_server import app

from backend.routes.allowlist import (
    BACKEND_EXACT_PATHS,
    BACKEND_MOUNT_PATHS,
    BACKEND_PATH_PREFIXES,
)


def _is_backend_route(route) -> bool:
    """Keep the route on the backend if its path is in the management surface."""
    path = getattr(route, "path", None)
    if path is None:
        return False
    if isinstance(route, Mount):
        # The dashboard UI static mounts are served by the dedicated UI container.
        # Only Mounts in the backend allowlist (e.g. swagger docs) remain on backend.
        return path in BACKEND_MOUNT_PATHS
    if path in BACKEND_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in BACKEND_PATH_PREFIXES)


# See gateway/main.py for why the trim runs inside the lifespan instead of at
# module scope.
_proxy_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _backend_lifespan(app_):
    async with _proxy_lifespan(app_):
        app_.router.routes = [r for r in app_.router.routes if _is_backend_route(r)]
        yield


app.router.lifespan_context = _backend_lifespan
