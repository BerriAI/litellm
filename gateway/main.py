"""Gateway entrypoint.

Reuses the existing FastAPI app from `litellm.proxy.proxy_server` and trims its
route table to just the LLM data-plane surface. The trim is purely additive —
no existing module is modified, the full app continues to work via the legacy
entrypoint (`litellm.proxy.proxy_server:app`).

Run with:
    uvicorn gateway.main:app --host 0.0.0.0 --port 4000
"""

from contextlib import asynccontextmanager

from fastapi.routing import Mount

# Assemble DATABASE_URL (+ DATABASE_URL_READ_REPLICA) from the discrete
# DATABASE_* env vars before proxy_server imports spin up Prisma. Handles
# both IAM (mint a token) and password auth, writer and reader. The standard
# CLI flow does this in proxy_cli.py; we bypass proxy_cli by uvicorn'ing the
# app directly, so without this Prisma initializes with the placeholder URL
# and every DB-needing endpoint returns "Database not connected".
from litellm.proxy.db.db_url_settings import DatabaseURLSettings

DatabaseURLSettings.from_env().apply_to_env()

from litellm.proxy.proxy_server import app

from gateway.routes.allowlist import (
    GATEWAY_EXACT_PATHS,
    GATEWAY_MOUNT_PATHS,
    GATEWAY_PATH_PREFIXES,
)


def _is_gateway_route(route) -> bool:
    """Keep the route on the gateway if its path is in the LLM data-plane surface.

    Prometheus registers /metrics as a Mount (``app.mount("/metrics", make_asgi_app())``),
    so Mounts are matched against GATEWAY_MOUNT_PATHS instead of being dropped with
    the UI static mounts.
    """
    path = getattr(route, "path", None)
    if path is None:
        return False
    if isinstance(route, Mount):
        return path in GATEWAY_MOUNT_PATHS
    if path in GATEWAY_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in GATEWAY_PATH_PREFIXES)


# Wrap proxy_server's existing lifespan so the route trim runs *after* its
# startup hooks (and any plugin code those hooks load) have had a chance to
# register routes. A module-load filter would miss routes added during
# startup; running inside the lifespan, after the inner __aenter__, catches
# them while still completing before uvicorn opens the listener.
_proxy_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _gateway_lifespan(app_):
    async with _proxy_lifespan(app_):
        app_.router.routes = [r for r in app_.router.routes if _is_gateway_route(r)]
        yield


app.router.lifespan_context = _gateway_lifespan
