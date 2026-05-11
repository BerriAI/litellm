"""UI backend entrypoint.

Reuses the existing FastAPI app from `litellm.proxy.proxy_server` and trims its
route table to just the management/admin surface used by the dashboard. Purely
additive — no existing module is modified.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 4001
"""

from fastapi.routing import Mount

# See gateway/main.py for why we mint IAM-signed DATABASE_URL(s) here before
# importing proxy_server.
from litellm.proxy.auth.rds_iam_token import init_iam_db_url_from_env

init_iam_db_url_from_env()

from litellm.proxy.proxy_server import app

from backend.routes.allowlist import BACKEND_EXACT_PATHS, BACKEND_PATH_PREFIXES


def _is_backend_route(route) -> bool:
    """Keep the route on the backend if its path is in the management surface."""
    path = getattr(route, "path", None)
    if path is None:
        return False
    if isinstance(route, Mount):
        # Static UI mounts are served by the dedicated UI container, not here.
        return False
    if path in BACKEND_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in BACKEND_PATH_PREFIXES)


app.router.routes = [r for r in app.router.routes if _is_backend_route(r)]
