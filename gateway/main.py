"""Gateway entrypoint.

Reuses the existing FastAPI app from `litellm.proxy.proxy_server` and trims its
route table to just the LLM data-plane surface. The trim is purely additive —
no existing module is modified, the full app continues to work via the legacy
entrypoint (`litellm.proxy.proxy_server:app`).

Run with:
    uvicorn gateway.main:app --host 0.0.0.0 --port 4000
"""

from fastapi.routing import Mount

# Mint RDS IAM tokens and assemble DATABASE_URL (+ DATABASE_URL_READ_REPLICA)
# before proxy_server imports spin up Prisma. The standard CLI flow does this
# in proxy_cli.py; we bypass proxy_cli by uvicorn'ing the app directly, so
# without this call Prisma initializes with the placeholder URL and every
# DB-needing endpoint returns "Database not connected". No-op when
# IAM_TOKEN_DB_AUTH is unset.
from litellm.proxy.auth.rds_iam_token import init_iam_db_url_from_env

init_iam_db_url_from_env()

from litellm.proxy.proxy_server import app

from gateway.routes.allowlist import GATEWAY_EXACT_PATHS, GATEWAY_PATH_PREFIXES


def _is_gateway_route(route) -> bool:
    """Keep the route on the gateway if its path is in the LLM data-plane surface."""
    path = getattr(route, "path", None)
    if path is None:
        return False
    if isinstance(route, Mount):
        # Gateway never serves the static UI or its asset bundles.
        return False
    if path in GATEWAY_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in GATEWAY_PATH_PREFIXES)


app.router.routes = [r for r in app.router.routes if _is_gateway_route(r)]
