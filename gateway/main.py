"""Gateway entrypoint.

Reuses the existing FastAPI app from `litellm.proxy.proxy_server` and trims its
route table to just the LLM data-plane surface. The trim is purely additive —
no existing module is modified, the full app continues to work via the legacy
entrypoint (`litellm.proxy.proxy_server:app`).

`LITELLM_GATEWAY_WORKLOAD` narrows the surface further to one workload
(``llm``, ``mcp`` or ``agent``) so each can run in its own container. The
default ``all`` keeps the whole gateway surface.

Run with:
    uvicorn gateway.main:app --host 0.0.0.0 --port 4000
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from starlette.applications import Starlette
from starlette.routing import BaseRoute, Mount, Route, WebSocketRoute

# Assemble DATABASE_URL (+ DATABASE_URL_READ_REPLICA) from the discrete
# DATABASE_* env vars before proxy_server imports spin up Prisma. Handles
# both IAM (mint a token) and password auth, writer and reader. The standard
# CLI flow does this in proxy_cli.py; we bypass proxy_cli by uvicorn'ing the
# app directly, so without this Prisma initializes with the placeholder URL
# and every DB-needing endpoint returns "Database not connected".
from litellm.proxy.db.db_url_settings import DatabaseURLSettings

DatabaseURLSettings.from_env().apply_to_env()

from gateway.routes.allowlist import (
    GATEWAY_WORKLOAD_ENV_VAR,
    GatewayWorkload,
    gateway_serves_mount,
    gateway_serves_path,
    parse_gateway_workload,
)
from litellm._logging import verbose_proxy_logger
from litellm.proxy._lazy_features import LazyFeature, disable_lazy_features
from litellm.proxy.proxy_server import app

GATEWAY_WORKLOAD: Final = parse_gateway_workload(os.environ.get(GATEWAY_WORKLOAD_ENV_VAR))


def _is_gateway_route(route: BaseRoute, workload: GatewayWorkload = GATEWAY_WORKLOAD) -> bool:
    """Keep the route on the gateway if its path is in the workload's data-plane surface.

    Prometheus registers /metrics as a Mount (``app.mount("/metrics", make_asgi_app())``),
    so Mounts are matched against the workload's mount paths instead of being dropped with
    the UI static mounts.
    """
    if isinstance(route, Mount):
        return gateway_serves_mount(workload, route.path)
    if isinstance(route, (Route, WebSocketRoute)):
        return gateway_serves_path(workload, route.path)
    return False


def _keeps_lazy_feature(feat: LazyFeature, workload: GatewayWorkload = GATEWAY_WORKLOAD) -> bool:
    return workload == "all" or any(gateway_serves_path(workload, f"{prefix}/") for prefix in feat.path_prefixes)


# Wrap proxy_server's existing lifespan so the route trim runs *after* its
# startup hooks (and any plugin code those hooks load) have had a chance to
# register routes. A module-load filter would miss routes added during
# startup; running inside the lifespan, after the inner __aenter__, catches
# them while still completing before uvicorn opens the listener. Lazy
# features register on first request, so the ones outside this workload are
# marked loaded here and can never bring their routes back.
_proxy_lifespan: Final = app.router.lifespan_context


@asynccontextmanager
async def _gateway_lifespan(_: Starlette) -> AsyncGenerator[None]:
    async with _proxy_lifespan(app):
        app.router.routes = [r for r in app.router.routes if _is_gateway_route(r)]
        disabled: Final = disable_lazy_features(app, _keeps_lazy_feature)
        verbose_proxy_logger.info(
            "gateway workload=%s serving %d routes, %d lazy features disabled",
            GATEWAY_WORKLOAD,
            len(app.router.routes),
            len(disabled),
        )
        yield


app.router.lifespan_context = _gateway_lifespan
