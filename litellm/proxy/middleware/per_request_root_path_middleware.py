"""
Per-request ``root_path`` resolution for a proxy fronting several
client-visible URL path prefixes from one deployment (``SERVER_ROOT_PATHS``).

``SERVER_ROOT_PATH`` is a single scalar stamped onto the app at startup, so
one deployment serves exactly one client-visible prefix: Starlette strips that
prefix during route matching and rebuilds it into ``request.base_url``, and a
request under any other prefix 404s before a handler runs. A pod fronting
several ingress path prefixes (e.g. ``/tenant-a/*`` and ``/tenant-b/*`` both
terminating at the same LiteLLM) is therefore forced into one deployment per
prefix purely to encode it.

This middleware generalizes the same mechanism to N prefixes: the operator
lists the prefixes the ingress preserves, and the one matching an incoming
request becomes that request's ``scope["root_path"]``. Everything downstream
is unchanged Starlette behavior — route matching strips ``root_path``
(``starlette._utils.get_route_path``) so routes stay registered
root-relative, and ``request.base_url`` re-includes it, so every emitted URL
(the MCP OAuth discovery documents' RFC 9728 §3 ``resource`` and the 401
challenges' ``resource_metadata`` among them) lands under the prefix the
client actually called. A path under no configured prefix passes through
untouched: it routes root-relative exactly as today, and 404s otherwise.

Opt-in by design: LiteLLM cannot detect whether the proxy in front strips or
preserves the client-visible prefix. Setting ``SERVER_ROOT_PATHS`` is the
operator affirming the prefix reaches the pod intact (path-based ALB /
ingress routing); a topology that rewrites the path away must keep using the
scalar ``SERVER_ROOT_PATH`` / path-carrying ``PROXY_BASE_URL``.
"""

import os
from collections.abc import Sequence
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

SERVER_ROOT_PATHS_ENV: Final = "SERVER_ROOT_PATHS"


def normalize_root_paths(raw_paths: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize configured prefixes: strip whitespace and trailing
    slashes, dedupe, order longest-first so the most-specific nested prefix
    wins at match time. Entries without a leading ``/`` and the bare root are
    dropped with a warning — a typo'd prefix silently matching nothing would
    surface only as unexplained 404s at request time."""
    kept: list[str] = []
    for entry in raw_paths:
        candidate = entry.strip()
        if not candidate:
            continue
        if not candidate.startswith("/"):
            verbose_proxy_logger.warning(
                "%s entry %r does not start with '/' and will be ignored.",
                SERVER_ROOT_PATHS_ENV,
                entry,
            )
            continue
        candidate = candidate.rstrip("/")
        if not candidate:
            verbose_proxy_logger.warning(
                "%s entry %r is the bare root and will be ignored; a root-mounted deployment needs no entry.",
                SERVER_ROOT_PATHS_ENV,
                entry,
            )
            continue
        if candidate not in kept:
            kept.append(candidate)
    return tuple(sorted(kept, key=len, reverse=True))


def get_server_root_paths() -> tuple[str, ...]:
    """The normalized ``SERVER_ROOT_PATHS`` prefixes, empty when unset."""
    configured: Final = os.getenv(SERVER_ROOT_PATHS_ENV, "")
    if not configured.strip():
        return ()
    return normalize_root_paths(configured.split(","))


class PerRequestRootPathMiddleware:
    """ASGI middleware that sets ``scope["root_path"]`` to the configured
    prefix matching the request path on a whole-segment boundary, leaving
    ``scope["path"]`` untouched (Starlette strips ``root_path`` from the path
    at route-match time; stripping here too would double-strip).

    Must be the outermost middleware (added last): inner middlewares and the
    router need ``root_path`` already resolved — LazyFeatureMiddleware strips
    it before feature prefix matching, Starlette strips it during route
    matching. A matched prefix overrides a scalar ``root_path``
    (``SERVER_ROOT_PATH``) for that request; combining both mechanisms is
    warned about at startup.
    """

    def __init__(self, app: ASGIApp, root_paths: Sequence[str]) -> None:
        self.app = app
        self.root_paths: Final = normalize_root_paths(root_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path: Final = scope.get("path", "")
            for prefix in self.root_paths:
                if path == prefix or path.startswith(prefix + "/"):
                    scope["root_path"] = prefix
                    break
        await self.app(scope, receive, send)
