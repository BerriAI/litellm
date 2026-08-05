"""Per-request ``root_path`` resolution from ``SERVER_ROOT_PATHS``.

``SERVER_ROOT_PATH`` is a startup scalar, so one deployment serves exactly one
client-visible URL path prefix; a request under any other prefix 404s before a
handler runs. When the ingress preserves several prefixes into one pod (e.g.
``/tenant-a/*`` and ``/tenant-b/*``), the matched prefix becomes that
request's ``scope["root_path"]`` instead: Starlette strips it during route
matching and rebuilds it into ``request.base_url``, so every emitted URL —
the MCP OAuth discovery ``resource`` (RFC 9728 §3) and the 401 challenges'
``resource_metadata`` among them — lands under the prefix the client called.
Opt-in: with ``SERVER_ROOT_PATHS`` unset the middleware is not added at all.
"""

import os
from collections.abc import Sequence
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

SERVER_ROOT_PATHS_ENV: Final = "SERVER_ROOT_PATHS"


def normalize_root_paths(raw_paths: Sequence[str]) -> tuple[str, ...]:
    """Strip whitespace and trailing slashes, dedupe, order longest-first;
    warn and drop entries missing a leading ``/`` and the bare root."""
    kept: list[str] = []  # mutable-ok: local accumulator; escapes only as a tuple
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
    """Sets ``scope["root_path"]`` to the configured prefix matching the
    request path on a whole-segment boundary. ``scope["path"]`` is left
    untouched (Starlette strips ``root_path`` at route-match time). Must be
    the outermost middleware so inner middlewares and the router see the
    resolved value; a matched prefix overrides a scalar ``SERVER_ROOT_PATH``
    for that request.
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
