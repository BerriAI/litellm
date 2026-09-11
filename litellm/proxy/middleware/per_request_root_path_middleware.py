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
from contextvars import ContextVar
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

SERVER_ROOT_PATHS_ENV: Final = "SERVER_ROOT_PATHS"

# The effective ``root_path`` for the currently-handled request. Populated by
# ``PerRequestRootPathMiddleware`` from the (possibly-mutated) scope so code
# that emits URLs off the request path — the 401 challenges' resource_metadata
# and ``get_custom_url``'s SSO callbacks among them — can pick up the prefix
# the client actually called without threading scope through every call site.
# ``None`` means "middleware did not run" (the ``SERVER_ROOT_PATHS`` env is
# unset, so no per-request prefix exists); readers fall back to the scalar
# ``SERVER_ROOT_PATH`` in that case, which matches the pre-middleware behavior.
_request_root_path_var: Final[ContextVar[str | None]] = ContextVar("_request_root_path_var", default=None)


def get_request_root_path() -> str:
    """Return the effective ``root_path`` for the current request.

    Reads the value ``PerRequestRootPathMiddleware`` stashed for this request;
    falls back through :func:`~litellm.proxy.utils.get_server_root_path` (i.e.
    the ``SERVER_ROOT_PATH`` env) when the middleware did not run — the
    scalar-only deployment. Delegating to the existing helper keeps every
    existing ``monkeypatch.setattr("litellm.proxy.utils.get_server_root_path"``
    test override working, and keeps a single source of truth for the scalar.
    """
    value: Final = _request_root_path_var.get()
    if value is not None:
        return value
    # Lazy import: utils.py imports this module (via the lazy import inside
    # get_custom_url), so a top-level import would build a cycle at load time.
    from litellm.proxy.utils import get_server_root_path  # noqa: PLC0415  # lazy import breaks a two-way dep

    return get_server_root_path()


def normalize_root_paths(raw_paths: Sequence[str]) -> tuple[str, ...]:
    """Strip whitespace and trailing slashes, dedupe, order longest-first;
    warn and drop entries missing a leading ``/`` and the bare root."""
    kept: Final[list[str]] = []  # mutable-ok: local accumulator; escapes only as a tuple
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
                    scope["root_path"] = prefix  # rebind-ok: ASGI middleware contract; Router and base_url read it
                    break
            # Stash the effective root_path (matched prefix, or the scope's
            # existing value when nothing matched — i.e. FastAPI's scalar
            # SERVER_ROOT_PATH) so code that emits URLs off the request path
            # picks the same prefix the router will resolve the request under.
            token: Final = _request_root_path_var.set(str(scope.get("root_path", "")))
            try:
                await self.app(scope, receive, send)
            finally:
                _request_root_path_var.reset(token)
            return
        await self.app(scope, receive, send)
