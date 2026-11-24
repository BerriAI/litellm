"""Honour a front end's ``X-Forwarded-Host`` when building absolute URLs.

Starlette builds ``request.base_url`` from the ``Host`` header when it is
present, and from ``scope["server"]`` otherwise. A front end that forwards a
``Host`` value which is not a bare ``host[:port]`` authority — e.g.
``Host: example.com/base`` or ``Host: example.com/?x=`` — therefore makes the
gateway emit URLs pointing at its own bound socket: the ``GET /ui`` redirect,
the SSO callback and the MCP OAuth discovery ``resource`` all become
browser-unreachable. On versions where route resolution reads
``request.url.path`` the same collision also shifts the matched route, which is
the ``Host: evil.com/?x=`` authentication-bypass shape.

The client-visible authority is exactly what ``X-Forwarded-Host`` carries, and
it is the conventional remedy (Django ``USE_X_FORWARDED_HOST``), but neither
litellm nor uvicorn's ``ProxyHeadersMiddleware`` reads it.

Opt-in via ``USE_X_FORWARDED_HOST=1``: the header is only trusted when a front
end is known to set it. Must be added last so it is the outermost middleware —
the rewritten ``host`` header and ``scope["server"]`` have to be in place
before any inner middleware or the router builds a URL.
"""

import os
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

USE_XFH_ENV: Final = "USE_X_FORWARDED_HOST"


def use_x_forwarded_host() -> bool:
    return os.getenv(USE_XFH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _authority(value: str) -> str:
    """First entry of a comma list, path/query stripped, whitespace removed."""
    first = value.split(",")[0].strip()
    for sep in ("/", "?", "#"):
        cut = first.find(sep)
        if cut != -1:
            first = first[:cut]
    return first.strip()


def _split_port(authority: str, default_port: int) -> tuple[str, int]:
    """Split ``host[:port]``, keeping bracketed IPv6 intact."""
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            return authority, default_port
        host = authority[1:end]
        rest = authority[end + 1 :]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except ValueError:
                return host, default_port
        return host, default_port
    if authority.count(":") == 1:
        host, port = authority.rsplit(":", 1)
        try:
            return host, int(port)
        except ValueError:
            return host, default_port
    return authority, default_port


class ForwardedHostMiddleware:
    """Rewrite the ``host`` header and ``scope["server"]`` from
    ``X-Forwarded-Host`` so absolute URLs carry the client-visible authority."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            raw: str | None = None
            for name, value in scope["headers"]:
                if name == b"x-forwarded-host":
                    raw = value.decode("latin1")
            if raw is not None:
                authority = _authority(raw)
                if authority:
                    server = scope.get("server")
                    default_port = server[1] if server else 80
                    host, port = _split_port(authority, default_port)
                    scope["server"] = (host, port)
                    scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"host"] + [
                        (b"host", authority.encode("latin1"))
                    ]
        await self.app(scope, receive, send)
