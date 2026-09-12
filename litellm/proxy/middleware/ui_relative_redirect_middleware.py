"""Relativizes same-origin Location headers on redirects under the /ui mount.

Starlette's own trailing-slash redirect (``Router.app``) and ``StaticFiles``'s
directory-index redirect both build the ``Location`` header from
``scope["scheme"]``. That scheme is only "https" when uvicorn's
``ProxyHeadersMiddleware`` trusts the request's peer IP enough to honor
``X-Forwarded-Proto`` (``FORWARDED_ALLOW_IPS``, default loopback-only).
Behind a TLS-terminating load balancer whose peer IP isn't trusted, this
bakes the plaintext-hop scheme into an absolute ``http://...`` redirect,
which browsers refuse to follow off an https page (LIT-7455).

Rewriting a same-origin ``Location`` to be path-relative removes the scheme
dependency entirely: the browser resolves a relative ``Location`` against
the page's own (correct) scheme. Only ``Location`` values whose host matches
the request's own ``Host`` header are touched, so an intentionally
cross-origin redirect elsewhere in the app (e.g. SSO) is never affected.
"""

from typing import Final
from urllib.parse import SplitResult, urlsplit, urlunsplit

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware.route_path_utils import get_route_path

_REDIRECT_STATUS_CODES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})


def _request_host(scope: Scope) -> str | None:
    for key, value in scope["headers"]:
        if key == b"host":
            return value.decode("latin-1")
    return None


def _relativize_same_origin(location: str, host: str) -> str:
    parts: Final[SplitResult] = urlsplit(location)
    if not parts.scheme and not parts.netloc:
        return location  # already relative
    if parts.netloc.lower() != host.lower():
        return location  # cross-origin: leave untouched
    path: Final = parts.path or "/"
    if not path.startswith("/") or path.startswith("//"):
        # Not a normal absolute path (or would be interpreted as
        # protocol-relative once the scheme/netloc are dropped); safest to
        # leave the Location as-is rather than risk changing its target.
        return location
    return urlunsplit(("", "", path, parts.query, parts.fragment))


class UiRelativeRedirectMiddleware:
    """Rewrites same-origin Location headers on /ui responses to be path-relative."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route_path: Final = get_route_path(scope) if scope["type"] == "http" else ""
        if not (route_path == "/ui" or route_path.startswith("/ui/")):
            await self.app(scope, receive, send)
            return

        host: Final = _request_host(scope)

        async def send_with_relative_location(message: Message) -> None:
            is_redirect_start: Final = (
                message["type"] == "http.response.start" and message["status"] in _REDIRECT_STATUS_CODES
            )
            if host is not None and is_redirect_start:
                headers: Final = MutableHeaders(scope=message)
                location: Final = headers.get("location")
                if location is not None:
                    headers["location"] = _relativize_same_origin(location, host)
            await send(message)

        await self.app(scope, receive, send_with_relative_location)
