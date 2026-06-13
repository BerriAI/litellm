"""gateway — the transport edge: a raw ASGI app in front of the SDK.

``TransportGateway`` owns path parsing and identity publication, then hands the
streamable-HTTP request to the SDK's ``StreamableHTTPSessionManager`` unchanged.

Seams established here for later sections to fill without restructuring:
  - ``parse_route`` -> ``RouteTarget``: the routing seam (S2 consumes ``.server``).
  - ``resolve_subject`` (injected): the identity seam (S6 fills the body).
  - ``use_subject``: publishes the caller for downstream layers.

Tenant is not in the URL; it rides in on the resolved ``Subject``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from starlette.types import Receive, Scope, Send

from litellm.proxy.gateway.mcp.authn.context import use_subject
from litellm.proxy.gateway.mcp.authn.resolver import resolve_subject
from litellm.proxy.gateway.mcp.foundation import Subject
from litellm.proxy.gateway.mcp.transport.routes import parse_route


class _Dispatcher(Protocol):
    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None: ...


async def _send_404(send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 404,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": b"Not Found"})


def _http_request(scope: Scope) -> tuple[str, list[tuple[bytes, bytes]]] | None:
    if scope["type"] != "http":
        return None
    return cast(str, scope["path"]), cast("list[tuple[bytes, bytes]]", scope["headers"])


def _bearer_from_headers(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        raw = value.decode("latin-1").strip()
        if raw[:7].lower() == "bearer ":
            return raw[7:].strip() or None
        return None
    return None


class TransportGateway:
    def __init__(
        self,
        manager: _Dispatcher,
        resolve_subject: Callable[[str | None], Subject] = resolve_subject,
    ) -> None:
        self._manager = manager
        self._resolve_subject = resolve_subject

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return

        request = _http_request(scope)
        if request is None:
            return

        path, headers = request
        if parse_route(path) is None:
            await _send_404(send)
            return

        subject = self._resolve_subject(_bearer_from_headers(headers))
        with use_subject(subject):
            await self._manager.handle_request(scope, receive, send)
