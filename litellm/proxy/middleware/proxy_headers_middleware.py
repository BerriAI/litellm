from collections.abc import Callable, Mapping
from typing import (
    Final,
    cast,  # noqa: TID251  # uvicorn and Starlette have incompatible ASGI protocol stubs
)

from pydantic import TypeAdapter
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn._types import ASGI3Application
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.network import (
    PROXY_HEADER_CONTEXT,
    ProxyHeaderContext,
    normalize_cidr_ranges,
    parse_trusted_proxy_ranges,
)

_CLIENT: Final = TypeAdapter[tuple[str, int] | None](tuple[str, int] | None)
_SCHEME: Final = TypeAdapter[str](str)


class TrustedProxyMiddleware:
    def __init__(self, app: ASGIApp, settings: Callable[[], Mapping[str, object]]) -> None:
        self.app = app
        self.settings = settings
        self._ranges: tuple[str, ...] = ()
        self._middleware: ProxyHeadersMiddleware | None = None
        self._warned_legacy = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        settings: Final = self.settings()
        configured: Final = settings.get("trusted_proxy_ranges") is not None
        ranges: Final = tuple(
            normalize_cidr_ranges(settings.get("trusted_proxy_ranges"), setting_name="trusted_proxy_ranges")
        )
        if self._middleware is None or ranges != self._ranges:
            hosts: Final = [
                str(network) for network in parse_trusted_proxy_ranges(ranges, setting_name="trusted_proxy_ranges")
            ]
            app: Final = cast(ASGI3Application, self.app)  # cast-ok: equivalent ASGI protocols use different stubs
            self._middleware = ProxyHeadersMiddleware(app, trusted_hosts=hosts)
            self._ranges = ranges
        if settings.get("use_x_forwarded_for") is True and not self._warned_legacy:
            verbose_proxy_logger.warning(
                "general_settings.use_x_forwarded_for is deprecated for client-IP resolution. "
                "Set trusted_proxy_ranges to your reverse-proxy CIDRs, or [] for direct connections. "
                "When trusted_proxy_ranges is unset, legacy raw X-Forwarded-For handling remains enabled. "
                "MCP continues to use its separate proxy settings."
            )
            self._warned_legacy = True

        client: Final = _CLIENT.validate_python(scope.get("client"))
        peer: Final = client[0] if client else None
        context: Final = ProxyHeaderContext(
            direct_client=client,
            direct_scheme=_SCHEME.validate_python(scope.get("scheme", "http" if scope["type"] == "http" else "ws")),
            configured=configured,
            via_trusted_proxy=peer in self._middleware.trusted_hosts,
        )
        forwarded_proto: Final = (
            Headers(raw=scope.get("headers", [])).get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
        )
        scheme: Final = (
            forwarded_proto
            if context.via_trusted_proxy and forwarded_proto in {"http", "https"}
            else context.direct_scheme
        )
        resolved_scope: Final = {
            **scope,
            "scheme": scheme.replace("http", "ws") if scope["type"] == "websocket" else scheme,
            PROXY_HEADER_CONTEXT: context,
        }
        handler: Final = cast(ASGIApp, self._middleware)  # cast-ok: equivalent ASGI protocol stubs
        await handler(resolved_scope, receive, send)
