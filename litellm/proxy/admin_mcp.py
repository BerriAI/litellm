import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Final
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

_REQUEST_HEADERS: Final = frozenset(
    {
        b"authorization",
        b"cookie",
        b"content-length",
        b"content-type",
        b"transfer-encoding",
        b"connection",
        b"accept",
        b"accept-encoding",
        b"mcp-protocol-version",
        b"mcp-session-id",
    }
)


class _CallerContext:
    def __init__(self, app: ASGIApp, caller: ContextVar[Request]) -> None:
        self.app = app
        self.caller = caller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token: Final = self.caller.set(Request(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            self.caller.reset(token)


@asynccontextmanager
async def admin_mcp_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    enabled: Final = os.environ.get("LITELLM_ENABLE_ADMIN_MCP", "false").strip().lower()
    if enabled in ("false", "0", ""):
        yield
        return
    if enabled not in ("true", "1"):
        raise ValueError("LITELLM_ENABLE_ADMIN_MCP must be true or false")

    try:
        import httpx2
        from litellm_admin_mcp.config import (  # pyright: ignore[reportMissingTypeStubs]  # upstream has no py.typed marker
            Config,
            env_bool,
        )
        from litellm_admin_mcp.gateway import (  # pyright: ignore[reportMissingTypeStubs]  # upstream has no py.typed marker
            Gateway,
        )
        from litellm_admin_mcp.server import (  # pyright: ignore[reportMissingTypeStubs]  # upstream has no py.typed marker
            create_http_app,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Admin MCP requires Python 3.12+ and the admin-mcp dependency group. "
            "Use a LiteLLM image that bundles it, or run uv sync --extra proxy --group admin-mcp."
        ) from exc

    from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager

    public_url: Final = urlsplit(os.environ.get("LITELLM_MCP_PUBLIC_URL") or os.environ.get("PROXY_BASE_URL", ""))
    config: Final = Config(
        base_url="http://localhost",
        public_url=f"{public_url.scheme}://{public_url.netloc}" if public_url.netloc else "",
        read_only=env_bool("LITELLM_ADMIN_READ_ONLY"),
        allowed_tools=frozenset(
            name.strip() for name in os.environ.get("LITELLM_ADMIN_TOOLS", "").split(",") if name.strip()
        ),
        response_view=os.environ.get("LITELLM_ADMIN_RESPONSE_VIEW", "full").strip(),
        schema_mode=os.environ.get("LITELLM_ADMIN_SCHEMA_MODE", "full").strip(),
    )
    caller: Final[ContextVar[Request]] = ContextVar("admin_mcp_caller")

    async def management_api(scope: Scope, receive: Receive, send: Send) -> None:
        request: Final = caller.get()
        caller_headers: Final = tuple(pair for pair in request.headers.raw if pair[0] not in _REQUEST_HEADERS)
        api_headers: Final = tuple(pair for pair in Headers(scope=scope).raw if pair[0] in _REQUEST_HEADERS)
        headers: Final = list(caller_headers + api_headers)  # mutable-ok: ASGI middleware modifies headers
        gateway_scope: Final[Scope] = {
            **scope,
            "client": request.client,
            "scheme": request.url.scheme,
            "headers": headers,
        }
        await app(gateway_scope, receive, send)

    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=management_api)) as client:
        admin_app: Final = create_http_app(Gateway(config, client))
        route: Final = Mount("/admin", app=_CallerContext(admin_app, caller), name="admin_mcp")
        async with admin_app.router.lifespan_context(admin_app):
            app.router.routes.insert(0, route)
            try:
                yield
            finally:
                GracefulShutdownManager.start_shutdown()
                try:
                    await GracefulShutdownManager.wait_for_drain()
                finally:
                    app.router.routes.remove(route)
