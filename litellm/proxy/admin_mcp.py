import os
import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Final
from urllib.parse import urlsplit

from fastapi import FastAPI
from pydantic import TypeAdapter
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.constants import STANDARD_CUSTOMER_ID_HEADERS
from litellm.proxy._types import SpecialHeaders
from litellm.proxy.middleware.admission_control_middleware import ADMISSION_LEASE_SCOPE_KEY

_REQUEST_HEADERS: Final = frozenset(
    {
        b"authorization",
        b"litellm-changed-by",
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
_CREDENTIAL_HEADERS: Final = frozenset(
    name.encode("ascii") for name in SpecialHeaders.litellm_credential_header_names()
)
_RESERVED_KEY_HEADERS: Final = (
    frozenset(
        {
            "host",
            "origin",
            "user-agent",
            "forwarded",
            "te",
            "trailer",
            "upgrade",
            "x-litellm-user-id",
            "x-litellm-team-id",
            "x-litellm-trace-id",
            "traceparent",
            "tracestate",
        }
    )
    | frozenset(STANDARD_CUSTOMER_ID_HEADERS)
    | frozenset(name.decode("ascii") for name in _REQUEST_HEADERS - {b"authorization"})
)
_SETTINGS: Final = TypeAdapter(Mapping[str, object])
_IDENTITY_MAPPINGS: Final = TypeAdapter(tuple[dict[str, object], ...] | dict[str, object] | None)
_OAUTH_MAPPINGS: Final = TypeAdapter(dict[str, str])


def _configured_key_header() -> bytes | None:
    from litellm.proxy.proxy_server import general_settings

    settings: Final = _SETTINGS.validate_python(general_settings)
    name: Final = settings.get("litellm_key_header_name")
    if name is not None and (not isinstance(name, str) or re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", name) is None):
        raise ValueError("Hosted admin MCP requires a valid litellm_key_header_name")
    raw_mappings: Final = _IDENTITY_MAPPINGS.validate_python(settings.get("user_header_mappings"))
    mappings: Final = (raw_mappings,) if isinstance(raw_mappings, dict) else raw_mappings or ()
    mapped_names: Final = tuple(mapping.get("header_name") for mapping in mappings)
    oauth_names: Final = (
        tuple(_OAUTH_MAPPINGS.validate_python(settings.get("oauth2_config_mappings") or {}).values())
        if settings.get("enable_oauth2_proxy_auth") is True
        else ()
    )
    policy_names: Final = (
        settings.get("user_header_name"),
        settings.get("mcp_client_id_header"),
        *mapped_names,
        *oauth_names,
    )
    policy_headers: Final = frozenset(value.lower() for value in policy_names if isinstance(value, str))
    overwritten_headers: Final = frozenset(value.decode("ascii") for value in _REQUEST_HEADERS | _CREDENTIAL_HEADERS)
    if policy_headers & overwritten_headers:
        raise ValueError("Hosted admin MCP cannot overwrite configured identity headers")
    if name is None:
        return None
    normalized: Final = name.lower()
    if normalized in _RESERVED_KEY_HEADERS | policy_headers or normalized.startswith("x-forwarded-"):
        raise ValueError("Hosted admin MCP litellm_key_header_name cannot replace a transport, audit, or policy header")
    return normalized.encode("ascii")


def _require_enterprise_license() -> None:
    from litellm.proxy.utils import require_enterprise_license

    require_enterprise_license("Hosted admin MCP")


class _CallerContext:
    def __init__(self, app: ASGIApp, caller: ContextVar[Request]) -> None:
        self.app = app
        self.caller = caller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        _require_enterprise_license()
        token: Final = self.caller.set(Request(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            self.caller.reset(token)


@asynccontextmanager
async def admin_mcp_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    enabled: Final = os.environ.get("LITELLM_ENABLE_ADMIN_MCP", "false").strip().lower()
    if enabled in ("false", "0", "off", "no", ""):
        yield
        return
    if enabled not in ("true", "1", "on", "yes"):
        raise ValueError("LITELLM_ENABLE_ADMIN_MCP must be true or false")
    _require_enterprise_license()
    _configured_key_header()

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

    configured_url: Final = os.environ.get("LITELLM_MCP_PUBLIC_URL") or os.environ.get("PROXY_BASE_URL", "")
    public_url: Final = urlsplit(configured_url)
    config: Final = Config(
        base_url="http://localhost",
        public_url=f"{public_url.scheme}://{public_url.netloc}" if public_url.netloc else configured_url,
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
        configured_header: Final = _configured_key_header()
        excluded: Final = (
            _REQUEST_HEADERS
            | _CREDENTIAL_HEADERS
            | (frozenset({configured_header}) if configured_header is not None else frozenset())
        )
        caller_headers: Final = tuple(pair for pair in request.headers.raw if pair[0].lower() not in excluded)
        generated_headers: Final = Headers(scope=scope)
        api_headers: Final = tuple(pair for pair in generated_headers.raw if pair[0] in _REQUEST_HEADERS)
        configured_auth: Final = (
            ((configured_header, generated_headers["authorization"].encode("ascii")),)
            if configured_header is not None and configured_header != b"authorization"
            else ()
        )
        headers: Final = list(caller_headers + api_headers + configured_auth)
        gateway_scope: Final[Scope] = {
            **scope,
            "client": request.client,
            "scheme": request.url.scheme,
            "headers": headers,
            ADMISSION_LEASE_SCOPE_KEY: request.scope.get(ADMISSION_LEASE_SCOPE_KEY),
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
                app.router.routes[:] = [existing for existing in app.router.routes if existing is not route]
