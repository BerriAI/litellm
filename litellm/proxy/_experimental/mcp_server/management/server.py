"""Lifecycle and HTTP admission for the built-in management MCP endpoint.

The server is a stateless Streamable HTTP MCP server exposing the management
tools built from the proxy's own OpenAPI spec in catalog.py. Admission runs
LiteLLM ``user_api_key_auth`` once per HTTP request against an isolated
request view and stashes the trusted request slices in
a ContextVar that the dispatcher reads per tool call. Nothing is keyed by
mcp-session-id: each request carries its own caller context.
"""

from collections.abc import Awaitable, Mapping
from contextvars import ContextVar
from typing import Final, Protocol, cast

import mcp.types as mcp_types
from fastapi import FastAPI, HTTPException
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message, Receive, Scope, Send
from typing_extensions import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.management.catalog import (
    ManagementCatalog,
    build_catalog,
)
from litellm.proxy._experimental.mcp_server.management.dispatcher import (
    Dispatch,
    ManagementRequestContext,
    build_management_asgi_app,
    build_management_client,
    call_tool,
    error_result,
    set_dispatch,
)
from litellm.proxy._types import UserAPIKeyAuth

MANAGEMENT_MCP_PATH: Final = "/litellm-management/mcp"

_request_context: ContextVar[ManagementRequestContext | None] = ContextVar(
    "litellm_management_mcp_request_context", default=None
)


class _EmptyArguments(TypedDict, total=False):
    pass


_empty_arguments: Final[_EmptyArguments] = {}


class _AsgiHandleFn(Protocol):
    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]: ...


class _McpStreamBridge(Protocol):
    def __call__(self, handle_fn: _AsgiHandleFn, scope: dict[str, object], receive: Receive) -> Awaitable[Response]: ...


async def _on_list_tools(ctx: object, params: object) -> mcp_types.ListToolsResult:
    server: Final = _active_server
    tools: Final[list[mcp_types.Tool]] = (
        [tool.mcp_tool for tool in server.catalog.tools.values()]  # mutable-ok: SDK result field is a list
        if server is not None
        else []  # mutable-ok: SDK result field is a list
    )
    return mcp_types.ListToolsResult(tools=tools)


async def _on_call_tool(ctx: object, params: mcp_types.CallToolRequestParams) -> mcp_types.CallToolResult:
    caller_ctx: Final = _request_context.get()
    if caller_ctx is None:
        return error_result(500, "no caller context")
    tool_arguments: Final = params.arguments if params.arguments is not None else _empty_arguments
    return await call_tool(params.name, tool_arguments, caller_ctx)


class ManagementMCPServer:
    def __init__(self, catalog: ManagementCatalog) -> None:
        self.catalog: Final = catalog
        self.server: Final = Server(
            "litellm-management",
            on_list_tools=_on_list_tools,
            on_call_tool=_on_call_tool,
        )
        self.manager: Final = StreamableHTTPSessionManager(
            app=self.server,
            json_response=True,
            stateless=True,
        )
        self._manager_cm = None

    async def start(self) -> None:
        self._manager_cm = self.manager.run()
        await self._manager_cm.__aenter__()

    async def close(self) -> None:
        if self._manager_cm is not None:
            await self._manager_cm.__aexit__(None, None, None)
            self._manager_cm = None


_active_server: ManagementMCPServer | None = None


def management_mcp_enabled() -> bool:
    return _active_server is not None


async def start_management_mcp_server(general_settings: Mapping[str, object], app: FastAPI) -> None:
    global _active_server
    if not general_settings.get("enable_management_mcp"):
        return
    catalog: Final = build_catalog(app.openapi())
    internal_app: Final = build_management_asgi_app(app)
    set_dispatch(
        Dispatch(
            catalog=catalog,
            internal_app=internal_app,
            http_client=build_management_client(internal_app),
        )
    )
    server: Final = ManagementMCPServer(catalog)
    await server.start()
    _active_server = server
    verbose_proxy_logger.info(
        "Management MCP endpoint enabled at %s with %s tools", MANAGEMENT_MCP_PATH, len(catalog.tools)
    )


async def shutdown_management_mcp_server() -> None:
    global _active_server
    server: Final = _active_server
    _active_server = None
    dispatch: Final = set_dispatch(None)
    try:
        if server is not None:
            await server.close()
    finally:
        if dispatch is not None:
            await dispatch.http_client.aclose()


def _caller_credential_header(request: Request) -> str:
    from litellm.proxy.proxy_server import general_settings

    configured_header: Final = general_settings.get("litellm_key_header_name")
    if isinstance(configured_header, str):
        return configured_header.lower()
    return "x-litellm-api-key" if request.headers.get("x-litellm-api-key") else "authorization"


def _caller_api_key(request: Request) -> str:
    return request.headers.get(_caller_credential_header(request)) or ""


async def _authenticate_admission(request: Request) -> UserAPIKeyAuth:
    """Run LiteLLM admission auth on an isolated, empty-body view of the request.

    The real request body stays untouched for the MCP transport, while auth sees
    a fixed management path so route checks classify the endpoint itself rather
    than whichever tool a JSON-RPC payload names.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    scope = dict(request.scope)  # mutable-ok: admission sees its own scope copy
    empty_state: Final[dict[str, object]] = {}  # mutable-ok: ASGI state must be a mutable dict
    scope["state"] = empty_state
    scope["path"] = MANAGEMENT_MCP_PATH
    scope["raw_path"] = MANAGEMENT_MCP_PATH.encode()

    async def _empty_receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}  # mutable-ok: ASGI messages are dicts

    admission_request: Final = Request(scope, _empty_receive)
    credential: Final = _caller_api_key(request)
    if not credential:
        raise HTTPException(status_code=401, detail="Management MCP requires a LiteLLM credential")
    return await user_api_key_auth(
        request=admission_request,
        api_key=credential,
        custom_litellm_key_header=request.headers.get("x-litellm-api-key") or None,
    )


async def handle_management_mcp_request(request: Request) -> Response:
    active: Final = _active_server
    if active is None:
        return Response(status_code=404)

    from litellm.proxy import proxy_server

    origin: Final = request.headers.get("origin")
    if origin is not None and (origin in ("", "null", "*") or origin not in proxy_server.origins):
        return Response(status_code=403)

    await _authenticate_admission(request)

    credential_header: Final = _caller_credential_header(request)
    caller_ctx: Final = ManagementRequestContext(
        credential_header=credential_header,
        credential_value=_caller_api_key(request),
        client=(request.client.host, request.client.port) if request.client else None,
        forwarded_for=request.headers.get("x-forwarded-for"),
        root_path=str(request.scope.get("root_path") or ""),
        litellm_changed_by=request.headers.get("litellm-changed-by"),
        request_id=request.headers.get("x-request-id"),
    )
    token: Final = _request_context.set(caller_ctx)
    try:
        stream_bridge: Final[_McpStreamBridge] = (
            cast(  # cast-ok: pins the helper's untyped signature to the Protocol the caller uses
                _McpStreamBridge,
                proxy_server._stream_mcp_asgi_response,  # pyright: ignore[reportPrivateUsage]  # same bridging helper the aggregate MCP routes use
            )
        )
        return await stream_bridge(
            active.manager.handle_request,
            dict(request.scope),  # mutable-ok: the SDK transport receives its own mutable scope copy
            request.receive,
        )
    finally:
        _request_context.reset(token)
