"""Lifecycle and HTTP admission for the built-in management MCP endpoint.

The server is a stateless Streamable HTTP MCP server exposing the 10 tools in
catalog.py. Admission runs LiteLLM ``user_api_key_auth`` once per HTTP request
against an isolated request view, requires PROXY_ADMIN, and stashes the trusted
request slices in a ContextVar that the dispatcher reads per tool call. Nothing
is keyed by mcp-session-id: each request carries its own caller context.
"""

from collections.abc import Awaitable, Mapping
from contextvars import ContextVar
from typing import Final, Protocol, cast

import mcp.types as mcp_types
from fastapi import HTTPException
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message, Receive, Scope, Send
from typing_extensions import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.management.catalog import mcp_tools
from litellm.proxy._experimental.mcp_server.management.dispatcher import (
    ManagementRequestContext,
    call_tool,
    error_result,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

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
    tools: Final[list[mcp_types.Tool]] = list(mcp_tools())  # mutable-ok: SDK result field is a list
    return mcp_types.ListToolsResult(tools=tools)


async def _on_call_tool(ctx: object, params: mcp_types.CallToolRequestParams) -> mcp_types.CallToolResult:
    caller_ctx: Final = _request_context.get()
    if caller_ctx is None:
        return error_result(500, "no caller context")
    tool_arguments: Final = params.arguments if params.arguments is not None else _empty_arguments
    return await call_tool(params.name, tool_arguments, caller_ctx)


class ManagementMCPServer:
    def __init__(self) -> None:
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


async def start_management_mcp_server(general_settings: Mapping[str, object]) -> None:
    global _active_server
    if not general_settings.get("enable_management_mcp"):
        return
    server: Final = ManagementMCPServer()
    await server.start()
    _active_server = server
    verbose_proxy_logger.info("Management MCP endpoint enabled at %s", MANAGEMENT_MCP_PATH)


async def shutdown_management_mcp_server() -> None:
    global _active_server
    server: Final = _active_server
    _active_server = None
    if server is not None:
        await server.close()


def _caller_api_key(request: Request) -> str:
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    return MCPRequestHandler.get_litellm_api_key_from_headers(request.headers) or ""


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
    caller: Final = await user_api_key_auth(request=admission_request, api_key=_caller_api_key(request))
    if caller.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail=f"Management MCP endpoint requires a proxy admin key. Your role={caller.user_role}",
        )
    return caller


async def handle_management_mcp_request(request: Request) -> Response:
    active: Final = _active_server
    if active is None:
        return Response(status_code=404)

    await _authenticate_admission(request)

    caller_ctx: Final = ManagementRequestContext(
        raw_headers=tuple(request.scope.get("headers") or ()),
        client=request.scope.get("client"),
        scheme=str(request.scope.get("scheme") or "http"),
        server=request.scope.get("server"),
        root_path=str(request.scope.get("root_path") or ""),
        http_version=str(request.scope.get("http_version") or "1.1"),
        app=request.scope.get("app"),
        api_key=_caller_api_key(request),
        litellm_changed_by=request.headers.get("litellm-changed-by"),
    )
    token: Final = _request_context.set(caller_ctx)
    try:
        from litellm.proxy import proxy_server

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
