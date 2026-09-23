"""Dispatch management MCP tool calls through the proxy's own route table.

Each call is re-entered into a middleware-free ASGI view of the FastAPI app
(the real routes, real dependencies, real auth), so a tool call is observably
identical to the same REST request: same status, same JSON body, same error
shape. Only a fixed allowlist of headers crosses from the caller's request;
nothing tool-supplied can become a header or touch the URL host.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, cast
from urllib.parse import urlencode

import httpx
import mcp.types as mcp_types
from fastapi import FastAPI
from fastapi.middleware.asyncexitstack import AsyncExitStackMiddleware
from pydantic import TypeAdapter
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.types import ASGIApp, ExceptionHandler, Receive, Scope, Send
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.management.catalog import (
    ManagementCatalog,
    ManagementTool,
)
from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    _sanitize_path_parameter_value,  # pyright: ignore[reportPrivateUsage]  # the traversal guard the upstream OpenAPI tool caller already enforces
)
from litellm.proxy._lazy_features import LazyFeatureMiddleware

_HANDLER_TIMEOUT_SECONDS: Final = 30.0
_MAX_RESULT_BYTES: Final = 1024 * 1024
_ARGUMENT_SECTIONS: Final = frozenset({"path", "query", "body"})
_INTERNAL_BASE_URL: Final = "http://litellm-management"


class _WrappedResult(TypedDict):
    result: ReadOnly[object]


_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class ManagementRequestContext:
    """The trusted slices of the admission request, captured before dispatch.

    Only these fields can reach the internal REST request; tool arguments
    never touch headers, host, scheme, or auth material.
    """

    credential_header: str
    credential_value: str
    client: tuple[str, int] | None
    root_path: str
    litellm_changed_by: str | None
    request_id: str | None
    forwarded_for: str | None = None


@dataclass(frozen=True, slots=True)
class Dispatch:
    catalog: ManagementCatalog
    internal_app: ASGIApp
    http_client: httpx.AsyncClient


_active_dispatch: Dispatch | None = None
_call_context: Final = ContextVar[ManagementRequestContext | None]("management_mcp_call_context", default=None)


def build_management_client(internal_app: ASGIApp) -> httpx.AsyncClient:
    """One shared client over the internal ASGI transport for the server lifetime.

    The transport is fixed, so the per-call client tuple and root path are
    stamped onto the scope through ``_call_context`` instead of rebuilding a
    client per tool call (a shared client is also what the async-client budget
    checker requires).
    """

    async def scoped_app(scope: Scope, receive: Receive, send: Send) -> None:
        ctx: Final = _call_context.get()
        if ctx is not None:
            scope["client"] = ctx.client
            scope["root_path"] = ctx.root_path
        await internal_app(scope, receive, send)

    client: Final = get_async_httpx_client(
        llm_provider="management_mcp",
        params={  # mutable-ok: existing HTTP factory accepts a parameter dict
            "transport": httpx.ASGITransport(app=scoped_app),
            "follow_redirects": False,
        },
    ).client
    client.base_url = _INTERNAL_BASE_URL
    return client


def set_dispatch(dispatch: Dispatch | None) -> Dispatch | None:
    global _active_dispatch
    previous: Final = _active_dispatch
    _active_dispatch = dispatch
    return previous


def build_management_asgi_app(app: FastAPI) -> ASGIApp:
    """A middleware-free ASGI view over the real route table.

    Replicates ``Starlette.build_middleware_stack`` with none of
    ``app.user_middleware`` except ``LazyFeatureMiddleware``, which optional
    feature routers (e.g. access groups) need to mount on first hit. The
    exception handlers stay (ProxyException and ManagementProblem map to
    their real JSON bodies), while auth-relevant state is what
    ``user_api_key_auth`` writes itself, so nothing else is needed in
    ``scope["state"]``. ``scope["app"]`` stays the real app.
    """

    handler_map: Final = cast(  # cast-ok: Starlette stores handlers under Any keys
        Mapping[object, ExceptionHandler], app.exception_handlers
    )
    exception_handlers: Final = {  # mutable-ok: filtered handler map for ExceptionMiddleware
        key: handler for key, handler in handler_map.items() if key not in (500, Exception)
    }
    error_handler: Final = app.exception_handlers.get(Exception) or app.exception_handlers.get(500)
    core: Final = LazyFeatureMiddleware(
        ServerErrorMiddleware(
            AsyncExitStackMiddleware(ExceptionMiddleware(app.router, handlers=exception_handlers)),
            handler=error_handler,
        ),
        app,
    )

    async def management_app(scope: Scope, receive: Receive, send: Send) -> None:
        scope["app"] = app
        await core(scope, receive, send)

    return management_app


def _tool_result(
    payload_text: str, structured: dict[str, object] | _WrappedResult | None, is_error: bool
) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=payload_text)],  # mutable-ok: SDK content field is a list
        structured_content=structured,
        is_error=is_error,
    )


def error_result(status: int, detail: object) -> mcp_types.CallToolResult:
    payload: Final = json.dumps({"status": status, "detail": detail}, default=str)  # mutable-ok: one-shot JSON payload
    return _tool_result(
        payload,
        _OBJECT_DICT_ADAPTER.validate_python({"status": status, "detail": detail}),  # mutable-ok: one-shot JSON payload
        is_error=True,
    )


def _validated_sections(
    tool: ManagementTool, arguments: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object] | None] | mcp_types.CallToolResult:
    unknown: Final = tuple(key for key in arguments if key not in _ARGUMENT_SECTIONS)
    if unknown:
        sections_list: Final = list(unknown)  # mutable-ok: error message rendering
        return error_result(400, f"unexpected argument section(s) {sections_list} for tool '{tool.name}'")
    path_args: Final[object] = arguments.get("path") or {}  # mutable-ok: default empty JSON section
    query_args: Final[object] = arguments.get("query") or {}  # mutable-ok: default empty JSON section
    body_args: Final[object] = arguments.get("body")
    sections_to_check: Final[tuple[tuple[str, object], ...]] = (("path", path_args), ("query", query_args))
    for label, section in sections_to_check:
        if not isinstance(section, dict):
            return error_result(400, f"'{label}' arguments for tool '{tool.name}' must be an object")
    if body_args is not None and not isinstance(body_args, dict):
        return error_result(400, f"'body' arguments for tool '{tool.name}' must be an object")
    if body_args is not None and not tool.has_body:
        return error_result(400, f"tool '{tool.name}' does not accept a request body")
    return (
        _OBJECT_DICT_ADAPTER.validate_python(path_args),
        _OBJECT_DICT_ADAPTER.validate_python(query_args),
        _OBJECT_DICT_ADAPTER.validate_python(body_args) if body_args is not None else None,
    )


def _target_url(tool: ManagementTool, path_args: Mapping[str, object], query_args: Mapping[str, object]) -> str:
    for param_name in tool.path_param_names:
        if path_args.get(param_name) is None:
            raise ValueError(f"missing path parameter '{param_name}' for tool '{tool.name}'")
    path: Final = "/".join(
        _sanitize_path_parameter_value(path_args.get(segment[1:-1]), segment[1:-1])
        if segment.startswith("{") and segment.endswith("}")
        else segment
        for segment in tool.path_template.split("/")
    )
    pairs: Final = tuple(
        (key, item)
        for key, value in query_args.items()
        if value is not None
        for item in (
            cast(Sequence[object], value)  # cast-ok: JSON array query values
            if isinstance(value, (list, tuple))
            else (value,)
        )
    )
    query: Final = urlencode(pairs)
    return f"{path}?{query}" if query else path


def _forwarded_headers(ctx: ManagementRequestContext) -> dict[str, str]:
    headers: Final = {  # mutable-ok: header map assembled for httpx
        "content-type": "application/json",
        ctx.credential_header: ctx.credential_value,
    }
    if ctx.forwarded_for is not None:
        headers["x-forwarded-for"] = ctx.forwarded_for
    if ctx.litellm_changed_by is not None:
        headers["litellm-changed-by"] = ctx.litellm_changed_by
    if ctx.request_id is not None:
        headers["x-request-id"] = ctx.request_id
    return headers


async def _read_capped_body(response: httpx.Response) -> bytes:
    chunks: Final[list[bytes]] = []  # mutable-ok: response body accumulator flushed into a single bytes join
    total: int = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_RESULT_BYTES:
            raise _ResultTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


class _ResultTooLarge(Exception):
    pass


async def _execute(
    tool: ManagementTool,
    url: str,
    body: Mapping[str, object] | None,
    ctx: ManagementRequestContext,
    client: httpx.AsyncClient,
) -> mcp_types.CallToolResult:
    is_mutation: Final = tool.method != "GET"
    try:
        async with asyncio.timeout(_HANDLER_TIMEOUT_SECONDS):
            token: Final = _call_context.set(ctx)
            try:
                async with client.stream(
                    tool.method,
                    url,
                    content=json.dumps(body).encode() if body is not None else None,
                    headers=_forwarded_headers(ctx),
                ) as response:
                    status: Final = response.status_code
                    raw: Final = await _read_capped_body(response)
            finally:
                _call_context.reset(token)
    except _ResultTooLarge:
        if is_mutation:
            return error_result(
                500,
                f"{tool.name} completed but the result exceeded 1 MiB and could not be returned. "
                "Inspect the current state rather than retrying blindly.",
            )
        return error_result(500, "result too large")
    except TimeoutError:
        if is_mutation:
            return error_result(
                504,
                f"{tool.name} timed out after {_HANDLER_TIMEOUT_SECONDS}s and may or may not have "
                "completed. Inspect the current state before repeating this call.",
            )
        return error_result(504, f"{tool.name} timed out")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        verbose_proxy_logger.error("management MCP tool %s failed (%s)", tool.name, type(exc).__name__)
        return error_result(500, "internal error")

    if status >= 400:
        if not raw:
            return error_result(status, "REST endpoint returned an empty error response")
        return _tool_result(raw.decode("utf-8", errors="replace"), None, is_error=True)
    if not raw:
        return _tool_result("{}", {}, is_error=False)  # mutable-ok: empty structured result
    try:
        decoded: Final[object] = cast(object, json.loads(raw))  # cast-ok: JSON parse result is untyped
    except (json.JSONDecodeError, UnicodeDecodeError):
        return error_result(500, f"{tool.name} returned a non-JSON response")
    structured: Final[dict[str, object] | None] = (
        cast(dict[str, object], decoded) if isinstance(decoded, dict) else None  # cast-ok: JSON object
    )
    return _tool_result(raw.decode(), structured, is_error=False)


async def call_tool(
    name: str, arguments: Mapping[str, object], ctx: ManagementRequestContext
) -> mcp_types.CallToolResult:
    dispatch: Final = _active_dispatch
    if dispatch is None:
        return error_result(500, "management MCP dispatch not initialized")
    tool: Final = dispatch.catalog.tools.get(name)
    if tool is None:
        return error_result(404, f"unknown tool '{name}'")
    sections: Final = _validated_sections(tool, arguments)
    if isinstance(sections, mcp_types.CallToolResult):
        return sections
    path_args, query_args, body_args = sections
    try:
        url: Final = _target_url(tool, path_args, query_args)
    except ValueError as exc:
        return error_result(400, str(exc))
    return await _execute(tool, url, body_args, ctx, dispatch.http_client)
