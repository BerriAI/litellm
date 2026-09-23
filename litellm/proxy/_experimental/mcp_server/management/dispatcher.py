"""Dispatch management MCP tool calls through the proxy's own route table.

Each call is re-entered into a middleware-free ASGI view of the FastAPI app
(the real routes, real dependencies, real auth), so a tool call is observably
identical to the same REST request: same status, same JSON body, same error
shape. Only a fixed allowlist of headers crosses from the caller's request;
nothing tool-supplied can become a header or touch the URL host.
"""

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
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
from starlette.routing import Router
from starlette.types import ASGIApp, ExceptionHandler, Receive, Scope, Send
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
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
_SK_PATTERN: Final = re.compile(r"sk-[A-Za-z0-9_-]+")
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


@dataclass(frozen=True, slots=True)
class Dispatch:
    catalog: ManagementCatalog
    internal_app: ASGIApp


_active_dispatch: Dispatch | None = None


def set_dispatch(dispatch: Dispatch | None) -> None:
    global _active_dispatch
    _active_dispatch = dispatch


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

    # app.routes is aliased, not copied: optional feature routers mount lazily
    # (e.g. access_groups on first hit) and must stay reachable through this view.
    router: Final = Router()
    router.routes = app.routes
    handler_map: Final = cast(  # cast-ok: Starlette stores handlers under Any keys
        Mapping[object, ExceptionHandler], app.exception_handlers
    )
    exception_handlers: Final = {  # mutable-ok: filtered handler map for ExceptionMiddleware
        key: handler for key, handler in handler_map.items() if key not in (500, Exception)
    }
    error_handler: Final = app.exception_handlers.get(Exception) or app.exception_handlers.get(500)
    core: Final = LazyFeatureMiddleware(
        ServerErrorMiddleware(
            AsyncExitStackMiddleware(ExceptionMiddleware(router, handlers=exception_handlers)),
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


def _redact_sk_substrings(text: str) -> str:
    return _SK_PATTERN.sub("sk-***", text)


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
    headers: Final = dict(
        {  # mutable-ok: header map assembled for httpx
            "content-type": "application/json",
            ctx.credential_header: ctx.credential_value,
        }
    )
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
    internal_app: ASGIApp,
) -> mcp_types.CallToolResult:
    is_mutation: Final = tool.method != "GET"
    try:
        async with asyncio.timeout(_HANDLER_TIMEOUT_SECONDS):
            transport: Final = httpx.ASGITransport(
                app=internal_app,
                client=ctx.client or ("127.0.0.1", 0),
                root_path=ctx.root_path,
            )
            async with (
                httpx.AsyncClient(transport=transport, base_url=_INTERNAL_BASE_URL) as client,
                client.stream(
                    tool.method,
                    url,
                    content=json.dumps(body).encode() if body is not None else None,
                    headers=_forwarded_headers(ctx),
                ) as response,
            ):
                status: Final = response.status_code
                raw: Final = await _read_capped_body(response)
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
        verbose_proxy_logger.exception("management MCP tool %s failed: %s", tool.name, _redact_sk_substrings(str(exc)))
        return error_result(500, "internal error")

    if not raw:
        return _tool_result("{}", {}, is_error=False)  # mutable-ok: empty structured result
    if status >= 400:
        return _tool_result(raw.decode("utf-8", errors="replace"), None, is_error=True)
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
    return await _execute(tool, url, body_args, ctx, dispatch.internal_app)
