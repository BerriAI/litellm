"""Dispatch management MCP tool calls to the REST management handlers.

Each tool is wired to its handler by an explicit typed call site: no
reflection, no generic kwargs. Authentication re-runs per tool against the
synthetic request built for the concrete REST route, so per-key allowed_routes,
custom auth hooks, and DISABLE_ADMIN_ENDPOINTS all see the real endpoint path.
"""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeVar
from urllib.parse import urlencode

import mcp.types as mcp_types
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.types import Message, Scope
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.management.catalog import (
    MANAGEMENT_TOOLS_BY_NAME,
    AccessGroupIdArguments,
    GetVirtualKeyArguments,
    ListVirtualKeysArguments,
    ManagementTool,
    UpdateAccessGroupArguments,
    path_param_names,
)
from litellm.proxy._types import (
    GenerateKeyRequest,
    GenerateKeyResponse,
    KeyListResponseObject,
    KeyRequest,
    LitellmUserRoles,
    ProxyException,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.list_api.common import ManagementProblem
from litellm.proxy.utils import hash_token
from litellm.types.access_group import (
    AccessGroupCreateRequest,
    AccessGroupResponse,
)

_HANDLER_TIMEOUT_SECONDS: Final = 30.0
_MAX_RESULT_BYTES: Final = 1024 * 1024
_SK_PATTERN: Final = re.compile(r"sk-[A-Za-z0-9_-]+")
_STRIPPED_REQUEST_HEADERS: Final = frozenset(
    {"content-type", "content-length", "mcp-session-id", "mcp-protocol-version", "accept"}
)
_ECHOED_SECRET_FIELDS: Final = frozenset({"key", "keys", "deleted_keys"})


class KeyInfoResult(TypedDict, total=False):
    key: ReadOnly[str | None]
    info: ReadOnly[dict[str, object]]


class DeletedKeysResult(TypedDict, total=False):
    deleted_keys: ReadOnly[list[str]]


class _WrappedResult(TypedDict):
    result: ReadOnly[object]


class _ErrorPayload(TypedDict):
    status: ReadOnly[int]
    detail: ReadOnly[object]


class _ValidationIssue(TypedDict):
    loc: ReadOnly[tuple[object, ...]]
    msg: ReadOnly[object]
    type: ReadOnly[object]


class _ValidationPayload(TypedDict):
    detail: ReadOnly[tuple[_ValidationIssue, ...]]


_KEY_INFO_ADAPTER: Final = TypeAdapter(KeyInfoResult)
_DELETED_KEYS_ADAPTER: Final = TypeAdapter(DeletedKeysResult)
_KEY_MUTATION_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[object])
_KEY_LIST_ADAPTER: Final = TypeAdapter(KeyListResponseObject)
_ACCESS_GROUP_LIST_ADAPTER: Final = TypeAdapter(list[AccessGroupResponse])
_PLAIN_OBJECT_ADAPTER: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class ManagementRequestContext:
    """The trusted slices of the outer HTTP request, captured at admission.

    The dispatcher rebuilds a synthetic Starlette request per tool call from
    these fields; nothing client-controlled in the tool arguments can reach
    into the request the handler sees.
    """

    raw_headers: tuple[tuple[bytes, bytes], ...]
    client: tuple[str, int] | None
    scheme: str
    server: tuple[str, int | None] | None
    root_path: str
    http_version: str
    app: object
    api_key: str
    litellm_changed_by: str | None


_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def _coerce(expected: type[_ArgsT], arguments: BaseModel) -> _ArgsT:
    if not isinstance(arguments, expected):
        raise TypeError(f"arguments for tool dispatch are not a {expected.__name__}")
    return arguments


def _as_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return _OBJECT_DICT_ADAPTER.validate_python(value)


def _tool_result(
    payload_text: str, structured: dict[str, object] | _WrappedResult, is_error: bool
) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=payload_text)],  # mutable-ok: SDK content field is a list
        structured_content=structured,
        is_error=is_error,
    )


def _text_result(payload: object, is_error: bool) -> mcp_types.CallToolResult:
    encoded: Final[object] = _PLAIN_OBJECT_ADAPTER.validate_python(jsonable_encoder(payload))
    wrapped: Final[_WrappedResult] = {"result": encoded}
    encoded_dict: Final = _as_object_dict(encoded)
    structured: Final = encoded_dict if encoded_dict is not None else wrapped
    return _tool_result(json.dumps(encoded, default=str), structured, is_error)


def error_result(status: int, detail: object) -> mcp_types.CallToolResult:
    return _text_result(_ErrorPayload(status=status, detail=detail), is_error=True)


def _redact_sk_substrings(text: str) -> str:
    return _SK_PATTERN.sub("sk-***", text)


def _validation_error_result(tool: ManagementTool, exc: ValidationError) -> mcp_types.CallToolResult:
    path_params: Final = path_param_names(tool)
    default_location: Final = "query" if tool.http_method in ("GET", "DELETE") else "body"
    detail: Final[tuple[_ValidationIssue, ...]] = tuple(
        _ValidationIssue(
            loc=(
                "path" if error["loc"] and str(error["loc"][0]) in path_params else default_location,
                *error["loc"],
            ),
            msg=error["msg"],
            type=error["type"],
        )
        for error in exc.errors(include_url=False)
    )
    return _text_result(_ValidationPayload(detail=detail), is_error=True)


def _model_fields(arguments: BaseModel, *, drop_none: bool) -> dict[str, object]:
    dumped: Final[dict[str, object]] = arguments.model_dump(exclude_unset=True, exclude_none=drop_none)
    return dumped


def _query_string(tool: ManagementTool, arguments: BaseModel) -> bytes:
    path_params: Final = path_param_names(tool)
    values: Final = _model_fields(arguments, drop_none=True)
    pairs: Final[tuple[tuple[str, object], ...]] = tuple(
        (key, item)
        for key, value in values.items()
        if key not in path_params
        for item in (_OBJECT_LIST_ADAPTER.validate_python(value) if isinstance(value, (list, tuple)) else (value,))
    )
    return urlencode(pairs).encode()


def _request_body(tool: ManagementTool, arguments: BaseModel) -> bytes:
    if tool.http_method in ("GET", "DELETE"):
        return b""
    if tool.name == "update_access_group":
        return _coerce(UpdateAccessGroupArguments, arguments).data.model_dump_json(exclude_unset=True).encode()
    return json.dumps(_model_fields(arguments, drop_none=False)).encode()


def _substituted_path(tool: ManagementTool, arguments: BaseModel) -> str:
    path: Final[str] = tool.rest_path
    values: Final = _model_fields(arguments, drop_none=False)
    return "/".join(
        str(values[segment[1:-1]]) if segment.startswith("{") and segment.endswith("}") else segment
        for segment in path.split("/")
    )


def _build_request(tool: ManagementTool, arguments: BaseModel, ctx: ManagementRequestContext) -> Request:
    path: Final = _substituted_path(tool, arguments)
    body: Final = _request_body(tool, arguments)
    forwarded: Final[tuple[tuple[bytes, bytes], ...]] = tuple(
        (name, value)
        for name, value in ctx.raw_headers
        if name.decode("latin-1").lower() not in _STRIPPED_REQUEST_HEADERS
    ) + (((b"content-type", b"application/json"),) if body else ())
    path_params: Final[dict[str, object]] = {  # mutable-ok: ASGI path_params must be a mutable dict
        name: getattr(arguments, name) for name in path_param_names(tool) if getattr(arguments, name, None) is not None
    }
    scope: Final[Scope] = {
        "type": "http",
        "http_version": ctx.http_version,
        "method": tool.http_method,
        "scheme": ctx.scheme,
        "server": ctx.server,
        "client": ctx.client,
        "root_path": ctx.root_path,
        "path": path,
        "raw_path": path.encode(),
        "query_string": _query_string(tool, arguments),
        "headers": list(forwarded),  # mutable-ok: ASGI headers must be a mutable list
        "state": {},
        "path_params": path_params,
        "app": ctx.app,
    }  # mutable-ok: ASGI scope must be a mutable mapping

    async def _receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}  # mutable-ok: ASGI messages are dicts

    return Request(scope, _receive)


def _redact_secret_echoes(result: object) -> object:
    """Hash any raw ``sk-`` token echoed back in the top-level key fields."""
    fields: Final = _as_object_dict(result)
    if fields is None:
        return result

    def _mask(value: object) -> object:
        if isinstance(value, str) and value.startswith("sk-"):
            return hash_token(value)
        if isinstance(value, list):
            items: Final = _OBJECT_LIST_ADAPTER.validate_python(value)
            return tuple(
                hash_token(item) if isinstance(item, str) and item.startswith("sk-") else item for item in items
            )
        return value

    return {  # mutable-ok: rebuilt result dict goes straight to jsonable_encoder
        k: (_mask(v) if k in _ECHOED_SECRET_FIELDS else v) for k, v in fields.items()
    }


async def _authenticate(request: Request, ctx: ManagementRequestContext) -> UserAPIKeyAuth:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    caller: Final = await user_api_key_auth(request=request, api_key=ctx.api_key)
    if caller.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise ProxyException(
            message=f"Management MCP tools require a proxy admin key. Your role={caller.user_role}",
            type="auth_error",
            param="None",
            code=403,
        )
    return caller


async def _call_handler(
    tool: ManagementTool, arguments: BaseModel, request: Request, caller: UserAPIKeyAuth, changed_by: str | None
) -> object:
    """The typed dispatch table: one keyword-explicit call site per tool."""
    from litellm.proxy.management_endpoints import access_group_endpoints as age
    from litellm.proxy.management_endpoints import key_management_endpoints as kme

    match tool.name:
        case "list_virtual_keys":
            args = _coerce(ListVirtualKeysArguments, arguments)
            return await kme.list_keys(
                request=request,
                user_api_key_dict=caller,
                page=args.page,
                size=args.size,
                user_id=args.user_id,
                team_id=args.team_id,
                organization_id=args.organization_id,
                key_hash=args.key_hash,
                key_alias=args.key_alias,
                search=args.search,
                return_full_object=args.return_full_object,
                include_team_keys=args.include_team_keys,
                include_created_by_keys=args.include_created_by_keys,
                sort_by=args.sort_by,
                sort_order=args.sort_order,
                expand=args.expand,
                status=args.status,
                project_id=args.project_id,
                access_group_id=args.access_group_id,
                agent_id=args.agent_id,
                substring_matching=args.substring_matching,
                expires=args.expires,
            )
        case "get_virtual_key":
            return _PLAIN_OBJECT_ADAPTER.validate_python(
                await kme.info_key_fn(
                    key=_coerce(GetVirtualKeyArguments, arguments).key,
                    user_api_key_dict=caller,
                )
            )
        case "create_virtual_key":
            return await kme.generate_key_fn(
                data=_coerce(GenerateKeyRequest, arguments),
                user_api_key_dict=caller,
                litellm_changed_by=changed_by,
            )
        case "update_virtual_key":
            return _PLAIN_OBJECT_ADAPTER.validate_python(
                await kme.update_key_fn(
                    request=request,
                    data=_coerce(UpdateKeyRequest, arguments),
                    user_api_key_dict=caller,
                    litellm_changed_by=changed_by,
                )
            )
        case "delete_virtual_keys":
            return await kme.delete_key_fn(
                data=_coerce(KeyRequest, arguments),
                user_api_key_dict=caller,
                litellm_changed_by=changed_by,
            )
        case "list_access_groups":
            return await age.list_access_groups(user_api_key_dict=caller)
        case "get_access_group":
            return await age.get_access_group(
                access_group_id=_coerce(AccessGroupIdArguments, arguments).access_group_id,
                user_api_key_dict=caller,
            )
        case "create_access_group":
            return await age.create_access_group(
                data=_coerce(AccessGroupCreateRequest, arguments),
                user_api_key_dict=caller,
            )
        case "update_access_group":
            update_args = _coerce(UpdateAccessGroupArguments, arguments)
            return await age.update_access_group(
                access_group_id=update_args.access_group_id,
                data=update_args.data,
                user_api_key_dict=caller,
            )
        case "delete_access_group":
            return await age.delete_access_group(
                access_group_id=_coerce(AccessGroupIdArguments, arguments).access_group_id,
                user_api_key_dict=caller,
            )
        case _:  # pragma: no cover - catalog and dispatch stay in lockstep
            raise ValueError(f"no handler wired for management MCP tool '{tool.name}'")


def _validate_result(tool: ManagementTool, result: object) -> object:
    match tool.name:
        case "create_virtual_key":
            return GenerateKeyResponse.model_validate(result)
        case "list_virtual_keys":
            return _KEY_LIST_ADAPTER.validate_python(result)
        case "get_virtual_key":
            return _KEY_INFO_ADAPTER.validate_python(result)
        case "update_virtual_key":
            return _KEY_MUTATION_ADAPTER.validate_python(result)
        case "delete_virtual_keys":
            return _DELETED_KEYS_ADAPTER.validate_python(result)
        case "list_access_groups":
            return _ACCESS_GROUP_LIST_ADAPTER.validate_python(result)
        case "get_access_group" | "create_access_group" | "update_access_group":
            return AccessGroupResponse.model_validate(result)
        case "delete_access_group":
            empty: Final[dict[str, object]] = {}  # mutable-ok: empty result payload returned to the MCP client
            return empty if result is None else _KEY_MUTATION_ADAPTER.validate_python(result)
        case _:  # pragma: no cover - defensive default for future tools
            return result


def _to_error_result(tool: ManagementTool, exc: BaseException) -> mcp_types.CallToolResult:
    if isinstance(exc, ManagementProblem):
        return error_result(exc.problem.status, _redact_sk_substrings(exc.problem.detail or "error"))
    if isinstance(exc, HTTPException):
        return error_result(exc.status_code, _redact_sk_substrings(str(exc.detail)))
    if isinstance(exc, ProxyException):
        try:
            status: Final = int(exc.code)
        except (TypeError, ValueError):
            return error_result(500, "internal error")
        return error_result(status, _redact_sk_substrings(exc.message))
    if PrismaDBExceptionHandler.is_database_service_unavailable_error_in_chain(exc):
        return error_result(503, "database unavailable")
    verbose_proxy_logger.exception("management MCP tool %s failed: %s", tool.name, exc)
    return error_result(500, "internal error")


async def call_tool(
    name: str, arguments: Mapping[str, object], ctx: ManagementRequestContext
) -> mcp_types.CallToolResult:
    tool: Final = MANAGEMENT_TOOLS_BY_NAME.get(name)
    if tool is None:
        return error_result(404, f"unknown tool '{name}'")
    try:
        args_model: Final = tool.arguments_model.model_validate(arguments)
    except ValidationError as exc:
        return _validation_error_result(tool, exc)

    request: Final = _build_request(tool, args_model, ctx)
    is_mutation: Final = tool.http_method not in ("GET",)
    try:
        caller: Final = await _authenticate(request, ctx)
    except (HTTPException, ProxyException) as exc:
        return _to_error_result(tool, exc)
    except Exception as exc:
        return _to_error_result(tool, exc)

    try:
        result: Final = await asyncio.wait_for(
            _call_handler(tool, args_model, request, caller, ctx.litellm_changed_by),
            timeout=_HANDLER_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        if is_mutation:
            return error_result(
                504,
                f"{tool.name} timed out after {_HANDLER_TIMEOUT_SECONDS}s and may or may not have "
                "completed. Inspect the current state before repeating this call.",
            )
        return error_result(504, f"{tool.name} timed out")
    except (HTTPException, ProxyException, ManagementProblem) as exc:
        return _to_error_result(tool, exc)
    except Exception as exc:
        return _to_error_result(tool, exc)

    try:
        validated: Final = _validate_result(tool, result)
    except ValidationError:
        verbose_proxy_logger.exception("management MCP tool %s returned an unexpected shape", tool.name)
        return error_result(500, "internal error")

    redacted: Final[object] = (
        _redact_secret_echoes(validated)
        if tool.name in ("get_virtual_key", "delete_virtual_keys", "update_virtual_key")
        else validated
    )

    encoded: Final[object] = _PLAIN_OBJECT_ADAPTER.validate_python(jsonable_encoder(redacted))
    payload: Final = json.dumps(encoded, default=str)
    if len(payload.encode()) > _MAX_RESULT_BYTES:
        if is_mutation:
            return error_result(
                500,
                f"{tool.name} completed but the result exceeded 1 MiB and could not be returned. "
                "Inspect the current state rather than retrying blindly.",
            )
        return error_result(500, "result too large")

    wrapped: Final[_WrappedResult] = {"result": encoded}
    encoded_dict: Final = _as_object_dict(encoded)
    structured: Final = encoded_dict if encoded_dict is not None else wrapped
    return _tool_result(payload, structured, is_error=False)
