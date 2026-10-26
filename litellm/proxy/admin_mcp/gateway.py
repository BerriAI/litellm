import asyncio
import json
import re
import time
from collections.abc import Mapping
from io import BytesIO
from types import TracebackType
from typing import Final
from urllib.parse import quote

import httpx2
import jsonschema
from mcp import types
from pydantic import JsonValue

from ._types import INTEGER, JSON_VALUE, JsonObject, object_value, string_value
from .catalog import BY_NAME, OPERATIONS, Operation
from .config import Config
from .results import (
    MAX_RESULT_BYTES,
    PAGE_BYTES,
    READ_SCHEMA,
    RESPONSE_SCHEMA,
    ResultError,
    ResultStore,
    encode,
    page,
    select,
)
from .schema import SchemaError, discovery_schema, tool_schema

DESCRIBE_SCHEMA: Final[JsonObject] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "description": "Exact name of an enabled administrative tool."},
        "response": dict(RESPONSE_SCHEMA),
    },
}
HELPERS: Final = {
    "describe_admin_tool": (
        DESCRIBE_SCHEMA,
        "Read the complete input schema and annotations of an enabled tool, including all advanced arguments and "
        "referenced definitions. Required before using a tool in discovery schema mode. Returns full detail by default; "
        "response.view=compact pages unusually large schemas.",
    ),
    "read_admin_result": (
        READ_SCHEMA,
        "Read an exact saved result without re-executing its operation. Browse fields, array items and string chunks "
        "using returned paths/next arguments. view=full returns the entire selected value. Snapshots expire after five "
        "minutes; never repeat a write to recover its response.",
    ),
}


class AdminError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def valid_credential(value: str) -> str:
    if not value or len(value) > 8192 or any(not 33 <= ord(c) <= 126 for c in value):
        raise AdminError("A personal LiteLLM admin credential is required.", 401)
    return value


def _model_params(item: JsonValue) -> JsonObject:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        try:
            parsed: Final = JSON_VALUE.validate_json(item)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_field(key: str, item: JsonValue, credential: str) -> JsonValue:
    if key.lower() == "litellm_params":
        parsed: Final = _model_params(item)
        return safe_result({k: v for k, v in parsed.items() if k in {"model", "litellm_credential_name"}}, credential)
    if key.lower() in {
        "api_key",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "client_secret",
        "authorization",
        "authentication_token",
        "credential_values",
    } and not (key.lower() in {"api_key", "token"} and isinstance(item, str) and re.fullmatch(r"[a-f0-9]{64}", item)):
        return "[redacted]"
    return safe_result(item, credential)


def safe_result(value: JsonValue, credential: str) -> JsonValue:
    if isinstance(value, dict):
        return {key: _safe_field(key, item, credential) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_result(item, credential) for item in value]
    if not isinstance(value, str):
        return value
    try:
        decoded: Final = JSON_VALUE.validate_json(value)
    except ValueError:
        return value.replace(credential, "[credential redacted]")
    if isinstance(decoded, (dict, list)):
        return json.dumps(safe_result(decoded, credential))
    return value.replace(credential, "[credential redacted]")


def _resource_path(path: str, arguments: JsonObject) -> str:
    def replacement(match: re.Match[str]) -> str:
        value: Final = str(arguments[match.group(1)])
        if value in {"", ".", ".."} or any(c in value for c in ("/", "\\", "%", "\r", "\n", "\x00")):
            raise AdminError("Invalid resource identifier.")
        return quote(value, safe="")

    return re.sub(r"\{([^{}]+)\}", replacement, path)


def _query_parts(field: str, value: JsonValue) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    parts: Final = value if isinstance(value, list) else [value]
    return tuple((field, json.dumps(part) if isinstance(part, (dict, bool)) else str(part)) for part in parts)


def _response_view(arguments: JsonObject, default: str) -> str:
    return string_value(object_value(arguments.get("response", {})).get("view", default))


class Gateway:
    def __init__(self, config: Config, client: httpx2.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client
        self._owns_client = client is None
        self._tools: Mapping[str, types.Tool] | None = None
        self._loaded_at = 0.0
        self._schema_lock = asyncio.Lock()
        self.results = ResultStore()

    async def __aenter__(self) -> "Gateway":
        if self.client is None:
            self.client = httpx2.AsyncClient(timeout=httpx2.Timeout(30, read=120), follow_redirects=False)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.results.clear()
        if self._owns_client and self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _request(
        self,
        method: str,
        path: str,
        credential: str,
        *,
        read_only: bool | None = None,
        max_bytes: int = 2_000_000,
        headers: Mapping[str, str] | None = None,
        params: tuple[tuple[str, str], ...] = (),
        content: bytes | None = None,
    ) -> JsonValue:
        assert self.client is not None
        write: Final = method != "GET" if read_only is None else not read_only
        uncertain: Final = " Check gateway state before retrying this change." if write else ""
        chunks: Final = BytesIO()
        try:
            async with self.client.stream(
                method,
                self.config.base_url + path,
                headers={"Authorization": "Bearer " + valid_credential(credential), "Cookie": "", **(headers or {})},
                follow_redirects=False,
                params=params,
                content=content,
            ) as result:
                if not 200 <= result.status_code < 300:
                    raise AdminError(f"Gateway returned HTTP {result.status_code}." + uncertain, result.status_code)
                async for chunk in result.aiter_bytes():
                    if chunks.tell() + len(chunk) > max_bytes:
                        raise AdminError(
                            "Result exceeds the upstream byte limit; use pagination or a narrower query." + uncertain
                        )
                    _ = chunks.write(chunk)
        except httpx2.RequestError:
            raise AdminError("Gateway request could not be completed." + uncertain, 504) from None
        try:
            return JSON_VALUE.validate_json(chunks.getvalue())
        except ValueError:
            raise AdminError("Gateway returned an invalid JSON response." + uncertain, 502) from None

    async def authorize(self, credential: str) -> str:
        try:
            data: Final = await self._request("GET", "/user/info", credential)
        except AdminError as exc:
            if exc.status != 404:
                raise
            raise AdminError("Only current LiteLLM proxy admins may use these tools.", 403) from None
        user: Final = data.get("user_info") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or not isinstance(user, dict)
            or user.get("user_role") != "proxy_admin"
            or not isinstance(user.get("user_id"), str)
            or not user["user_id"]
            or data.get("user_id") != user["user_id"]
            or user.get("blocked")
            or user.get("deleted")
            or user.get("is_active") is False
        ):
            raise AdminError("Only current LiteLLM proxy admins may use these tools.", 403)
        return string_value(user["user_id"])

    async def tools(self, credential: str) -> Mapping[str, types.Tool]:
        await self.authorize(credential)
        canonical: Final = await self._load_tools(credential)
        return {
            **{
                name: tool.model_copy(
                    update={
                        "input_schema": dict(discovery_schema(object_value(tool.input_schema), name)),
                        "description": (tool.description or "")
                        + " Call describe_admin_tool for the complete argument schema before use.",
                    }
                )
                if self.config.schema_mode == "discovery"
                else tool
                for name, tool in canonical.items()
            },
            **{
                name: types.Tool(
                    name=name,
                    description=description,
                    input_schema=dict(schema),
                    annotations=types.ToolAnnotations(
                        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
                    ),
                )
                for name, (schema, description) in HELPERS.items()
            },
        }

    def _tool(self, item: Operation, spec: JsonObject) -> types.Tool | None:
        if self.config.allowed_tools and item.name not in self.config.allowed_tools:
            return None
        if self.config.read_only and not item.read_only:
            return None
        path: Final = object_value(object_value(spec["paths"]).get(item.path, {}))
        raw: Final = path.get(item.method.lower())
        if not raw:
            return None
        operation: Final = object_value(raw)
        if operation.get("operationId") != item.operation_id:
            raise AdminError(f"Gateway operation changed for {item.name}; update the connector before using it.", 503)
        try:
            original: Final = tool_schema(spec, path, operation)
            schema: Final = {
                **original,
                "properties": {**object_value(original["properties"]), "response": dict(RESPONSE_SCHEMA)},
            }
            jsonschema.Draft202012Validator.check_schema(schema)
        except (SchemaError, jsonschema.SchemaError, TypeError, KeyError, ValueError):
            raise AdminError(f"Gateway schema is unsupported for {item.name}.", 503) from None
        return types.Tool(
            name=item.name,
            description=f"{item.summary}. {item.method} {item.path}. "
            + (
                "Read-only."
                if item.read_only
                else "Changes gateway state; execute only when requested. Never retry an uncertain write."
            ),
            input_schema=schema,
            annotations=types.ToolAnnotations(
                read_only_hint=item.read_only,
                destructive_hint=not item.read_only,
                idempotent_hint=item.read_only,
                open_world_hint=True,
            ),
        )

    async def _load_tools(self, credential: str) -> Mapping[str, types.Tool]:
        async with self._schema_lock:
            if self._tools is not None and time.monotonic() - self._loaded_at < 300:
                return self._tools
            spec: Final = await self._request("GET", "/openapi.json", credential, max_bytes=64_000_000)
            if not isinstance(spec, dict) or not isinstance(spec.get("paths"), dict):
                raise AdminError("Gateway OpenAPI schema is unavailable.", 503)
            tools: Final = tuple(self._tool(item, spec) for item in OPERATIONS)
            selected: Final = {tool.name: tool for tool in tools if tool is not None}
            if not selected:
                raise AdminError("No compatible, enabled admin tools were found on this gateway.", 503)
            self._tools = selected
            self._loaded_at = time.monotonic()
            return selected

    async def call(self, name: str, arguments: JsonObject, credential: str) -> JsonValue:
        principal: Final = await self.authorize(credential)
        loaded_at: Final = self._loaded_at
        available: Final = await self._load_tools(credential)
        if name not in available and name not in HELPERS:
            raise AdminError("This tool is not enabled for this connector.", 403)
        try:
            schema: Final = HELPERS[name][0] if name in HELPERS else object_value(available[name].input_schema)
            validation_error: Final = next(jsonschema.Draft202012Validator(schema).iter_errors(arguments), None)
            if validation_error is not None:
                raise validation_error
        except jsonschema.ValidationError as exc:
            location: Final = "/" + "/".join(str(p) for p in exc.absolute_path)
            raise AdminError(
                f"Arguments do not match the complete tool schema at {location}. Use describe_admin_tool for all accepted fields."
            ) from None
        if name in HELPERS:
            return await self._helper(name, arguments, credential, principal, available)
        if loaded_at != self._loaded_at:
            await self._verify(credential, principal)
        return await self._execute(
            name, arguments, credential, principal, _response_view(arguments, self.config.response_view)
        )

    async def _verify(self, credential: str, principal: str) -> None:
        if await self.authorize(credential) != principal:
            raise AdminError("Admin identity changed during the request. Inspect gateway state before retrying.", 403)

    def _view(self, name: str, value: JsonValue, credential: str, principal: str, view: str) -> JsonValue:
        if view == "compact" and len(encode(value)) > PAGE_BYTES:
            token: Final = self.results.put(name, value, credential, principal)
            if token is not None:
                return page(self.results.get(token, credential, principal), token)
        return value

    async def _helper(
        self,
        name: str,
        arguments: JsonObject,
        credential: str,
        principal: str,
        available: Mapping[str, types.Tool],
    ) -> JsonValue:
        try:
            if name == "describe_admin_tool":
                target: Final = string_value(arguments["name"])
                if target not in available:
                    raise AdminError("This tool is not enabled for this connector.", 403)
                description: Final = JSON_VALUE.validate_json(
                    available[target].model_dump_json(by_alias=True, exclude_none=True)
                )
                await self._verify(credential, principal)
                return self._view(
                    "schema:" + target, description, credential, principal, _response_view(arguments, "full")
                )
            token: Final = string_value(arguments["result_id"])
            item: Final = self.results.get(token, credential, principal)
            if item.source.removeprefix("schema:") not in available:
                raise AdminError("The source operation is no longer enabled.", 403)
            if arguments.get("view", "page") == "full":
                if any(k in arguments for k in ("offset", "limit", "max_chars")):
                    raise AdminError("Full reads do not accept pagination arguments.")
                await self._verify(credential, principal)
                return select(JSON_VALUE.validate_json(item.raw), string_value(arguments.get("path", "")))
            result: Final = page(
                item,
                token,
                path=string_value(arguments.get("path", "")),
                offset=INTEGER.validate_python(arguments.get("offset", 0), strict=True),
                limit=INTEGER.validate_python(arguments.get("limit", 20), strict=True),
                max_chars=INTEGER.validate_python(arguments.get("max_chars", 4000), strict=True),
            )
        except ResultError as exc:
            raise AdminError(str(exc), 404) from None
        await self._verify(credential, principal)
        return result

    async def _execute(
        self,
        name: str,
        arguments: JsonObject,
        credential: str,
        principal: str,
        view: str,
    ) -> JsonValue:
        item: Final = BY_NAME[name]
        path: Final = _resource_path(item.path, object_value(arguments.get("path", {})))
        query_groups: Final = tuple(
            _query_parts(field, value) for field, value in object_value(arguments.get("query", {})).items()
        )
        query: Final = sum(query_groups, ())
        headers: Final = {
            "litellm-changed-by": principal,
            **({"Content-Type": "application/json"} if "body" in arguments else {}),
        }
        result: Final = await self._request(
            item.method,
            path,
            credential,
            params=query,
            content=encode(arguments["body"]) if "body" in arguments else None,
            read_only=item.read_only,
            max_bytes=MAX_RESULT_BYTES,
            headers=headers,
        )
        await self._verify(credential, principal)
        return self._view(name, safe_result(result, credential), credential, principal, view)
