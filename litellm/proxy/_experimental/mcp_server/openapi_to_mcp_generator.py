"""
This module is used to generate MCP tools from OpenAPI specs.
"""

import asyncio
import contextvars
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Final, TypeAlias, TypedDict
from urllib.parse import quote

import httpx

# Tool names emitted from OpenAPI specs must work across all major LLM providers.
# OpenAI/Anthropic/Bedrock all enforce a character class roughly equivalent to
# ^[a-zA-Z0-9_-]+$ on tool names. Many specs (notably GitHub's REST API) use
# tag-namespaced operationIds like "actions/download-job-logs-for-workflow-run"
# which include '/'. Sanitize here so the same regex passes everywhere downstream.
_OPENAPI_TOOL_NAME_INVALID_CHARS: Final = re.compile(r"[^a-zA-Z0-9_-]")
_OPENAPI_TOOL_NAME_MAX_LEN: Final = 128


def sanitize_openapi_tool_name(raw_name: str) -> str:
    """Map an OpenAPI operationId / fallback to a provider-safe tool name.

    Replaces any character outside ``[a-zA-Z0-9_-]`` with ``_`` and caps the
    result at 128 chars (the most restrictive of the major providers).
    Lowercased to match the existing convention in
    ``register_tools_from_openapi``.
    """
    if not raw_name:
        return raw_name
    sanitized: Final = _OPENAPI_TOOL_NAME_INVALID_CHARS.sub("_", raw_name).lower()
    return sanitized[:_OPENAPI_TOOL_NAME_MAX_LEN]


from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import async_safe_get
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)

_OpenAPIParameter: TypeAlias = Mapping[str, Any]


class _OpenAPIJSONSchema(TypedDict, total=False):
    properties: Mapping[str, object]


class _OpenAPIMediaType(TypedDict, total=False):
    schema: _OpenAPIJSONSchema


class _OpenAPIRequestBody(TypedDict, total=False):
    description: str
    required: bool
    content: Mapping[str, _OpenAPIMediaType]


class _OpenAPIOperation(TypedDict, total=False):
    operationId: str
    summary: str
    description: str
    parameters: Sequence[_OpenAPIParameter]
    requestBody: _OpenAPIRequestBody


class _OpenAPIPathItem(TypedDict, total=False):
    summary: str
    description: str
    parameters: Sequence[_OpenAPIParameter]


class _OpenAPIComponents(TypedDict, total=False):
    parameters: Mapping[str, _OpenAPIParameter]


# Store the base URL and headers globally
BASE_URL: Final = ""
HEADERS: Final[dict[str, str]] = {}

# Per-request auth header override for BYOK servers.
# Set this ContextVar before calling a local tool handler to inject the user's
# stored credential into the HTTP request made by the tool function closure.
_request_auth_header: contextvars.ContextVar[str | None] = contextvars.ContextVar("_request_auth_header", default=None)

# Per-request extra headers forwarded from the client request.
# Populated from MCPServer.extra_headers names matched against raw request
# headers in server.py before dispatching to a local/OpenAPI tool handler.
_request_extra_headers: Final[contextvars.ContextVar[dict[str, str] | None]] = contextvars.ContextVar(
    "_request_extra_headers", default=None
)

# Per-request headers carrying the gateway-resolved upstream credential
# (stored per-user OAuth token, minted M2M token, exchanged OBO token).
# Set from MCPServerManager.resolve_openapi_upstream_auth; authoritative
# over every other Authorization source in _merge_openapi_tool_request_headers.
_request_resolved_auth_headers: Final[contextvars.ContextVar[dict[str, str] | None]] = contextvars.ContextVar(
    "_request_resolved_auth_headers", default=None
)


def _sanitize_path_parameter_value(param_value: object, param_name: str) -> str:
    """Ensure path params cannot introduce directory traversal."""
    if param_value is None:
        return ""

    value_str: Final = str(param_value)
    if value_str == "":
        return ""

    normalized_value: Final = value_str.replace("\\", "/")
    if "/" in normalized_value:
        raise ValueError(f"Path parameter '{param_name}' must not contain path separators")

    if any(part in {".", ".."} for part in PurePosixPath(normalized_value).parts):
        raise ValueError(f"Path parameter '{param_name}' cannot include '.' or '..' segments")

    return quote(value_str, safe="")


def load_openapi_spec(filepath: str) -> dict[str, Any]:
    """
    Sync wrapper. For URL specs, use the shared/custom MCP httpx client.
    """
    try:
        # If we're already inside an event loop, prefer the async function.
        asyncio.get_running_loop()
        raise RuntimeError(
            "load_openapi_spec() was called from within a running event loop. "
            "Use 'await load_openapi_spec_async(...)' instead."
        )
    except RuntimeError as e:
        # "no running event loop" is fine; other RuntimeErrors we re-raise
        if "no running event loop" not in str(e).lower():
            raise
    return asyncio.run(load_openapi_spec_async(filepath))


async def load_openapi_spec_async(filepath: str) -> dict[str, Any]:
    if filepath.startswith("http://") or filepath.startswith("https://"):
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
        r: Final[httpx.Response] = await async_safe_get(client, filepath)
        r.raise_for_status()
        return r.json()

    # fallback: local file
    # Local filesystem path
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OpenAPI spec not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_url(spec: Mapping[str, Any], spec_path: str | None = None) -> str:
    """Extract base URL from OpenAPI spec."""
    # OpenAPI 3.x
    if "servers" in spec and spec["servers"]:
        server_url: Final[str] = spec["servers"][0]["url"]

        # If the server URL is relative (starts with /), derive base from spec_path
        if server_url.startswith("/") and spec_path:
            if spec_path.startswith("http://") or spec_path.startswith("https://"):
                # Extract base URL from spec_path (e.g., https://petstore3.swagger.io/api/v3/openapi.json)
                # Combine domain with the relative server URL
                from urllib.parse import urlparse

                parsed: Final = urlparse(spec_path)
                base_domain: Final = f"{parsed.scheme}://{parsed.netloc}"
                full_base_url: Final = base_domain + server_url
                verbose_logger.info(
                    "OpenAPI spec has relative server URL '%s'. Deriving base from spec_path: %s",
                    server_url,
                    full_base_url,
                )
                return full_base_url

        return server_url
    # OpenAPI 2.x (Swagger)
    elif "host" in spec:
        scheme: Final[str] = spec.get("schemes", ["https"])[0]
        base_path: Final[str] = spec.get("basePath", "")
        return f"{scheme}://{spec['host']}{base_path}"

    # Fallback: derive base URL from spec_path if it's a URL
    if spec_path and (spec_path.startswith("http://") or spec_path.startswith("https://")):
        for suffix in [
            "/openapi.json",
            "/openapi.yaml",
            "/swagger.json",
            "/swagger.yaml",
        ]:
            if spec_path.endswith(suffix):
                base_url = spec_path[: -len(suffix)]
                verbose_logger.info("No server info in OpenAPI spec. Using derived base URL: %s", base_url)
                return base_url

        if spec_path.split("/")[-1].endswith((".json", ".yaml", ".yml")):
            base_url = "/".join(spec_path.split("/")[:-1])
            verbose_logger.info("No server info in OpenAPI spec. Using derived base URL: %s", base_url)
            return base_url

    return ""


def _resolve_ref(
    param: _OpenAPIParameter, component_params: Mapping[str, _OpenAPIParameter]
) -> _OpenAPIParameter | None:
    """Resolve a single parameter, following a $ref if present.

    Returns the resolved param dict, or None if the $ref target is absent from
    components (so callers can skip/filter it rather than propagating a stub
    with name=None that would corrupt deduplication).
    """
    ref: Final[str] = param.get("$ref", "")
    if not ref.startswith("#/components/parameters/"):
        return param
    return component_params.get(ref.split("/")[-1])


def _resolve_param_list(
    raw: Sequence[_OpenAPIParameter], component_params: Mapping[str, _OpenAPIParameter]
) -> list[_OpenAPIParameter]:
    """Resolve $refs in a parameter list, dropping any unresolvable entries."""
    result: Final = []
    for p in raw:
        resolved = _resolve_ref(p, component_params)
        if resolved is not None and resolved.get("name"):
            result.append(resolved)
    return result


def resolve_operation_params(
    operation: _OpenAPIOperation,
    path_item: _OpenAPIPathItem,
    components: _OpenAPIComponents,
) -> dict[str, Any]:
    """Return a copy of *operation* with fully-resolved, merged parameters.

    Handles two common patterns in real-world OpenAPI specs:

    1. **$ref parameters** — ``{"$ref": "#/components/parameters/per-page"}``
       instead of inline objects.  Each ref is resolved against
       ``components["parameters"]``; unresolvable refs are silently dropped so
       they cannot corrupt the deduplication set with ``(None, None)`` keys.

    2. **Path-level parameters** — params defined on the path item that apply
       to every HTTP method on that path (e.g. ``owner``, ``repo``).  They are
       merged with the operation-level params; operation-level wins when the
       same ``name`` + ``in`` combination appears in both.
    """
    component_params: Final[Mapping[str, _OpenAPIParameter]] = components.get("parameters", {})
    path_level: Final = _resolve_param_list(path_item.get("parameters", []), component_params)
    op_level: Final = _resolve_param_list(operation.get("parameters", []), component_params)
    op_keys: Final = {(p["name"], p.get("in")) for p in op_level}
    merged: Final = [p for p in path_level if (p["name"], p.get("in")) not in op_keys] + op_level
    result: Final = dict(operation)
    result["parameters"] = merged
    return result


def extract_parameters(operation: Mapping[str, Any]) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
    """Extract parameter names from OpenAPI operation."""
    path_params: Final = []
    query_params: Final = []
    body_params: Final = []

    # OpenAPI 3.x and 2.x parameters
    if "parameters" in operation:
        for param in operation["parameters"]:
            if "name" not in param:
                continue
            param_name = param["name"]
            if param.get("in") == "path":
                path_params.append(param_name)
            elif param.get("in") == "query":
                query_params.append(param_name)
            elif param.get("in") == "body":
                body_params.append(param_name)

    # OpenAPI 3.x requestBody
    if "requestBody" in operation:
        body_params.append("body")

    return path_params, query_params, body_params


def build_input_schema(operation: Mapping[str, Any]) -> dict[str, Any]:
    """Build MCP input schema from OpenAPI operation."""
    properties: Final = {}
    required: Final = []

    # Process parameters
    if "parameters" in operation:
        for param in operation["parameters"]:
            if "name" not in param:
                continue
            param_name = param["name"]
            param_schema = param.get("schema", {})
            param_type = param_schema.get("type", "string")

            properties[param_name] = {
                "type": param_type,
                "description": param.get("description", ""),
            }

            if param.get("required", False):
                required.append(param_name)

    # Process requestBody (OpenAPI 3.x)
    if "requestBody" in operation:
        request_body: Final[_OpenAPIRequestBody] = operation["requestBody"]
        content: Final[Mapping[str, _OpenAPIMediaType]] = request_body.get("content", {})

        # Try to get JSON schema
        if "application/json" in content:
            schema: Final[_OpenAPIJSONSchema] = content["application/json"].get("schema", {})
            properties["body"] = {
                "type": "object",
                "description": request_body.get("description", "Request body"),
                "properties": schema.get("properties", {}),
            }
            if request_body.get("required", False):
                required.append("body")

    return {
        "type": "object",
        "properties": properties,
        "required": required if required else [],
    }


def _merge_openapi_tool_request_headers(
    static_headers: dict[str, str],
) -> dict[str, str]:
    """Merge static closure headers with per-request ContextVar overrides.

    Precedence (highest to lowest):
        1. ``_request_resolved_auth_headers`` — the gateway-resolved upstream
           credential (stored per-user OAuth token, minted M2M token,
           exchanged OBO token). The resolver is authoritative: a BYOK or
           forwarded ``Authorization`` must not shadow it, mirroring
           ``_resolve_v2_auth`` on the MCPClient path
        2. ``_request_auth_header`` — BYOK override of ``Authorization``
        3. ``static_headers`` — operator-configured headers baked into the
           tool closure at registration time
        4. ``_request_extra_headers`` — per-request headers forwarded from
           the MCP caller (allowlisted by ``MCPServer.extra_headers``)

    This matches the existing MCP invariant in
    :func:`litellm.proxy._experimental.mcp_server.utils.merge_mcp_headers`
    and the managed MCP path, where ``static_headers`` always wins over
    caller-forwarded headers. Keeping the same precedence here prevents an
    authenticated caller from overriding an operator-configured value
    (e.g. a tenant id or upstream API key) by sending the same header name.

    Header names are compared case-insensitively so different casing cannot
    bypass the precedence rules.
    """
    request_extra: Final = _request_extra_headers.get() or {}
    static: Final = static_headers or {}

    static_lower_names: Final = {k.lower() for k in static}
    effective_headers: dict[str, str] = {k: v for k, v in request_extra.items() if k.lower() not in static_lower_names}
    effective_headers.update(static)

    override_auth: Final = _request_auth_header.get()
    if override_auth:
        for existing in [k for k in effective_headers if k.lower() == "authorization"]:
            del effective_headers[existing]
        effective_headers["Authorization"] = override_auth

    resolved_auth_headers: Final = _request_resolved_auth_headers.get() or {}
    for name, value in resolved_auth_headers.items():
        for existing in [k for k in effective_headers if k.lower() == name.lower()]:
            del effective_headers[existing]
        effective_headers[name] = value

    return effective_headers


def create_tool_function(
    path: str,
    method: str,
    operation: Mapping[str, Any],
    base_url: str,
    headers: dict[str, str] | None = None,
):
    """Create a tool function for an OpenAPI operation.

    This function creates an async tool function that can be called with
    keyword arguments. Parameter names from the OpenAPI spec are accessed
    directly via **kwargs, avoiding syntax errors from invalid Python identifiers.

    Args:
        path: API endpoint path
        method: HTTP method (get, post, put, delete, patch)
        operation: OpenAPI operation object
        base_url: Base URL for the API
        headers: Optional headers to include in requests (e.g., authentication)

    Returns:
        An async function that accepts **kwargs and makes the HTTP request
    """
    if headers is None:
        headers = {}

    path_params, query_params, body_params = extract_parameters(operation)
    original_method: Final = method.lower()

    async def tool_function(**kwargs: object) -> str:
        """
        Dynamically generated tool function.

        Accepts keyword arguments where keys are the original OpenAPI parameter names.
        The function safely handles parameter names that aren't valid Python identifiers
        by using **kwargs instead of named parameters.
        """
        effective_headers: Final = _merge_openapi_tool_request_headers(headers)

        # Build URL from base_url and path
        url = base_url + path

        # Replace path parameters using original names from OpenAPI spec
        # Apply path traversal validation and URL encoding
        for param_name in path_params:
            param_value = kwargs.get(param_name, "")
            if param_value:
                try:
                    # Sanitize and encode path parameter to prevent traversal attacks
                    safe_value = _sanitize_path_parameter_value(param_value, param_name)
                except ValueError as exc:
                    return "Invalid path parameter: " + str(exc)
                # Replace {param_name} or {{param_name}} in URL
                url = url.replace("{" + param_name + "}", safe_value)
                url = url.replace("{{" + param_name + "}}", safe_value)

        # Build query params using original parameter names
        params: Final[dict[str, Any]] = {}
        for param_name in query_params:
            param_value = kwargs.get(param_name, "")
            if param_value:
                # Use original parameter name in query string (as expected by API)
                params[param_name] = param_value

        # Build request body
        json_body: dict[str, Any] | None = None
        if body_params:
            # Try "body" first (most common), then check all body param names
            body_value = kwargs.get("body", {})
            if not body_value:
                for param_name in body_params:
                    body_value = kwargs.get(param_name, {})
                    if body_value:
                        break

            if isinstance(body_value, dict):
                json_body = body_value
            elif body_value:
                # If it's a string, try to parse as JSON
                try:
                    json_body = json.loads(body_value) if isinstance(body_value, str) else {"data": body_value}
                except (json.JSONDecodeError, TypeError):
                    json_body = {"data": body_value}

        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)

        if original_method == "get":
            response = await client.get(url, params=params, headers=effective_headers)
        elif original_method == "post":
            response = await client.post(url, params=params, json=json_body, headers=effective_headers)
        elif original_method == "put":
            response = await client.put(url, params=params, json=json_body, headers=effective_headers)
        elif original_method == "delete":
            response = await client.delete(url, params=params, headers=effective_headers)
        elif original_method == "patch":
            response = await client.patch(url, params=params, json=json_body, headers=effective_headers)
        else:
            return f"Unsupported HTTP method: {original_method}"

        return response.text

    return tool_function


def register_tools_from_openapi(spec: Mapping[str, Any], base_url: str) -> None:
    """Register MCP tools from OpenAPI specification."""
    paths: Final[Mapping[str, Mapping[str, Any]]] = spec.get("paths", {})
    used_names: Final = set()

    for path, path_item in paths.items():
        for method in ["get", "post", "put", "delete", "patch"]:
            if method in path_item:
                operation = path_item[method]

                # Generate tool name. Sanitize to ^[a-zA-Z0-9_-]+$ (lowercase)
                # so the resulting name is valid across OpenAI/Anthropic/Bedrock.
                # Many specs (e.g. GitHub REST) use tag-namespaced operationIds
                # like "actions/download-job-logs-for-workflow-run" which
                # contain '/' and would 400 at the LLM provider boundary.
                operation_id = operation.get("operationId", f"{method}_{path}")
                tool_name = sanitize_openapi_tool_name(operation_id)

                # Disambiguate collisions: two operationIds that differ only
                # by sanitized characters (e.g. "foo/list" and "foo.list")
                # would both become "foo_list". Append _2, _3, … to keep
                # every tool reachable, mirroring the Anthropic-side logic
                # in _build_anthropic_tool_name_maps.
                unique = tool_name
                n = 1
                while unique in used_names:
                    n += 1
                    suffix = f"_{n}"
                    unique = tool_name[: _OPENAPI_TOOL_NAME_MAX_LEN - len(suffix)] + suffix
                tool_name = unique
                used_names.add(tool_name)

                # Get description
                description = operation.get("summary", operation.get("description", f"{method.upper()} {path}"))

                # Build input schema
                input_schema = build_input_schema(operation)

                # Create tool function
                tool_func = create_tool_function(path, method, operation, base_url)
                tool_func.__name__ = tool_name
                tool_func.__doc__ = description

                # Register tool with local registry
                global_mcp_tool_registry.register_tool(
                    name=tool_name,
                    description=description,
                    input_schema=input_schema,
                    handler=tool_func,
                )
                verbose_logger.debug("Registered tool: %s", tool_name)
