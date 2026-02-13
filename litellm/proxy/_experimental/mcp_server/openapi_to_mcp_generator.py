"""
This module is used to generate MCP tools from OpenAPI specs.
"""

import json
import asyncio
import os
from pathlib import PurePosixPath
from typing import Any, Dict, Optional
from urllib.parse import quote

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)

# Store the base URL and headers globally
BASE_URL = ""
HEADERS: Dict[str, str] = {}


def _sanitize_path_parameter_value(param_value: Any, param_name: str) -> str:
    """Ensure path params cannot introduce directory traversal."""
    if param_value is None:
        return ""

    value_str = str(param_value)
    if value_str == "":
        return ""

    normalized_value = value_str.replace("\\", "/")
    if "/" in normalized_value:
        raise ValueError(
            f"Path parameter '{param_name}' must not contain path separators"
        )

    if any(part in {".", ".."} for part in PurePosixPath(normalized_value).parts):
        raise ValueError(
            f"Path parameter '{param_name}' cannot include '.' or '..' segments"
        )

    return quote(value_str, safe="")


def load_openapi_spec(filepath: str) -> Dict[str, Any]:
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

async def load_openapi_spec_async(filepath: str) -> Dict[str, Any]:
    if filepath.startswith("http://") or filepath.startswith("https://"):
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
        # NOTE: do not close shared client if get_async_httpx_client returns a shared singleton.
        # If it returns a new client each time, consider wrapping it in an async context manager.
        r = await client.get(filepath)
        r.raise_for_status()
        return r.json()

    # fallback: local file
    # Local filesystem path
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OpenAPI spec not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_url(spec: Dict[str, Any]) -> str:
    """Extract base URL from OpenAPI spec."""
    # OpenAPI 3.x
    if "servers" in spec and spec["servers"]:
        return spec["servers"][0]["url"]
    # OpenAPI 2.x (Swagger)
    elif "host" in spec:
        scheme = spec.get("schemes", ["https"])[0]
        base_path = spec.get("basePath", "")
        return f"{scheme}://{spec['host']}{base_path}"
    return ""


def extract_parameters(operation: Dict[str, Any]) -> tuple:
    """Extract parameter names from OpenAPI operation."""
    path_params = []
    query_params = []
    body_params = []

    # OpenAPI 3.x and 2.x parameters
    if "parameters" in operation:
        for param in operation["parameters"]:
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


def build_input_schema(operation: Dict[str, Any]) -> Dict[str, Any]:
    """Build MCP input schema from OpenAPI operation."""
    properties = {}
    required = []

    # Process parameters
    if "parameters" in operation:
        for param in operation["parameters"]:
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
        request_body = operation["requestBody"]
        content = request_body.get("content", {})

        # Try to get JSON schema
        if "application/json" in content:
            schema = content["application/json"].get("schema", {})
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


def create_tool_function(
    path: str,
    method: str,
    operation: Dict[str, Any],
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
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
    original_method = method.lower()

    async def tool_function(**kwargs: Any) -> str:
        """
        Dynamically generated tool function.

        Accepts keyword arguments where keys are the original OpenAPI parameter names.
        The function safely handles parameter names that aren't valid Python identifiers
        by using **kwargs instead of named parameters.
        """
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
        params: Dict[str, Any] = {}
        for param_name in query_params:
            param_value = kwargs.get(param_name, "")
            if param_value:
                # Use original parameter name in query string (as expected by API)
                params[param_name] = param_value

        # Build request body
        json_body: Optional[Dict[str, Any]] = None
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
                    json_body = (
                        json.loads(body_value)
                        if isinstance(body_value, str)
                        else {"data": body_value}
                    )
                except (json.JSONDecodeError, TypeError):
                    json_body = {"data": body_value}

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)

        if original_method == "get":
            response = await client.get(url, params=params, headers=headers)
        elif original_method == "post":
            response = await client.post(
                url, params=params, json=json_body, headers=headers
            )
        elif original_method == "put":
            response = await client.put(
                url, params=params, json=json_body, headers=headers
            )
        elif original_method == "delete":
            response = await client.delete(url, params=params, headers=headers)
        elif original_method == "patch":
            response = await client.patch(
                url, params=params, json=json_body, headers=headers
            )
        else:
            return f"Unsupported HTTP method: {original_method}"

        return response.text

    return tool_function


def register_tools_from_openapi(spec: Dict[str, Any], base_url: str):
    """Register MCP tools from OpenAPI specification."""
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        for method in ["get", "post", "put", "delete", "patch"]:
            if method in path_item:
                operation = path_item[method]

                # Generate tool name
                operation_id = operation.get(
                    "operationId", f"{method}_{path.replace('/', '_')}"
                )
                tool_name = operation_id.replace(" ", "_").lower()

                # Get description
                description = operation.get(
                    "summary", operation.get("description", f"{method.upper()} {path}")
                )

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
                verbose_logger.debug(f"Registered tool: {tool_name}")
