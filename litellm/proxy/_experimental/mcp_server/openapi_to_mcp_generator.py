"""
This module is used to generate MCP tools from OpenAPI specs.
"""

import json
import keyword
import re
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)

# Store the base URL and headers globally
BASE_URL = ""
HEADERS: Dict[str, str] = {}


def to_safe_identifier(name: str) -> str:
    """
    Convert an OpenAPI parameter name to a safe Python identifier.
    
    This function ensures that any parameter name from an OpenAPI spec can be
    used as a Python function parameter without causing syntax errors or
    security issues. It handles:
    - Hyphens, dots, and other special characters
    - Leading digits
    - Python keywords
    - Special characters like $, @, etc.
    
    Args:
        name: The original parameter name from the OpenAPI spec
        
    Returns:
        A valid Python identifier that can be used in function signatures
        
    Examples:
        >>> to_safe_identifier("repository-id")
        'repository_id'
        >>> to_safe_identifier("2fa-code")
        '_2fa_code'
        >>> to_safe_identifier("user.name")
        'user_name'
        >>> to_safe_identifier("$filter")
        '_filter'
        >>> to_safe_identifier("class")
        'class_'
    """
    if not name:
        return "_empty_"
    
    # Start with underscore if first char is not a letter
    # Replace all non-alphanumeric chars (except underscore) with underscore
    # Collapse multiple underscores
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    safe = re.sub(r'_+', '_', safe)  # Collapse multiple underscores
    
    # If starts with digit, prefix with underscore
    if safe and safe[0].isdigit():
        safe = '_' + safe
    
    # If empty after sanitization, use a default
    if not safe:
        safe = '_param_'
    
    # If it's a Python keyword, append underscore
    if keyword.iskeyword(safe):
        safe = safe + '_'
    
    # Ensure it doesn't start with a digit (shouldn't happen after above, but double-check)
    if safe and safe[0].isdigit():
        safe = '_' + safe
    
    return safe


def load_openapi_spec(filepath: str) -> Dict[str, Any]:
    """Load OpenAPI specification from JSON file."""
    with open(filepath, "r") as f:
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
    keyword arguments. Parameter names from the OpenAPI spec are safely
    mapped to valid Python identifiers to avoid syntax errors and security
    issues.

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
    all_params = path_params + query_params + body_params

    # Create mapping from original parameter names to safe identifiers
    # This allows us to accept kwargs with original names but use safe names internally
    param_name_map: Dict[str, str] = {}
    safe_to_original_map: Dict[str, str] = {}
    
    for orig_name in all_params:
        safe_name = to_safe_identifier(orig_name)
        # Handle collisions: if safe name already exists, append a counter
        counter = 1
        original_safe = safe_name
        while safe_name in safe_to_original_map:
            safe_name = f"{original_safe}_{counter}"
            counter += 1
        
        param_name_map[orig_name] = safe_name
        safe_to_original_map[safe_name] = orig_name

    # Store original parameter lists for use in the closure
    original_path_params = path_params
    original_query_params = query_params
    original_body_params = body_params
    original_method = method.lower()

    async def tool_function(**kwargs: Any) -> str:
        """
        Dynamically generated tool function.
        
        Accepts keyword arguments where keys are the original OpenAPI parameter names.
        The function safely handles parameter names that aren't valid Python identifiers.
        """
        # Build URL from base_url and path
        url = base_url + path
        
        # Replace path parameters using original names from OpenAPI spec
        for orig_param_name in original_path_params:
            # Try to get value using original name first, then safe name
            param_value = kwargs.get(orig_param_name, "")
            if not param_value and orig_param_name in param_name_map:
                safe_name = param_name_map[orig_param_name]
                param_value = kwargs.get(safe_name, "")
            
            if param_value:
                # Replace {param_name} or {{param_name}} in URL
                url = url.replace("{" + orig_param_name + "}", str(param_value))
                url = url.replace("{{" + orig_param_name + "}}", str(param_value))
        
        # Build query params using original parameter names
        params: Dict[str, Any] = {}
        for orig_param_name in original_query_params:
            # Try to get value using original name first, then safe name
            param_value = kwargs.get(orig_param_name, "")
            if not param_value and orig_param_name in param_name_map:
                safe_name = param_name_map[orig_param_name]
                param_value = kwargs.get(safe_name, "")
            
            if param_value:
                # Use original parameter name in query string (as expected by API)
                params[orig_param_name] = param_value
        
        # Build request body
        json_body: Optional[Dict[str, Any]] = None
        if original_body_params:
            # Try "body" first (most common), then check all body param names
            body_value = kwargs.get("body", {})
            if not body_value:
                for orig_param_name in original_body_params:
                    body_value = kwargs.get(orig_param_name, {})
                    if body_value:
                        break
                    # Also try safe name
                    if orig_param_name in param_name_map:
                        safe_name = param_name_map[orig_param_name]
                        body_value = kwargs.get(safe_name, {})
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
        
        # Make HTTP request
        async with httpx.AsyncClient() as client:
            if original_method == "get":
                response = await client.get(url, params=params, headers=headers)
            elif original_method == "post":
                response = await client.post(url, params=params, json=json_body, headers=headers)
            elif original_method == "put":
                response = await client.put(url, params=params, json=json_body, headers=headers)
            elif original_method == "delete":
                response = await client.delete(url, params=params, headers=headers)
            elif original_method == "patch":
                response = await client.patch(url, params=params, json=json_body, headers=headers)
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
