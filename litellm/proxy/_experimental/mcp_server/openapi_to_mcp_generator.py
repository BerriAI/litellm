"""
This module is used to generate MCP tools from OpenAPI specs.
"""

import json
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)

# Store the base URL and headers globally
BASE_URL = ""
HEADERS: Dict[str, str] = {}


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

    Args:
        path: API endpoint path
        method: HTTP method (get, post, put, delete, patch)
        operation: OpenAPI operation object
        base_url: Base URL for the API
        headers: Optional headers to include in requests (e.g., authentication)
    """
    if headers is None:
        headers = {}

    path_params, query_params, body_params = extract_parameters(operation)
    all_params = path_params + query_params + body_params

    # Build function signature dynamically
    if all_params:
        params_str = ", ".join(f"{p}: str = ''" for p in all_params)
    else:
        params_str = ""

    # Create the function code as a string
    func_code = f'''
async def tool_function({params_str}) -> str:
    """Dynamically generated tool function."""
    url = base_url + path
    
    # Replace path parameters
    path_param_names = {path_params}
    for param_name in path_param_names:
        param_value = locals().get(param_name, "")
        if param_value:
            url = url.replace("{{" + param_name + "}}", str(param_value))
    
    # Build query params
    query_param_names = {query_params}
    params = {{}}
    for param_name in query_param_names:
        param_value = locals().get(param_name, "")
        if param_value:
            params[param_name] = param_value
    
    # Build request body
    body_param_names = {body_params}
    json_body = None
    if body_param_names:
        body_value = locals().get("body", {{}})
        if isinstance(body_value, dict):
            json_body = body_value
        elif body_value:
            # If it's a string, try to parse as JSON
            import json as json_module
            try:
                json_body = json_module.loads(body_value) if isinstance(body_value, str) else {{"data": body_value}}
            except:
                json_body = {{"data": body_value}}
    
    # Make HTTP request
    async with httpx.AsyncClient() as client:
        if "{method.lower()}" == "get":
            response = await client.get(url, params=params, headers=headers)
        elif "{method.lower()}" == "post":
            response = await client.post(url, params=params, json=json_body, headers=headers)
        elif "{method.lower()}" == "put":
            response = await client.put(url, params=params, json=json_body, headers=headers)
        elif "{method.lower()}" == "delete":
            response = await client.delete(url, params=params, headers=headers)
        elif "{method.lower()}" == "patch":
            response = await client.patch(url, params=params, json=json_body, headers=headers)
        else:
            return "Unsupported HTTP method: {method}"
        
        return response.text
'''

    # Execute the function code to create the actual function
    local_vars = {
        "httpx": httpx,
        "headers": headers,
        "base_url": base_url,
        "path": path,
        "method": method,
    }
    exec(func_code, local_vars)

    return local_vars["tool_function"]


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
