"""
Generic HTTP handler that executes operations based on JSON endpoint definitions.

This is the core of the JSON-driven approach - it reads the endpoint definition
and executes HTTP requests accordingly.
"""

import subprocess

# Import hooks - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.experimental.endpoint_definitions.loader import get_api_key
from litellm.experimental.endpoint_definitions.schema import (
    EndpointDefinition,
    Operation,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)

if TYPE_CHECKING:
    from litellm.experimental.endpoint_definitions.hooks import GenericEndpointHooks


class GenericEndpointHandler:
    """
    Generic handler that executes API operations based on JSON definitions.
    
    This replaces the need for hand-written HTTP handlers for each endpoint.
    """
    
    def __init__(
        self,
        definition: EndpointDefinition,
        hooks: Optional["GenericEndpointHooks"] = None,
    ):
        self.definition = definition
        self.hooks = hooks
    
    def _get_gcloud_access_token(self) -> Optional[str]:
        """Get access token from gcloud CLI."""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _get_bearer_token(self) -> str:
        """Get bearer token from env vars or gcloud."""
        # First try env vars
        token = get_api_key(self.definition.auth.env_vars)
        if token:
            return token
        
        # Try gcloud if configured
        if self.definition.auth.gcloud_auth:
            token = self._get_gcloud_access_token()
            if token:
                return token
        
        raise ValueError(
            f"Bearer token required. Set one of: {self.definition.auth.env_vars} "
            "or authenticate with gcloud"
        )
    
    def _build_url(
        self,
        operation: Operation,
        path_params: Dict[str, str],
        stream: bool = False,
    ) -> str:
        """Build the complete URL for the request."""
        # Choose path based on streaming
        path = operation.path
        
        # Substitute api_version
        path = path.replace("{api_version}", self.definition.api_version)
        
        # Substitute path parameters (with defaults)
        defaults = self.definition.defaults or {}
        for param_name in operation.path_params:
            value = path_params.get(param_name) or defaults.get(param_name)
            if not value:
                raise ValueError(f"Missing required path parameter: {param_name}")
            path = path.replace(f"{{{param_name}}}", value)
        
        # Handle base_url with region substitution
        base_url = self.definition.base_url
        if "{region}" in base_url:
            region = path_params.get("region") or defaults.get("region")
            if not region:
                raise ValueError("Missing required parameter: region")
            base_url = base_url.replace("{region}", region)
        
        url = f"{base_url}{path}"
        
        # Add auth query param
        if self.definition.auth.type == "query_param":
            api_key = get_api_key(self.definition.auth.env_vars)
            if not api_key:
                raise ValueError(
                    f"API key required. Set one of: {self.definition.auth.env_vars}"
                )
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{self.definition.auth.param_name}={api_key}"
        
        # Add streaming query params
        if stream and operation.streaming_query_params:
            for key, value in operation.streaming_query_params.items():
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{key}={value}"
        
        return url
    
    def _build_headers(
        self,
        extra_headers: Optional[Dict[str, str]] = None,
        skip_auth: bool = False,
    ) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        
        # Skip auth if hooks will handle it
        if not skip_auth:
            # Add bearer auth if configured
            if self.definition.auth.type == "bearer":
                token = self._get_bearer_token()
                headers["Authorization"] = f"Bearer {token}"
            elif self.definition.auth.type == "header":
                api_key = get_api_key(self.definition.auth.env_vars)
                if api_key and self.definition.auth.header_name:
                    headers[self.definition.auth.header_name] = api_key
        
        if extra_headers:
            headers.update(extra_headers)
        
        return headers
    
    def _build_request_body(
        self,
        operation: Operation,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build request body from kwargs based on operation definition."""
        if not operation.request:
            return None
        
        request_def = operation.request
        
        # Use body_template if defined (transformation mode)
        if request_def.body_template:
            return self._build_from_template(request_def.body_template, kwargs)
        
        # If body_schema is defined, pass through kwargs as body (minus internal params)
        if request_def.body_schema:
            internal_params = {"extra_headers", "timeout", "stream"}
            internal_params.update(operation.path_params)
            return {k: v for k, v in kwargs.items() if k not in internal_params and v is not None}
        
        # Standard flat request body using required/optional field lists
        body: Dict[str, Any] = {}
        
        # Add required fields
        for field in request_def.required:
            if field in kwargs and kwargs[field] is not None:
                body[field] = kwargs[field]
        
        # Add required_one_of fields
        for field in request_def.required_one_of:
            if field in kwargs and kwargs[field] is not None:
                body[field] = kwargs[field]
        
        # Add optional fields
        for field in request_def.optional:
            if field in kwargs and kwargs[field] is not None:
                body[field] = kwargs[field]
        
        return body
    
    def _build_from_template(
        self,
        template: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build request body from a template with $variable substitution.
        
        Template example:
        {
            "instances": [{"prompt": "$prompt"}],
            "parameters": {"sampleCount": "$sample_count"}
        }
        
        Variables like $prompt are replaced with kwargs values.
        Keys with None values are omitted.
        """
        return self._substitute_template(template, kwargs)
    
    def _substitute_template(self, obj: Any, kwargs: Dict[str, Any]) -> Any:
        """Recursively substitute $variables in template."""
        if isinstance(obj, str):
            # Check if it's a variable reference like "$prompt"
            if obj.startswith("$"):
                var_name = obj[1:]  # Remove $ prefix
                return kwargs.get(var_name)
            return obj
        
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                substituted = self._substitute_template(value, kwargs)
                # Only include if value is not None
                if substituted is not None:
                    # For dicts, also skip if empty
                    if isinstance(substituted, dict) and not substituted:
                        continue
                    result[key] = substituted
            return result
        
        elif isinstance(obj, list):
            result = []
            for item in obj:
                substituted = self._substitute_template(item, kwargs)
                if substituted is not None:
                    # For dicts in lists, skip if empty
                    if isinstance(substituted, dict) and not substituted:
                        continue
                    result.append(substituted)
            return result if result else None
        
        else:
            return obj
    
    def _extract_path_params(
        self,
        operation: Operation,
        kwargs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Extract path parameters from kwargs."""
        path_params = {}
        for param in operation.path_params:
            if param in kwargs:
                path_params[param] = str(kwargs.pop(param))
        return path_params
    
    def execute_sync(
        self,
        operation_name: str,
        stream: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Execute an operation synchronously.
        
        Args:
            operation_name: Name of the operation (e.g., "create", "get")
            stream: Whether to stream the response
            extra_headers: Additional headers
            timeout: Request timeout
            **kwargs: Operation-specific parameters
            
        Returns:
            Response dict or iterator of response dicts for streaming
        """
        operation = self.definition.operations.get(operation_name)
        if not operation:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        # Build initial headers (skip auth if hooks will handle it)
        headers = self._build_headers(extra_headers, skip_auth=self.hooks is not None)
        
        # Run pre-call hook if configured (handles auth)
        if self.hooks:
            headers = self.hooks.sync_pre_call_hook(operation_name, headers, kwargs)
        
        # Extract path params from kwargs (after hook, so hook can set project_id etc.)
        path_params = self._extract_path_params(operation, kwargs)
        
        # Build URL and body
        url = self._build_url(operation, path_params, stream=stream)
        body = self._build_request_body(operation, kwargs)
        
        verbose_logger.debug(f"Generic handler: {operation.method} {url}")
        verbose_logger.debug(f"Generic handler body: {body}")
        
        # Get HTTP client
        sync_client = _get_httpx_client()
        
        # Execute request
        if operation.method == "GET":
            response = sync_client.get(
                url=url,
                headers=headers,
            )
        elif operation.method == "POST":
            if stream and operation.supports_streaming:
                response = sync_client.post(
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout or request_timeout,
                    stream=True,
                )
                return self._sync_stream_response(response)
            else:
                response = sync_client.post(
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout or request_timeout,
                )
        elif operation.method == "DELETE":
            response = sync_client.delete(
                url=url,
                headers=headers,
                timeout=timeout or request_timeout,
            )
        else:
            raise ValueError(f"Unsupported method: {operation.method}")
        
        raw_response = response.json()
        
        # Apply response template if defined, otherwise return raw
        if operation.response.response_template:
            return self._extract_from_response(raw_response, operation.response.response_template)
        return raw_response
    
    async def execute_async(
        self,
        operation_name: str,
        stream: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Execute an operation asynchronously.
        
        Args:
            operation_name: Name of the operation (e.g., "create", "get")
            stream: Whether to stream the response
            extra_headers: Additional headers
            timeout: Request timeout
            **kwargs: Operation-specific parameters
            
        Returns:
            Response dict or async iterator of response dicts for streaming
        """
        operation = self.definition.operations.get(operation_name)
        if not operation:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        # Build initial headers (skip auth if hooks will handle it)
        headers = self._build_headers(extra_headers, skip_auth=self.hooks is not None)
        
        # Run pre-call hook if configured (handles auth)
        if self.hooks:
            headers = await self.hooks.async_pre_call_hook(operation_name, headers, kwargs)
        
        # Extract path params from kwargs (after hook, so hook can set project_id etc.)
        path_params = self._extract_path_params(operation, kwargs)
        
        # Build URL and body
        url = self._build_url(operation, path_params, stream=stream)
        body = self._build_request_body(operation, kwargs)
        
        verbose_logger.debug(f"Generic handler async: {operation.method} {url}")
        verbose_logger.debug(f"Generic handler body: {body}")
        
        # Get async HTTP client
        import litellm
        async_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders(self.definition.provider),
        )
        
        # Execute request
        if operation.method == "GET":
            response = await async_client.get(
                url=url,
                headers=headers,
            )
        elif operation.method == "POST":
            if stream and operation.supports_streaming:
                response = await async_client.post(
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout or request_timeout,
                    stream=True,
                )
                return self._async_stream_response(response)
            else:
                response = await async_client.post(
                    url=url,
                    headers=headers,
                    json=body,
                    timeout=timeout or request_timeout,
                )
        elif operation.method == "DELETE":
            response = await async_client.delete(
                url=url,
                headers=headers,
                timeout=timeout or request_timeout,
            )
        else:
            raise ValueError(f"Unsupported method: {operation.method}")
        
        raw_response = response.json()
        
        # Apply response template if defined, otherwise return raw
        if operation.response.response_template:
            return self._extract_from_response(raw_response, operation.response.response_template)
        return raw_response
    
    def _extract_from_response(
        self,
        response: Dict[str, Any],
        template: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract data from response using a template.
        
        Template example:
        {
            "images": [{"data": "$predictions[*].bytesBase64Encoded", "mime_type": "$predictions[*].mimeType"}]
        }
        
        The $path[*].field syntax extracts from arrays.
        """
        return self._extract_template(template, response)
    
    def _extract_template(self, template: Any, response: Dict[str, Any]) -> Any:
        """Recursively extract data from response using template."""
        if isinstance(template, str):
            if template.startswith("$"):
                return self._extract_path(template[1:], response)
            return template
        
        elif isinstance(template, dict):
            result = {}
            for key, value in template.items():
                extracted = self._extract_template(value, response)
                if extracted is not None:
                    result[key] = extracted
            return result
        
        elif isinstance(template, list):
            if len(template) == 1 and isinstance(template[0], dict):
                # Array template - extract from array in response
                item_template = template[0]
                # Find the array path from the first $path[*] reference
                array_path = self._find_array_path(item_template)
                if array_path:
                    source_array = self._get_nested(response, array_path)
                    if isinstance(source_array, list):
                        return [
                            self._extract_item(item_template, item, idx)
                            for idx, item in enumerate(source_array)
                        ]
            return [self._extract_template(item, response) for item in template]
        
        else:
            return template
    
    def _find_array_path(self, template: Dict[str, Any]) -> Optional[str]:
        """Find the array path from a template like {"data": "$predictions[*].field"}."""
        for value in template.values():
            if isinstance(value, str) and "[*]" in value:
                # Extract path before [*]
                path = value[1:] if value.startswith("$") else value
                return path.split("[*]")[0]
        return None
    
    def _extract_item(self, template: Dict[str, Any], item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Extract a single item from array using template."""
        result = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("$") and "[*]" in value:
                # Extract field after [*].
                field = value.split("[*].")[-1] if "[*]." in value else None
                if field:
                    result[key] = item.get(field)
            else:
                result[key] = value
        return result
    
    def _extract_path(self, path: str, response: Dict[str, Any]) -> Any:
        """Extract value from response using dot notation path."""
        if "[*]" in path:
            # Array extraction handled elsewhere
            return None
        return self._get_nested(response, path)
    
    def _get_nested(self, obj: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation."""
        current = obj
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _sync_stream_response(
        self,
        response: httpx.Response,
    ) -> Iterator[Dict[str, Any]]:
        """Process streaming response synchronously."""
        import json
        
        for line in response.iter_lines():
            if not line:
                continue
            
            # Handle SSE format: "data: {...}"
            if line.startswith("data: "):
                line = line[6:]
            
            if line == "[DONE]":
                break
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Failed to parse streaming chunk: {line[:100]}")
                continue
    
    async def _async_stream_response(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process streaming response asynchronously."""
        import json
        
        async for line in response.aiter_lines():
            if not line:
                continue
            
            # Handle SSE format: "data: {...}"
            if line.startswith("data: "):
                line = line[6:]
            
            if line == "[DONE]":
                break
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Failed to parse streaming chunk: {line[:100]}")
                continue

