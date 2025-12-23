"""
Generic HTTP handler that executes operations based on JSON endpoint definitions.

This is the core of the JSON-driven approach - it reads the endpoint definition
and executes HTTP requests accordingly.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.experimental.endpoint_definitions.loader import get_api_key
from litellm.experimental.endpoint_definitions.schema import (
    EndpointDefinition,
    Operation,
)
from litellm.llms.custom_httpx.http_handler import (
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
    
    def _build_url(
        self,
        operation: Operation,
        path_params: Dict[str, str],
    ) -> str:
        """Build the complete URL for the request."""
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
        
        # Add auth query param if configured
        if self.definition.auth.type == "query_param":
            api_key = get_api_key(self.definition.auth.env_vars)
            if not api_key:
                raise ValueError(
                    f"API key required. Set one of: {self.definition.auth.env_vars}"
                )
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{self.definition.auth.param_name}={api_key}"
        
        return url
    
    def _build_headers(
        self,
        extra_headers: Optional[Dict[str, str]] = None,
        skip_auth: bool = False,
    ) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        
        # Skip auth if hooks will handle it
        if not skip_auth and self.definition.auth.type == "header":
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
        """Build request body from kwargs - passthrough mode."""
        # Filter out internal parameters and pass through the rest
        internal_params = {"extra_headers", "timeout", "stream"}
        internal_params.update(operation.path_params)
        return {k: v for k, v in kwargs.items() if k not in internal_params and v is not None}
    
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
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute an operation synchronously.
        
        Args:
            operation_name: Name of the operation (e.g., "generate", "edit")
            extra_headers: Additional headers
            timeout: Request timeout
            **kwargs: Operation-specific parameters
            
        Returns:
            Response dict
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
        url = self._build_url(operation, path_params)
        body = self._build_request_body(operation, kwargs)
        
        verbose_logger.debug(f"Generic handler: {operation.method} {url}")
        verbose_logger.debug(f"Generic handler body: {body}")
        
        # Get HTTP client
        sync_client = _get_httpx_client()
        
        # Execute request
        if operation.method == "GET":
            response = sync_client.get(url=url, headers=headers)
        elif operation.method == "POST":
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
        
        # Log response
        verbose_logger.debug(f"Generic handler response status: {response.status_code}")
        raw_response = response.json()
        verbose_logger.debug(f"Generic handler response body: {str(raw_response)[:500]}")
        
        return raw_response
    
    async def execute_async(
        self,
        operation_name: str,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute an operation asynchronously.
        
        Args:
            operation_name: Name of the operation (e.g., "generate", "edit")
            extra_headers: Additional headers
            timeout: Request timeout
            **kwargs: Operation-specific parameters
            
        Returns:
            Response dict
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
        url = self._build_url(operation, path_params)
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
            response = await async_client.get(url=url, headers=headers)
        elif operation.method == "POST":
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
        
        # Log response
        verbose_logger.debug(f"Generic handler async response status: {response.status_code}")
        raw_response = response.json()
        verbose_logger.debug(f"Generic handler async response body: {str(raw_response)[:500]}")
        
        return raw_response
