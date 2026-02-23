"""
Generic container file handler for LiteLLM.

This module provides a single generic handler that can process any container file
endpoint defined in endpoints.json, eliminating the need for individual handler methods.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Type, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerFileObject,
    DeleteContainerFileResponse,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.containers.transformation import BaseContainerConfig


# Response type mapping
RESPONSE_TYPES: Dict[str, Type] = {
    "ContainerFileListResponse": ContainerFileListResponse,
    "ContainerFileObject": ContainerFileObject,
    "DeleteContainerFileResponse": DeleteContainerFileResponse,
}


def _load_endpoints_config() -> Dict:
    """Load the endpoints configuration from JSON file."""
    config_path = Path(__file__).parent.parent.parent / "containers" / "endpoints.json"
    with open(config_path) as f:
        return json.load(f)


def _get_endpoint_config(endpoint_name: str) -> Optional[Dict]:
    """Get config for a specific endpoint by name."""
    config = _load_endpoints_config()
    for endpoint in config["endpoints"]:
        if endpoint["name"] == endpoint_name or endpoint["async_name"] == endpoint_name:
            return endpoint
    return None


def _build_url(
    api_base: str,
    path_template: str,
    path_params: Dict[str, str],
) -> str:
    """Build the full URL by substituting path parameters.
    
    The api_base from get_complete_url already includes /containers,
    so we need to strip that prefix from the path_template.
    """
    # api_base ends with /containers, path_template starts with /containers
    # So we need to strip /containers from the path
    if path_template.startswith("/containers"):
        path_template = path_template[len("/containers"):]
    
    url = f"{api_base.rstrip('/')}{path_template}"
    for param, value in path_params.items():
        url = url.replace(f"{{{param}}}", value)
    return url


def _build_query_params(
    query_param_names: list,
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Build query parameters from kwargs."""
    params = {}
    for param_name in query_param_names:
        value = kwargs.get(param_name)
        if value is not None:
            params[param_name] = str(value) if not isinstance(value, str) else value
    return params


def _prepare_multipart_file_upload(
    file: Any,
    headers: Dict[str, Any],
) -> tuple:
    """
    Prepare file and headers for multipart upload.
    
    Returns:
        Tuple of (files_dict, headers_without_content_type)
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )
    
    extracted = extract_file_data(file)
    filename = extracted.get("filename") or "file"
    content = extracted.get("content") or b""
    content_type = extracted.get("content_type") or "application/octet-stream"
    files = {"file": (filename, content, content_type)}
    
    # Remove content-type header - httpx will set it automatically for multipart
    headers_copy = headers.copy()
    headers_copy.pop("content-type", None)
    headers_copy.pop("Content-Type", None)
    
    return files, headers_copy


class GenericContainerHandler:
    """
    Generic handler for container file API endpoints.
    
    This single handler can process any endpoint defined in endpoints.json,
    eliminating the need for individual handler methods per endpoint.
    """
    
    def handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        **kwargs,
    ) -> Union[Any, Coroutine[Any, Any, Any]]:
        """
        Generic handler for any container file endpoint.
        
        Args:
            endpoint_name: Name of the endpoint (e.g., "list_container_files")
            container_provider_config: Provider-specific configuration
            litellm_params: LiteLLM parameters including api_key, api_base
            logging_obj: Logging object for request logging
            extra_headers: Additional HTTP headers
            extra_query: Additional query parameters
            timeout: Request timeout
            _is_async: Whether to make async request
            client: Optional HTTP client
            **kwargs: Path params and query params (e.g., container_id, file_id, after, limit)
        """
        if _is_async:
            return self._async_handle(
                endpoint_name=endpoint_name,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
                **kwargs,
            )
        
        return self._sync_handle(
            endpoint_name=endpoint_name,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            client=client,
            **kwargs,
        )
    
    def _sync_handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        **kwargs,
    ) -> Any:
        """Synchronous request handler."""
        endpoint_config = _get_endpoint_config(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        # Get HTTP client
        if client is None or not isinstance(client, HTTPHandler):
            http_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            http_client = client
        
        # Build request
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )
        if extra_headers:
            headers.update(extra_headers)
        
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )
        
        # Build URL with path params
        path_params = {p: kwargs.get(p, "") for p in endpoint_config.get("path_params", [])}
        url = _build_url(api_base, endpoint_config["path"], path_params)
        
        # Build query params
        query_params = _build_query_params(endpoint_config.get("query_params", []), kwargs)
        if extra_query:
            query_params.update(extra_query)
        
        # Log request
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": query_params,
            },
        )
        
        # Make request
        method = endpoint_config["method"].upper()
        returns_binary = endpoint_config.get("returns_binary", False)
        is_multipart = endpoint_config.get("is_multipart", False)
        
        try:
            if method == "GET":
                response = http_client.get(url=url, headers=headers, params=query_params)
            elif method == "DELETE":
                response = http_client.delete(url=url, headers=headers, params=query_params)
            elif method == "POST":
                if is_multipart and "file" in kwargs:
                    files, headers = _prepare_multipart_file_upload(kwargs["file"], headers)
                    response = http_client.post(url=url, headers=headers, params=query_params, files=files)
                else:
                    response = http_client.post(url=url, headers=headers, params=query_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # For binary responses, return raw content
            if returns_binary:
                return response.content
            
            # Check for error response
            response_json = response.json()
            if "error" in response_json:
                from litellm.llms.base_llm.chat.transformation import BaseLLMException
                error_msg = response_json.get("error", {}).get("message", str(response_json))
                raise BaseLLMException(
                    status_code=response.status_code,
                    message=error_msg,
                    headers=dict(response.headers),
                )
            
            # Parse response
            response_type = RESPONSE_TYPES.get(endpoint_config["response_type"])
            if response_type:
                return response_type(**response_json)
            return response_json
            
        except Exception as e:
            raise e
    
    async def _async_handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        **kwargs,
    ) -> Any:
        """Asynchronous request handler."""
        endpoint_config = _get_endpoint_config(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        # Get HTTP client
        if client is None or not isinstance(client, AsyncHTTPHandler):
            http_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            http_client = client
        
        # Build request
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )
        if extra_headers:
            headers.update(extra_headers)
        
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )
        
        # Build URL with path params
        path_params = {p: kwargs.get(p, "") for p in endpoint_config.get("path_params", [])}
        url = _build_url(api_base, endpoint_config["path"], path_params)
        
        # Build query params
        query_params = _build_query_params(endpoint_config.get("query_params", []), kwargs)
        if extra_query:
            query_params.update(extra_query)
        
        # Log request
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": query_params,
            },
        )
        
        # Make request
        method = endpoint_config["method"].upper()
        returns_binary = endpoint_config.get("returns_binary", False)
        is_multipart = endpoint_config.get("is_multipart", False)
        
        try:
            if method == "GET":
                response = await http_client.get(url=url, headers=headers, params=query_params)
            elif method == "DELETE":
                response = await http_client.delete(url=url, headers=headers, params=query_params)
            elif method == "POST":
                if is_multipart and "file" in kwargs:
                    files, headers = _prepare_multipart_file_upload(kwargs["file"], headers)
                    response = await http_client.post(url=url, headers=headers, params=query_params, files=files)
                else:
                    response = await http_client.post(url=url, headers=headers, params=query_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # For binary responses, return raw content
            if returns_binary:
                return response.content
            
            # Check for error response
            response_json = response.json()
            if "error" in response_json:
                from litellm.llms.base_llm.chat.transformation import BaseLLMException
                error_msg = response_json.get("error", {}).get("message", str(response_json))
                raise BaseLLMException(
                    status_code=response.status_code,
                    message=error_msg,
                    headers=dict(response.headers),
                )
            
            # Parse response
            response_type = RESPONSE_TYPES.get(endpoint_config["response_type"])
            if response_type:
                return response_type(**response_json)
            return response_json
            
        except Exception as e:
            raise e


# Singleton instance
generic_container_handler = GenericContainerHandler()

