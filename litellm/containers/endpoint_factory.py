"""
Factory for generating container SDK functions from JSON config.

This module reads endpoints.json and dynamically generates SDK functions
that use the generic container handler.
"""

import asyncio
import contextvars
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import litellm
from litellm.constants import request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.llms.custom_httpx.container_handler import generic_container_handler
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerFileObject,
    DeleteContainerFileResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

# Response type mapping
RESPONSE_TYPES: Dict[str, Type] = {
    "ContainerFileListResponse": ContainerFileListResponse,
    "ContainerFileObject": ContainerFileObject,
    "DeleteContainerFileResponse": DeleteContainerFileResponse,
}


def _load_endpoints_config() -> Dict:
    """Load the endpoints configuration from JSON file."""
    config_path = Path(__file__).parent / "endpoints.json"
    with open(config_path) as f:
        return json.load(f)


def create_sync_endpoint_function(endpoint_config: Dict) -> Callable:
    """
    Create a sync SDK function from endpoint config.
    
    Uses the generic container handler instead of individual handler methods.
    """
    endpoint_name = endpoint_config["name"]
    response_type = RESPONSE_TYPES.get(endpoint_config["response_type"])
    path_params = endpoint_config.get("path_params", [])
    
    @client
    def endpoint_func(
        timeout: int = 600,
        custom_llm_provider: Literal["openai"] = "openai",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        local_vars = locals()
        try:
            litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")
            litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
            _is_async = kwargs.pop("async_call", False) is True

            # Check for mock response
            mock_response = kwargs.get("mock_response")
            if mock_response is not None:
                if isinstance(mock_response, str):
                    mock_response = json.loads(mock_response)
                if response_type:
                    return response_type(**mock_response)
                return mock_response

            # Get provider config
            litellm_params = GenericLiteLLMParams(**kwargs)
            container_provider_config: Optional[BaseContainerConfig] = (
                ProviderConfigManager.get_provider_container_config(
                    provider=litellm.LlmProviders(custom_llm_provider),
                )
            )

            if container_provider_config is None:
                raise ValueError(f"Container provider config not found for: {custom_llm_provider}")

            # Build optional params for logging
            optional_params = {k: kwargs.get(k) for k in path_params if k in kwargs}

            # Pre-call logging
            litellm_logging_obj.update_environment_variables(
                model="",
                optional_params=optional_params,
                litellm_params={"litellm_call_id": litellm_call_id},
                custom_llm_provider=custom_llm_provider,
            )

            # Use generic handler
            return generic_container_handler.handle(
                endpoint_name=endpoint_name,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=litellm_logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
                _is_async=_is_async,
                **kwargs,
            )

        except Exception as e:
            raise litellm.exception_type(
                model="",
                custom_llm_provider=custom_llm_provider,
                original_exception=e,
                completion_kwargs=local_vars,
                extra_kwargs=kwargs,
            )

    return endpoint_func


def create_async_endpoint_function(
    sync_func: Callable,
    endpoint_config: Dict,
) -> Callable:
    """Create an async SDK function that wraps the sync function."""
    
    @client
    async def async_endpoint_func(
        timeout: int = 600,
        custom_llm_provider: Literal["openai"] = "openai",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        local_vars = locals()
        try:
            loop = asyncio.get_event_loop()
            kwargs["async_call"] = True

            func = partial(
                sync_func,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                **kwargs,
            )

            ctx = contextvars.copy_context()
            func_with_context = partial(ctx.run, func)
            init_response = await loop.run_in_executor(None, func_with_context)

            if asyncio.iscoroutine(init_response):
                response = await init_response
            else:
                response = init_response

            return response
        except Exception as e:
            raise litellm.exception_type(
                model="",
                custom_llm_provider=custom_llm_provider,
                original_exception=e,
                completion_kwargs=local_vars,
                extra_kwargs=kwargs,
            )

    return async_endpoint_func


def generate_container_endpoints() -> Dict[str, Callable]:
    """
    Generate all container endpoint functions from the JSON config.
    
    Returns a dict mapping function names to their implementations.
    """
    config = _load_endpoints_config()
    endpoints = {}
    
    for endpoint_config in config["endpoints"]:
        # Create sync function
        sync_func = create_sync_endpoint_function(endpoint_config)
        endpoints[endpoint_config["name"]] = sync_func
        
        # Create async function
        async_func = create_async_endpoint_function(sync_func, endpoint_config)
        endpoints[endpoint_config["async_name"]] = async_func
    
    return endpoints


def get_all_endpoint_names() -> List[str]:
    """Get all endpoint names (sync and async) from config."""
    config = _load_endpoints_config()
    names = []
    for endpoint in config["endpoints"]:
        names.append(endpoint["name"])
        names.append(endpoint["async_name"])
    return names


def get_async_endpoint_names() -> List[str]:
    """Get all async endpoint names for router registration."""
    config = _load_endpoints_config()
    return [endpoint["async_name"] for endpoint in config["endpoints"]]


# Generate endpoints on module load
_generated_endpoints = generate_container_endpoints()

# Export generated functions dynamically
list_container_files = _generated_endpoints.get("list_container_files")
alist_container_files = _generated_endpoints.get("alist_container_files")
upload_container_file = _generated_endpoints.get("upload_container_file")
aupload_container_file = _generated_endpoints.get("aupload_container_file")
retrieve_container_file = _generated_endpoints.get("retrieve_container_file")
aretrieve_container_file = _generated_endpoints.get("aretrieve_container_file")
delete_container_file = _generated_endpoints.get("delete_container_file")
adelete_container_file = _generated_endpoints.get("adelete_container_file")
retrieve_container_file_content = _generated_endpoints.get("retrieve_container_file_content")
aretrieve_container_file_content = _generated_endpoints.get("aretrieve_container_file_content")
