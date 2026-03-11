"""
Factory for generating container proxy endpoints from JSON config.

This module reads the endpoints.json config and dynamically creates
FastAPI route handlers for ALL container file endpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)


def _load_endpoints_config() -> Dict:
    """Load the endpoints configuration from JSON file."""
    config_path = Path(__file__).parent.parent.parent / "containers" / "endpoints.json"
    with open(config_path) as f:
        return json.load(f)


def get_all_route_types() -> List[str]:
    """Get all async route types for registration in route_llm_request.py"""
    config = _load_endpoints_config()
    return [endpoint["async_name"] for endpoint in config["endpoints"]]


def _get_container_provider_config(custom_llm_provider: str):
    """Get the container provider config for the given provider."""
    if custom_llm_provider == "openai":
        from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
        return OpenAIContainerConfig()
    else:
        raise ValueError(f"Container API not supported for provider: {custom_llm_provider}")


def _create_handler_for_path_params(path_params: List[str], route_type: str, returns_binary: bool = False, is_multipart: bool = False):
    """
    Dynamically create a handler with the correct path parameter signature.
    """
    # For binary content endpoints, use a different handler
    if returns_binary and path_params == ["container_id", "file_id"]:
        async def handler_binary_content(
            request: Request,
            container_id: str,
            file_id: str,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            return await _process_binary_request(
                request=request,
                container_id=container_id,
                file_id=file_id,
                user_api_key_dict=user_api_key_dict,
            )
        return handler_binary_content
    
    # For multipart file upload endpoints
    if is_multipart:
        async def handler_multipart_upload(
            request: Request,
            container_id: str,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            return await _process_multipart_upload_request(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                route_type=route_type,
                container_id=container_id,
            )
        return handler_multipart_upload
    
    # Create handlers for different path parameter combinations
    if path_params == ["container_id"]:
        async def handler_container_id(
            request: Request,
            container_id: str,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            return await _process_request(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                route_type=route_type,
                path_params={"container_id": container_id},
            )
        return handler_container_id
    
    elif path_params == ["container_id", "file_id"]:
        async def handler_container_file(
            request: Request,
            container_id: str,
            file_id: str,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            return await _process_request(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                route_type=route_type,
                path_params={"container_id": container_id, "file_id": file_id},
            )
        return handler_container_file
    
    else:
        # Fallback for no path params
        async def handler_no_params(
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            return await _process_request(
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                route_type=route_type,
                path_params={},
            )
        return handler_no_params


async def _process_binary_request(
    request: Request,
    container_id: str,
    file_id: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Process binary content requests using the proper transformation pattern.
    
    This uses the provider config transformations and llm_http_handler
    to maintain consistency with the established pattern.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    # Extract custom_llm_provider
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    
    # Get the provider config
    container_provider_config = _get_container_provider_config(custom_llm_provider)
    
    # Build litellm_params - credentials are resolved by provider config from env
    litellm_params = GenericLiteLLMParams()
    
    # Create logging object
    logging_obj = Logging(
        model="container-file-content",
        messages=[],
        stream=False,
        call_type="container_file_content",
        start_time=None,
        litellm_call_id="",
        function_id="",
    )
    
    # Use the HTTP handler to make the request
    handler = BaseLLMHTTPHandler()
    
    try:
        content = await handler.async_container_file_content_handler(
            container_id=container_id,
            file_id=file_id,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
        )
        
        # Determine content type based on common file extensions in the file_id
        content_type = "application/octet-stream"
        file_id_lower = file_id.lower()
        if ".png" in file_id_lower or file_id_lower.endswith("png"):
            content_type = "image/png"
        elif ".jpg" in file_id_lower or ".jpeg" in file_id_lower:
            content_type = "image/jpeg"
        elif ".gif" in file_id_lower:
            content_type = "image/gif"
        elif ".csv" in file_id_lower:
            content_type = "text/csv"
        elif ".json" in file_id_lower:
            content_type = "application/json"
        elif ".txt" in file_id_lower:
            content_type = "text/plain"
        elif ".pdf" in file_id_lower:
            content_type = "application/pdf"
        
        return Response(
            content=content,
            media_type=content_type,
        )
        
    except Exception as e:
        raise e


async def _process_multipart_upload_request(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    route_type: str,
    container_id: str,
):
    """Process multipart file upload requests."""
    from litellm.proxy.common_utils.http_parsing_utils import (
        convert_upload_files_to_file_data,
        get_form_data,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Parse multipart form data and convert files
    form_data = await get_form_data(request)
    data = await convert_upload_files_to_file_data(form_data)
    
    if "file" not in data:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Missing required 'file' field")
    
    # convert_upload_files_to_file_data returns list of tuples, extract single file
    file_list = data["file"]
    if isinstance(file_list, list) and len(file_list) > 0:
        data["file"] = file_list[0]
    
    data["container_id"] = container_id

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type=route_type,  # type: ignore[arg-type]
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


async def _process_request(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    route_type: str,
    path_params: Dict[str, str],
):
    """Common request processing logic."""
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    query_params = dict(request.query_params)
    data: Dict[str, Any] = {
        "query_params": query_params,
        **path_params,
    }

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type=route_type,  # type: ignore[arg-type]
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


def register_container_file_endpoints(router: APIRouter) -> None:
    """
    Register ALL container file endpoints from JSON config to the router.
    
    This single function registers all endpoints defined in endpoints.json,
    eliminating the need for manual endpoint definitions.
    """
    config = _load_endpoints_config()
    
    for endpoint_config in config["endpoints"]:
        path = endpoint_config["path"]
        method = endpoint_config["method"].lower()
        path_params = endpoint_config.get("path_params", [])
        route_type = endpoint_config["async_name"]
        returns_binary = endpoint_config.get("returns_binary", False)
        is_multipart = endpoint_config.get("is_multipart", False)
        
        # Create handler with correct signature for path params
        handler = _create_handler_for_path_params(path_params, route_type, returns_binary, is_multipart)
        
        # Register routes
        route_method = getattr(router, method)
        
        # For binary endpoints, don't use ORJSONResponse
        if returns_binary:
            # Register both /v1/... and /... paths without JSON response class
            route_method(
                f"/v1{path}",
                dependencies=[Depends(user_api_key_auth)],
                tags=["containers"],
            )(handler)
            
            route_method(
                path,
                dependencies=[Depends(user_api_key_auth)],
                tags=["containers"],
            )(handler)
        else:
            # Register both /v1/... and /... paths with JSON response
            route_method(
                f"/v1{path}",
                dependencies=[Depends(user_api_key_auth)],
                response_class=ORJSONResponse,
                tags=["containers"],
            )(handler)
            
            route_method(
                path,
                dependencies=[Depends(user_api_key_auth)],
                response_class=ORJSONResponse,
                tags=["containers"],
            )(handler)
