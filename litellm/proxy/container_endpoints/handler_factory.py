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


def _create_handler_for_path_params(path_params: List[str], route_type: str):
    """
    Dynamically create a handler with the correct path parameter signature.
    """
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
        
        # Create handler with correct signature for path params
        handler = _create_handler_for_path_params(path_params, route_type)
        
        # Register routes
        route_method = getattr(router, method)
        
        # Register both /v1/... and /... paths
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
