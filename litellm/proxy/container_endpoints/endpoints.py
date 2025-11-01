#### Container Endpoints #####

from typing import Any, Dict
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
    get_custom_llm_provider_from_request_body,
)

router = APIRouter()


@router.post(
    "/v1/containers",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
@router.post(
    "/containers",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
async def create_container(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Container creation endpoint for creating new containers.
    
    Follows the OpenAI Containers API spec:
    https://platform.openai.com/docs/api-reference/containers
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/containers" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "My Container",
            "expires_after": {
                "anchor": "last_active_at",
                "minutes": 20
            }
        }'
    ```
    
    Or specify provider via header:
    ```bash
    curl -X POST "http://localhost:4000/v1/containers" \
        -H "Authorization: Bearer sk-1234" \
        -H "custom-llm-provider: azure" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "My Container"
        }'
    ```
    """
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

    # Read request body
    data = await _read_request_body(request=request)

    # Extract custom_llm_provider using priority chain
    # Priority: headers > query params > request body > default
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
        or "openai"
    )
    
    # Add custom_llm_provider to data
    data["custom_llm_provider"] = custom_llm_provider
    
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_container",
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


@router.get(
    "/v1/containers",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
@router.get(
    "/containers",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
async def list_containers(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Container list endpoint for retrieving a list of containers.
    
    Follows the OpenAI Containers API spec:
    https://platform.openai.com/docs/api-reference/containers
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/containers?limit=20&order=desc" \
        -H "Authorization: Bearer sk-1234"
    ```
    
    Or specify provider via header or query param:
    ```bash
    curl -X GET "http://localhost:4000/v1/containers?custom_llm_provider=azure" \
        -H "Authorization: Bearer sk-1234"
    ```
    """
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

    # Read query parameters
    query_params = dict(request.query_params)
    data: Dict[str, Any] = {"query_params": query_params}

    # Extract custom_llm_provider using priority chain
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    
    # Add custom_llm_provider to data
    data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_containers",
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


@router.get(
    "/v1/containers/{container_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
@router.get(
    "/containers/{container_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
async def retrieve_container(
    request: Request,
    container_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Container retrieve endpoint for getting details of a specific container.
    
    Follows the OpenAI Containers API spec:
    https://platform.openai.com/docs/api-reference/containers
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/containers/cntr_123" \
        -H "Authorization: Bearer sk-1234"
    ```
    
    Or specify provider via header:
    ```bash
    curl -X GET "http://localhost:4000/v1/containers/cntr_123" \
        -H "Authorization: Bearer sk-1234" \
        -H "custom-llm-provider: azure"
    ```
    """
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

    # Include container_id in request data
    data: Dict[str, Any] = {"container_id": container_id}

    # Extract custom_llm_provider using priority chain
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    
    # Add custom_llm_provider to data
    data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aretrieve_container",
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


@router.delete(
    "/v1/containers/{container_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
@router.delete(
    "/containers/{container_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["containers"],
)
async def delete_container(
    request: Request,
    container_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Container delete endpoint for deleting a specific container.
    
    Follows the OpenAI Containers API spec:
    https://platform.openai.com/docs/api-reference/containers
    
    Example:
    ```bash
    curl -X DELETE "http://localhost:4000/v1/containers/cntr_123" \
        -H "Authorization: Bearer sk-1234"
    ```
    
    Or specify provider via header:
    ```bash
    curl -X DELETE "http://localhost:4000/v1/containers/cntr_123" \
        -H "Authorization: Bearer sk-1234" \
        -H "custom-llm-provider: azure"
    ```
    """
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

    # Include container_id in request data
    data: Dict[str, Any] = {"container_id": container_id}

    # Extract custom_llm_provider using priority chain
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or "openai"
    )
    
    # Add custom_llm_provider to data
    data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_container",
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

