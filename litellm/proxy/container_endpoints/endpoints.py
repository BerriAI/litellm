#### Container Endpoints #####

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse

from litellm._uuid import uuid
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)

router = APIRouter()


def _parse_target_model_names_from_body(value: Any) -> List[str]:
    """Parse JSON `target_model_names` (comma-separated string or list of names)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


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

    target_model_names_list = _parse_target_model_names_from_body(
        data.get("target_model_names")
    )

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        if target_model_names_list:
            from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
            from litellm.utils import Rules, function_setup

            data.pop("target_model_names", None)

            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )

            managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_obj is None or not hasattr(
                managed_files_obj, "acreate_container"
            ):
                raise ProxyException(
                    message=(
                        "Managed containers with target_model_names require the "
                        "managed_files (enterprise) hook with acreate_container support."
                    ),
                    type="None",
                    param="managed_files",
                    code=503,
                )

            hook_data = await add_litellm_data_to_request(
                data=dict(data),
                request=request,
                general_settings=general_settings,
                user_api_key_dict=user_api_key_dict,
                version=version,
                proxy_config=proxy_config,
            )
            # function_setup() requires litellm_call_id (see litellm.utils.function_setup);
            # common_processing_pre_call_logic sets this after add_litellm_data_to_request.
            if not hook_data.get("litellm_call_id"):
                hook_data["litellm_call_id"] = request.headers.get(
                    "x-litellm-call-id", str(uuid.uuid4())
                )
            start_time = datetime.now()
            logging_obj, hook_data = function_setup(
                original_function="acreate_container",
                rules_obj=Rules(),
                start_time=start_time,
                **hook_data,
            )
            hook_data["litellm_logging_obj"] = logging_obj
            hook_data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                data=hook_data,
                call_type="acreate_container",  # type: ignore[arg-type]
            )
            if hook_data is None:
                raise HTTPException(
                    status_code=500,
                    detail={"error": "Invalid request after pre-call hook"},
                )

            response = await managed_files_obj.acreate_container(
                llm_router=llm_router,
                request_data=hook_data,
                target_model_names_list=target_model_names_list,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                user_api_key_dict=user_api_key_dict,
            )

            asyncio.create_task(
                proxy_logging_obj.update_request_status(
                    litellm_call_id=hook_data.get("litellm_call_id", ""), status="success"
                )
            )
            hooked = await proxy_logging_obj.post_call_success_hook(
                data=hook_data,
                user_api_key_dict=user_api_key_dict,
                response=response,
            )
            if hooked is not None:
                response = hooked

            hidden_params = getattr(response, "_hidden_params", {}) or {}
            model_id = hidden_params.get("model_id", None) or ""
            cache_key = hidden_params.get("cache_key", None) or ""
            api_base = hidden_params.get("api_base", None) or ""

            fastapi_response.headers.update(
                ProxyBaseLLMRequestProcessing.get_custom_headers(
                    user_api_key_dict=user_api_key_dict,
                    call_id=hook_data.get("litellm_call_id"),
                    model_id=model_id,
                    cache_key=cache_key,
                    api_base=api_base,
                    version=version,
                    model_region=getattr(
                        user_api_key_dict, "allowed_model_region", ""
                    ),
                    litellm_logging_obj=logging_obj,
                )
            )
            return response

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


# Register JSON-configured container file endpoints
from litellm.proxy.container_endpoints.handler_factory import (
    register_container_file_endpoints,
)

register_container_file_endpoints(router)
