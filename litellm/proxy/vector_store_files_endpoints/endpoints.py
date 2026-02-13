from typing import TYPE_CHECKING, Dict, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)
from litellm.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_files_endpoint,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.router import Router

router = APIRouter()


def _update_request_data_with_litellm_managed_vector_store_registry(
    data: Dict,
    vector_store_id: str,
    llm_router: Optional["Router"] = None,
) -> Dict:
    """
    Update request data with model routing information from managed vector store.
    
    This function handles two types of vector stores:
    1. Legacy vector stores from registry (non-managed)
    2. Managed vector stores with unified IDs (requires decoding)
    
    For managed vector stores, this function:
    - Decodes the unified vector store ID
    - Extracts the model_id and provider resource ID
    - Sets data["model"] so the router can use the correct deployment credentials
    - Replaces the unified ID with the provider-specific ID
    
    Args:
        data: Request data to update
        vector_store_id: Vector store ID (can be unified or legacy)
        llm_router: LiteLLM router for credential lookup (required for managed vector stores)
        
    Returns:
        Updated request data with model routing information
    """
    from litellm import verbose_logger
    from litellm.llms.base_llm.managed_resources.utils import (
        is_base64_encoded_unified_id,
        parse_unified_id,
    )

    # Check if this is a managed vector store ID (base64 encoded unified ID)
    decoded_id = is_base64_encoded_unified_id(vector_store_id)
    
    if decoded_id:
        # This is a managed vector store - decode and extract routing information
        verbose_logger.debug(
            f"Processing managed vector store ID: {vector_store_id}"
        )
        
        parsed_id = parse_unified_id(vector_store_id)
        
        if parsed_id:
            model_id = parsed_id.get("model_id")
            provider_resource_id = parsed_id.get("provider_resource_id")
            target_model_names = parsed_id.get("target_model_names", [])
            
            verbose_logger.debug(
                f"Decoded vector store - model_id: {model_id}, provider_resource_id: {provider_resource_id}, target_model_names: {target_model_names}"
            )
            
            # Set the model for routing - this tells the router which deployment to use
            # The router will automatically get the credentials from the deployment
            routing_model = None
            if model_id:
                routing_model = model_id
            elif target_model_names and len(target_model_names) > 0:
                routing_model = target_model_names[0]
            
            if routing_model:
                data["model"] = routing_model
                verbose_logger.info(
                    f"Routing vector store files operation to model: {routing_model}"
                )
            
            # Replace unified vector store ID with provider resource ID
            if provider_resource_id:
                data["vector_store_id"] = provider_resource_id
                verbose_logger.debug(
                    f"Replaced unified vector store ID with provider resource ID: {provider_resource_id}"
                )
        
        return data
    
    # Legacy path: Check vector store registry for non-managed vector stores
    if litellm.vector_store_registry is not None:
        vector_store_to_run = (
            litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
        )
        if vector_store_to_run is not None:
            if "custom_llm_provider" in vector_store_to_run:
                data["custom_llm_provider"] = vector_store_to_run.get(
                    "custom_llm_provider"
                )
            if "litellm_credential_name" in vector_store_to_run:
                data["litellm_credential_name"] = vector_store_to_run.get(
                    "litellm_credential_name"
                )
            if "litellm_params" in vector_store_to_run:
                litellm_params = vector_store_to_run.get("litellm_params", {}) or {}
                data.update(litellm_params)
    
    return data


async def _resolve_provider(
    *,
    data: Dict,
    request: Request,
) -> Optional[LlmProviders]:
    provider = (
        data.get("custom_llm_provider")
        or get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
    )

    if provider is None and request.method in {"POST", "PUT", "PATCH"}:
        provider = await get_custom_llm_provider_from_request_body(request=request)

    if provider is None:
        provider = "openai"

    try:
        return LlmProviders(provider)
    except Exception:
        return None


def _maybe_check_permissions(
    *,
    provider: Optional[LlmProviders],
    vector_store_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    if provider is None:
        return
    metadata = user_api_key_dict.metadata or {}
    team_metadata = user_api_key_dict.team_metadata or {}
    if not metadata.get("allowed_vector_store_indexes") and not team_metadata.get(
        "allowed_vector_store_indexes"
    ):
        return
    is_allowed_to_call_vector_store_files_endpoint(
        provider=provider,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )


@router.post(
    "/v1/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.post(
    "/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_create(
    vector_store_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        _read_request_body,
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

    data = await _read_request_body(request=request)
    if "vector_store_id" not in data:
        data["vector_store_id"] = vector_store_id

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_create",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_list(
    vector_store_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
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
    data: Dict[str, Optional[str]] = {"vector_store_id": vector_store_id}
    data.update(query_params)

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_list",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_retrieve(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
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

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_retrieve",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_content(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
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

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_content",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.post(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_update(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        _read_request_body,
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

    data = await _read_request_body(request=request)
    data["vector_store_id"] = vector_store_id
    data["file_id"] = file_id

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_update",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
@router.delete(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["vector_store_files"],
)
async def vector_store_file_delete(
    vector_store_id: str,
    file_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
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

    data: Dict[str, str] = {
        "vector_store_id": vector_store_id,
        "file_id": file_id,
    }

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, llm_router=llm_router
    )

    provider_enum = await _resolve_provider(data=data, request=request)

    _maybe_check_permissions(
        provider=provider_enum,
        vector_store_id=vector_store_id,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    if provider_enum is not None and "custom_llm_provider" not in data:
        data["custom_llm_provider"] = provider_enum.value

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_file_delete",
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
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
