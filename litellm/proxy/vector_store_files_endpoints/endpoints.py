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
from litellm.proxy.openai_files_endpoints.common_utils import (
    handle_model_based_routing,
    prepare_data_with_credentials,
)
from litellm.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_files_endpoint,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.router import Router

router = APIRouter()


def _update_request_data_with_managed_file_id(
    data: Dict,
    file_id: str,
    request: Request,
    llm_router: Optional["Router"] = None,
) -> tuple[Dict, Optional[str]]:
    """
    Update request data with model routing information from managed file ID.
    
    This function handles two types of file IDs:
    1. Simple encoded file IDs (format: litellm:{file_id};model,{model})
    2. Unified managed file IDs (format: litellm_proxy:{mime};unified_id,{uuid};...;llm_output_file_id,{file_id};...)
    
    For unified managed file IDs, it:
    - Decodes the unified ID to extract the actual provider file ID (llm_output_file_id)
    - Extracts the model routing information (target_model_names)
    - Updates data with credentials for the correct deployment
    
    Args:
        data: Request data to update
        file_id: File ID (can be managed/encoded or regular)
        request: FastAPI request object
        llm_router: LiteLLM router for credential lookup (required for managed files)
        
    Returns:
        Tuple of (updated request data, original_managed_file_id)
        - original_managed_file_id is the original file_id if it was managed/encoded, None otherwise
    """
    import re

    from litellm import verbose_logger
    from litellm.llms.base_llm.managed_resources.utils import (
        is_base64_encoded_unified_id,
        parse_unified_id,
    )

    # First, check if this is a unified managed file ID (base64 encoded)
    decoded_id = is_base64_encoded_unified_id(file_id)
    
    if decoded_id:
        # This is a unified managed file ID
        verbose_logger.debug(
            f"Processing unified managed file ID: {file_id}"
        )
        
        # Parse the unified ID to extract components
        parsed_id = parse_unified_id(file_id)
        
        if parsed_id:
            target_model_names = parsed_id.get("target_model_names", [])
            
            # Extract the actual provider file ID from llm_output_file_id field
            # Format: litellm_proxy:...;llm_output_file_id,{actual_file_id};...
            llm_output_file_id = None
            try:
                match = re.search(r"llm_output_file_id,([^;]+)", decoded_id)
                if match:
                    llm_output_file_id = match.group(1).strip()
            except Exception:
                pass
            
            verbose_logger.debug(
                f"Decoded unified file ID - target_model_names: {target_model_names}, llm_output_file_id: {llm_output_file_id}"
            )
            
            # Set the model for routing
            if target_model_names and len(target_model_names) > 0:
                routing_model = target_model_names[0]
                data["model"] = routing_model
                
                # Get credentials for the model
                if llm_router:
                    credentials = llm_router.get_deployment_credentials_with_provider(
                        model_id=routing_model
                    )
                    if credentials:
                        prepare_data_with_credentials(
                            data=data,
                            credentials=credentials,
                            file_id=llm_output_file_id,  # Use the actual provider file ID
                        )
                        verbose_logger.info(
                            f"Routing vector store file operation to model: {routing_model}, file_id: {file_id} -> {llm_output_file_id}"
                        )
                        return data, file_id  # Return original managed file ID
            
            # If we extracted the provider file ID but no routing, still use it
            if llm_output_file_id:
                data["file_id"] = llm_output_file_id
                verbose_logger.debug(
                    f"Replaced unified file ID with provider file ID: {llm_output_file_id}"
                )
                return data, file_id  # Return original managed file ID
        
        return data, file_id if decoded_id else None
    
    # Fall back to simple encoded file ID handling (format: litellm:{file_id};model,{model})
    should_route, model_used, original_file_id, credentials = handle_model_based_routing(
        file_id=file_id,
        request=request,
        llm_router=llm_router,
        data=data,
        check_file_id_encoding=True,
    )
    
    if should_route:
        # Use model-based routing with credentials from config
        prepare_data_with_credentials(
            data=data,
            credentials=credentials,  # type: ignore
            file_id=original_file_id,  # Use decoded file ID if from encoded ID
        )
        
        verbose_logger.debug(
            f"Routing vector store file operation using model: {model_used}"
            + (f", file_id: {file_id} -> {original_file_id}" if original_file_id else "")
        )
        return data, file_id  # Return original file ID for response replacement
    
    return data, None


def _replace_file_id_in_response(response, original_file_id: str):
    """
    Replace the provider file ID in the response with the original managed file ID.
    
    This ensures that when a user sends a managed file ID, they get back the same
    managed file ID in the response, not the decoded provider file ID.
    
    Args:
        response: The response object from the provider
        original_file_id: The original managed file ID to restore
        
    Returns:
        Modified response with original file ID
    """
    if response is None:
        return response
    
    # Handle different response types
    if isinstance(response, dict):
        # For dict responses (e.g., VectorStoreFileDeleteResponse)
        if "id" in response:
            response["id"] = original_file_id
        if "file_id" in response:
            response["file_id"] = original_file_id
    elif hasattr(response, "id"):
        # For object responses (e.g., VectorStoreFileObject)
        response.id = original_file_id
    elif hasattr(response, "file_id"):
        response.file_id = original_file_id
    
    return response


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

    # Handle managed file IDs if present in request body
    original_managed_file_id = None
    if "file_id" in data:
        data, original_managed_file_id = _update_request_data_with_managed_file_id(
            data=data, file_id=data["file_id"], request=request, llm_router=llm_router
        )

    # Then handle managed vector store IDs
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
        response = await processor.base_process_llm_request(
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
        
        # Replace provider file ID with original managed file ID in response
        if original_managed_file_id:
            response = _replace_file_id_in_response(response, original_managed_file_id)
        
        return response
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

    # Handle managed file IDs first
    data, original_managed_file_id = _update_request_data_with_managed_file_id(
        data=data, file_id=file_id, request=request, llm_router=llm_router
    )

    # Then handle managed vector store IDs
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
        response = await processor.base_process_llm_request(
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
        
        # Replace provider file ID with original managed file ID in response
        if original_managed_file_id:
            response = _replace_file_id_in_response(response, original_managed_file_id)
        
        return response
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

    # Handle managed file IDs first
    data, original_managed_file_id = _update_request_data_with_managed_file_id(
        data=data, file_id=file_id, request=request, llm_router=llm_router
    )

    # Then handle managed vector store IDs
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
        response = await processor.base_process_llm_request(
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
        
        # Replace provider file ID with original managed file ID in response
        if original_managed_file_id:
            response = _replace_file_id_in_response(response, original_managed_file_id)
        
        return response
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

    # Handle managed file IDs first
    data, original_managed_file_id = _update_request_data_with_managed_file_id(
        data=data, file_id=file_id, request=request, llm_router=llm_router
    )

    # Then handle managed vector store IDs
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
        response = await processor.base_process_llm_request(
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
        
        # Replace provider file ID with original managed file ID in response
        if original_managed_file_id:
            response = _replace_file_id_in_response(response, original_managed_file_id)
        
        return response
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

    # Handle managed file IDs first
    data, original_managed_file_id = _update_request_data_with_managed_file_id(
        data=data, file_id=file_id, request=request, llm_router=llm_router
    )

    # Then handle managed vector store IDs
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
        response = await processor.base_process_llm_request(
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
        
        # Replace provider file ID with original managed file ID in response
        if original_managed_file_id:
            response = _replace_file_id_in_response(response, original_managed_file_id)
        
        return response
    except Exception as e:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
