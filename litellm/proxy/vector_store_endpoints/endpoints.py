from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import litellm
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    LiteLLM_ManagedVectorStore,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import jsonify_object
from litellm.types.vector_stores import IndexCreateRequest

router = APIRouter()
########################################################
# OpenAI Compatible Endpoints
########################################################


def _check_vector_store_access(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Check if the user has access to the vector store based on team membership.
    
    Args:
        vector_store: The vector store to check access for
        user_api_key_dict: User API key authentication info
        
    Returns:
        True if user has access, False otherwise
        
    Access rules:
    - If vector store has no team_id, it's accessible to all (legacy behavior)
    - If user's team_id matches the vector store's team_id, access is granted
    - Otherwise, access is denied
    """
    vector_store_team_id = vector_store.get("team_id")
    
    # If vector store has no team_id, it's accessible to all (legacy behavior)
    if vector_store_team_id is None:
        return True
    
    # Check if user's team matches the vector store's team
    user_team_id = user_api_key_dict.team_id
    if user_team_id == vector_store_team_id:
        return True
    
    return False


def _update_request_data_with_litellm_managed_vector_store_registry(
    data: Dict,
    vector_store_id: str,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
) -> Dict:
    """
    Update the request data with the litellm managed vector store registry.
    
    Args:
        data: Request data to update
        vector_store_id: ID of the vector store
        user_api_key_dict: User API key authentication info for access control
        
    Raises:
        HTTPException: If user doesn't have access to the vector store
    """
    if litellm.vector_store_registry is not None:
        vector_store_to_run: Optional[LiteLLM_ManagedVectorStore] = (
            litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
        )
        if vector_store_to_run is not None:
            # Check access control if user_api_key_dict is provided
            if user_api_key_dict is not None:
                if not _check_vector_store_access(vector_store_to_run, user_api_key_dict):
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: You do not have permission to access this vector store",
                    )
            
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


@router.post(
    "/v1/vector_stores/{vector_store_id:path}/search",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/vector_stores/{vector_store_id:path}/search", dependencies=[Depends(user_api_key_auth)]
)
async def vector_store_search(
    request: Request,
    vector_store_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Search a vector store.

    API Reference:
    https://platform.openai.com/docs/api-reference/vector-stores/search
    """
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

    # Check for legacy vector store registry (non-managed vector stores)
    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id, user_api_key_dict=user_api_key_dict
    )

    # The managed_vector_stores pre-call hook will handle:
    # 1. Decoding managed vector store IDs
    # 2. Extracting model and provider resource ID
    # 3. Setting up proper routing
    # 4. Authentication checks
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_search",
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


@router.post("/v1/vector_stores", dependencies=[Depends(user_api_key_auth)])
@router.post("/vector_stores", dependencies=[Depends(user_api_key_auth)])
async def vector_store_create(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a vector store.

    API Reference:
    https://platform.openai.com/docs/api-reference/vector-stores/create
    
    Supports target_model_names parameter for creating vector stores across multiple models:
    ```json
    {
        "name": "my-vector-store",
        "target_model_names": "gpt-4,gemini-2.0"
    }
    ```
    """
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
    
    # Check for target_model_names parameter
    target_model_names = data.pop("target_model_names", None)
    
    if target_model_names:
        # Use managed vector stores for multi-model support
        if isinstance(target_model_names, str):
            target_model_names_list = [m.strip() for m in target_model_names.split(",")]
        elif isinstance(target_model_names, list):
            target_model_names_list = target_model_names
        else:
            raise HTTPException(
                status_code=400,
                detail="target_model_names must be a comma-separated string or list of model names",
            )
        
        # Get managed vector stores hook
        managed_vector_stores: Any = proxy_logging_obj.get_proxy_hook("managed_vector_stores")
        if managed_vector_stores is None:
            raise HTTPException(
                status_code=500,
                detail="Managed vector stores not configured. Please ensure the proxy is initialized with database support.",
            )
        
        if llm_router is None:
            raise HTTPException(
                status_code=500,
                detail="LLM Router not initialized. Ensure models are added to proxy.",
            )
        
        # Create vector store across multiple models
        response = await managed_vector_stores.acreate_vector_store(
            create_request=data,
            llm_router=llm_router,
            target_model_names_list=target_model_names_list,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            user_api_key_dict=user_api_key_dict,
        )
        
        return response
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avector_store_create",
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


@router.post(
    "/v1/indexes",
    dependencies=[Depends(user_api_key_auth)],
)
async def index_create(
    request: Request,
    index_create_request: IndexCreateRequest,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create an index. Just writes the index to the database.

    ```bash
    curl -L -X POST 'http://0.0.0.0:4000/indexes/create' \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer sk-1234' \
        -H 'LiteLLM-Beta: indexes_beta=v1' \
        -d '{ 
            "index_name": "dall-e-3",
            "vector_store_index": "real-index-name",
            "vector_store_name": "azure-ai-search"
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )
    ## 1. check if index already exists
    existing_index = (
        await prisma_client.db.litellm_managedvectorstoreindextable.find_unique(
            where={"index_name": index_create_request.index_name}
        )
    )

    ## 2. set created_by and updated_by

    if existing_index is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Index {index_create_request.index_name} already exists",
        )

    ## 2. create index
    index_data = index_create_request.model_dump(exclude_none=True)
    index_data["created_by"] = user_api_key_dict.user_id
    index_data["updated_by"] = user_api_key_dict.user_id
    new_index = await prisma_client.db.litellm_managedvectorstoreindextable.create(
        data=jsonify_object(index_data)
    )

    return new_index.model_dump()
