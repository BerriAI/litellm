from typing import Dict, Optional

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


def _update_request_data_with_litellm_managed_vector_store_registry(
    data: Dict,
    vector_store_id: str,
) -> Dict:
    """
    Update the request data with the litellm managed vector store registry.

    """
    if litellm.vector_store_registry is not None:
        vector_store_to_run: Optional[LiteLLM_ManagedVectorStore] = (
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


@router.post(
    "/v1/vector_stores/{vector_store_id}/search",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/vector_stores/{vector_store_id}/search", dependencies=[Depends(user_api_key_auth)]
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

    data = _update_request_data_with_litellm_managed_vector_store_registry(
        data=data, vector_store_id=vector_store_id
    )

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
