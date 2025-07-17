"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import copy
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_base_llm_request_processing import (
    ProxyBaseLLMRequestProcessing,
)
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

router = APIRouter()

########################################################
# OpenAI Compatible Endpoints
########################################################

@router.post("/v1/vector_stores/{vector_store_id}/search", dependencies=[Depends(user_api_key_auth)])
@router.post("/vector_stores/{vector_store_id}/search", dependencies=[Depends(user_api_key_auth)])
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
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="vector_store_search",
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
            route_type="vector_store_create",
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

########################################################
# Management Endpoints
########################################################
@router.post(
    "/vector_store/new",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new vector store.

    Parameters:
    - vector_store_id: str - Unique identifier for the vector store
    - custom_llm_provider: str - Provider of the vector store
    - vector_store_name: Optional[str] - Name of the vector store
    - vector_store_description: Optional[str] - Description of the vector store
    - vector_store_metadata: Optional[Dict] - Additional metadata for the vector store
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if vector store already exists
        existing_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                where={"vector_store_id": vector_store.get("vector_store_id")}
            )
        )
        if existing_vector_store is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Vector store with ID {vector_store.get('vector_store_id')} already exists",
            )

        if vector_store.get("vector_store_metadata") is not None:
            vector_store["vector_store_metadata"] = safe_dumps(
                vector_store.get("vector_store_metadata")
            )

        _new_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.create(
                data=vector_store
            )
        )

        new_vector_store: LiteLLM_ManagedVectorStore = LiteLLM_ManagedVectorStore(
            **_new_vector_store.model_dump()
        )

        # Add vector store to registry
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.add_vector_store_to_registry(
                vector_store=new_vector_store
            )

        return {
            "status": "success",
            "message": f"Vector store {vector_store.get('vector_store_id')} created successfully",
            "vector_store": new_vector_store,
        }
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
async def list_vector_stores(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = 1,
    page_size: int = 100,
):
    """
    List all available vector stores with optional filtering and pagination.
    Combines both in-memory vector stores and those stored in the database.

    Parameters:
    - page: int - Page number for pagination (default: 1)
    - page_size: int - Number of items per page (default: 100)
    """
    from litellm.proxy.proxy_server import prisma_client

    seen_vector_store_ids = set()

    try:
        # Get in-memory vector stores
        in_memory_vector_stores: List[LiteLLM_ManagedVectorStore] = []
        if litellm.vector_store_registry is not None:
            in_memory_vector_stores = copy.deepcopy(
                litellm.vector_store_registry.vector_stores
            )

        # Get vector stores from database
        vector_stores_from_db = await VectorStoreRegistry._get_vector_stores_from_db(
            prisma_client=prisma_client
        )

        # Combine in-memory and database vector stores
        combined_vector_stores: List[LiteLLM_ManagedVectorStore] = []
        for vector_store in in_memory_vector_stores + vector_stores_from_db:
            vector_store_id = vector_store.get("vector_store_id", None)
            if vector_store_id not in seen_vector_store_ids:
                combined_vector_stores.append(vector_store)
                seen_vector_store_ids.add(vector_store_id)

        total_count = len(combined_vector_stores)
        total_pages = (total_count + page_size - 1) // page_size

        # Format response using LiteLLM_ManagedVectorStoreListResponse
        response = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=combined_vector_stores,
            total_count=total_count,
            current_page=page,
            total_pages=total_pages,
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/delete",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_vector_store(
    data: VectorStoreDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a vector store.

    Parameters:
    - vector_store_id: str - ID of the vector store to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if vector store exists
        existing_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                where={"vector_store_id": data.vector_store_id}
            )
        )
        if existing_vector_store is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )

        # Delete vector store
        await prisma_client.db.litellm_managedvectorstorestable.delete(
            where={"vector_store_id": data.vector_store_id}
        )

        # Delete vector store from registry
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.delete_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )

        return {"message": f"Vector store {data.vector_store_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
