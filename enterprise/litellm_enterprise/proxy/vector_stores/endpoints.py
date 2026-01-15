"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import copy
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    LiteLLM_ManagedVectorStoresTable,
    ResponseLiteLLM_ManagedVectorStore,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
    VectorStoreInfoRequest,
    VectorStoreUpdateRequest,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

router = APIRouter()


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
    from litellm.types.router import GenericLiteLLMParams

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

        # Safely handle JSON serialization of litellm_params
        litellm_params_json: Optional[str] = None
        _input_litellm_params: dict = vector_store.get("litellm_params", {}) or {}
        if _input_litellm_params is not None:
            litellm_params_dict = GenericLiteLLMParams(
                **_input_litellm_params
            ).model_dump(exclude_none=True)
            litellm_params_json = safe_dumps(litellm_params_dict)
            del vector_store["litellm_params"]

        _new_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.create(
                data={
                    **vector_store,
                    "litellm_params": litellm_params_json,
                }
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

    try:
        # Get vector stores from database (source of truth)
        # Only return what's in the database to ensure consistency across instances
        vector_stores_from_db = await VectorStoreRegistry._get_vector_stores_from_db(
            prisma_client=prisma_client
        )
        
        # Also clean up in-memory registry to remove any deleted vector stores
        if litellm.vector_store_registry is not None:
            db_vector_store_ids = {
                vs.get("vector_store_id") 
                for vs in vector_stores_from_db 
                if vs.get("vector_store_id")
            }
            # Remove any in-memory vector stores that no longer exist in database
            vector_stores_to_remove = []
            for vs in litellm.vector_store_registry.vector_stores:
                vs_id = vs.get("vector_store_id")
                if vs_id and vs_id not in db_vector_store_ids:
                    vector_stores_to_remove.append(vs_id)
            for vs_id in vector_stores_to_remove:
                litellm.vector_store_registry.delete_vector_store_from_registry(
                    vector_store_id=vs_id
                )
                verbose_proxy_logger.debug(
                    f"Removed deleted vector store {vs_id} from in-memory registry"
                )

        # Use database as single source of truth for listing
        combined_vector_stores: List[LiteLLM_ManagedVectorStore] = vector_stores_from_db

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


@router.post(
    "/vector_store/info",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ResponseLiteLLM_ManagedVectorStore,
)
async def get_vector_store_info(
    data: VectorStoreInfoRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return a single vector store's details"""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        if litellm.vector_store_registry is not None:
            vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if vector_store is not None:
                vector_store_metadata = vector_store.get("vector_store_metadata")
                # Parse metadata if it's a JSON string
                parsed_metadata: Optional[dict] = None
                if isinstance(vector_store_metadata, str):
                    parsed_metadata = json.loads(vector_store_metadata)
                elif isinstance(vector_store_metadata, dict):
                    parsed_metadata = vector_store_metadata

                vector_store_pydantic_obj = LiteLLM_ManagedVectorStoresTable(
                    vector_store_id=vector_store.get("vector_store_id") or "",
                    custom_llm_provider=vector_store.get("custom_llm_provider") or "",
                    vector_store_name=vector_store.get("vector_store_name") or None,
                    vector_store_description=vector_store.get(
                        "vector_store_description"
                    )
                    or None,
                    vector_store_metadata=parsed_metadata,
                    created_at=vector_store.get("created_at") or None,
                    updated_at=vector_store.get("updated_at") or None,
                    litellm_credential_name=vector_store.get("litellm_credential_name"),
                    litellm_params=vector_store.get("litellm_params") or None,
                )
                return {"vector_store": vector_store_pydantic_obj}

        vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                where={"vector_store_id": data.vector_store_id}
            )
        )
        if vector_store is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )

        vector_store_dict = vector_store.model_dump()  # type: ignore[attr-defined]
        return {"vector_store": vector_store_dict}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting vector store info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/update",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_vector_store(
    data: VectorStoreUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update vector store details"""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        update_data = data.model_dump(exclude_unset=True)
        vector_store_id = update_data.pop("vector_store_id")
        if update_data.get("vector_store_metadata") is not None:
            update_data["vector_store_metadata"] = safe_dumps(
                update_data["vector_store_metadata"]
            )

        updated = await prisma_client.db.litellm_managedvectorstorestable.update(
            where={"vector_store_id": vector_store_id},
            data=update_data,
        )

        updated_vs = LiteLLM_ManagedVectorStore(**updated.model_dump())

        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=updated_vs,
            )

        return {"vector_store": updated_vs}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
