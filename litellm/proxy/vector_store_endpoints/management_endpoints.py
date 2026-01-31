"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import copy
import json
from typing import Any, Dict, List, Optional

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
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.secret_managers.main import get_secret
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
    VectorStoreInfoRequest,
    VectorStoreUpdateRequest,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

router = APIRouter()


async def _resolve_embedding_config_from_db(
    embedding_model: str, prisma_client
) -> Optional[Dict[str, Any]]:
    """
    Resolve embedding config from database model configuration.
    
    If litellm_embedding_model is provided but litellm_embedding_config is not,
    this function looks up the model in the database and extracts api_key, api_base,
    and api_version from the model's litellm_params to build the embedding config.
    
    Args:
        embedding_model: The embedding model string (e.g., "text-embedding-ada-002" or "azure/text-embedding-3-large")
        prisma_client: The Prisma client instance
        
    Returns:
        Dictionary with api_key, api_base, and api_version if model found, None otherwise
    """
    if not embedding_model:
        return None
    
    # Extract model name - could be "text-embedding-ada-002" or "azure/text-embedding-3-large"
    # Try to find model by exact match first, then try without provider prefix
    model_name_candidates = [embedding_model]
    if "/" in embedding_model:
        # If it has a provider prefix, also try without it
        _, model_name = embedding_model.split("/", 1)
        model_name_candidates.append(model_name)
    
    # Try to find model in database
    for model_name in model_name_candidates:
        try:
            db_model = await prisma_client.db.litellm_proxymodeltable.find_first(
                where={"model_name": model_name}
            )
            
            if db_model and db_model.litellm_params:
                # Extract litellm_params (could be dict or JSON string)
                model_params = db_model.litellm_params
                if isinstance(model_params, str):
                    model_params = json.loads(model_params)
                
                # Decrypt values from database (similar to how proxy_server.py does it)
                # Values stored in DB are encrypted, so we need to decrypt them first
                decrypted_params = {}
                if isinstance(model_params, dict):
                    for k, v in model_params.items():
                        if isinstance(v, str):
                            # Decrypt value - returns original value if decryption fails or no key is set
                            decrypted_value = decrypt_value_helper(
                                value=v, key=k, return_original_value=True
                            )
                            decrypted_params[k] = decrypted_value
                        else:
                            decrypted_params[k] = v
                else:
                    decrypted_params = model_params
                
                # Build embedding config from model params
                embedding_config = {}
                
                # Extract api_key
                api_key = decrypted_params.get("api_key")
                if api_key:
                    # Handle os.environ/ prefix (after decryption, values may be os.environ/ prefixed)
                    if isinstance(api_key, str) and api_key.startswith("os.environ/"):
                        api_key = get_secret(api_key)
                    embedding_config["api_key"] = api_key
                
                # Extract api_base
                api_base = decrypted_params.get("api_base")
                if api_base:
                    # Handle os.environ/ prefix (after decryption, values may be os.environ/ prefixed)
                    if isinstance(api_base, str) and api_base.startswith("os.environ/"):
                        api_base = get_secret(api_base)
                    embedding_config["api_base"] = api_base
                
                # Extract api_version
                api_version = decrypted_params.get("api_version")
                if api_version:
                    embedding_config["api_version"] = api_version
                
                # Only return config if we have at least api_key or api_base
                if embedding_config:
                    verbose_proxy_logger.debug(
                        f"Resolved embedding config from database model {model_name}: {list(embedding_config.keys())}"
                    )
                    return embedding_config
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error resolving embedding config for model {model_name}: {str(e)}"
            )
            continue
    
    return None


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
            # Auto-resolve embedding config if embedding model is provided but config is not
            embedding_model = _input_litellm_params.get("litellm_embedding_model")
            if embedding_model and not _input_litellm_params.get("litellm_embedding_config"):
                resolved_config = await _resolve_embedding_config_from_db(
                    embedding_model=embedding_model,
                    prisma_client=prisma_client
                )
                if resolved_config:
                    _input_litellm_params["litellm_embedding_config"] = resolved_config
                    verbose_proxy_logger.info(
                        f"Auto-resolved embedding config for model {embedding_model}"
                    )
            
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
