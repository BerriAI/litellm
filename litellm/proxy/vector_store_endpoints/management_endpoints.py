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


def _resolve_embedding_config_from_router(
    embedding_model: str, llm_router
) -> Optional[Dict[str, Any]]:
    """
    Resolve embedding config from router's config-defined models.
    
    Config-defined models (from proxy_config.yaml) are stored in the router's model_list,
    not in the database. This function looks up the model in the router and extracts
    api_key, api_base, and api_version from the deployment's litellm_params.
    
    Args:
        embedding_model: The embedding model string (e.g., "text-embedding-ada-002" or "azure/text-embedding-3-large")
        llm_router: The LiteLLM router instance
        
    Returns:
        Dictionary with api_key, api_base, and api_version if model found, None otherwise
    """
    if not embedding_model or llm_router is None:
        return None
    
    # Extract model name candidates - could be "text-embedding-ada-002" or "azure/text-embedding-3-large"
    # Try exact match first, then try without provider prefix
    model_name_candidates = [embedding_model]
    if "/" in embedding_model:
        # If it has a provider prefix, also try without it
        _, model_name = embedding_model.split("/", 1)
        model_name_candidates.append(model_name)
    
    # Try to find model in router
    for model_name in model_name_candidates:
        try:
            # Try to get deployment by model group name (model_name in config)
            deployment = llm_router.get_deployment_by_model_group_name(
                model_group_name=model_name
            )
            
            if deployment is not None and deployment.litellm_params is not None:
                litellm_params = deployment.litellm_params
                
                # Build embedding config from model params
                embedding_config: Dict[str, Any] = {}
                
                # Extract api_key
                api_key = getattr(litellm_params, "api_key", None)
                if api_key:
                    # Handle os.environ/ prefix
                    if isinstance(api_key, str) and api_key.startswith("os.environ/"):
                        api_key = get_secret(api_key)
                    embedding_config["api_key"] = api_key
                
                # Extract api_base
                api_base = getattr(litellm_params, "api_base", None)
                if api_base:
                    # Handle os.environ/ prefix
                    if isinstance(api_base, str) and api_base.startswith("os.environ/"):
                        api_base = get_secret(api_base)
                    embedding_config["api_base"] = api_base
                
                # Extract api_version
                api_version = getattr(litellm_params, "api_version", None)
                if api_version:
                    embedding_config["api_version"] = api_version

                project_id = getattr(litellm_params, "project_id", None)
                if project_id:
                    embedding_config["project_id"] = project_id
                
                # Only return config if we have at least api_key or api_base
                if embedding_config:
                    verbose_proxy_logger.debug(
                        f"Resolved embedding config from router model {model_name}: {list(embedding_config.keys())}"
                    )
                    return embedding_config
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error resolving embedding config from router for model {model_name}: {str(e)}"
            )
            continue
    
    return None


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


async def _resolve_embedding_config(
    embedding_model: str, prisma_client, llm_router=None
) -> Optional[Dict[str, Any]]:
    """
    Resolve embedding config from either router (config-defined) or database models.
    
    This function first checks the router for config-defined models, then falls back
    to the database. This allows users to use models defined in either location.
    
    Args:
        embedding_model: The embedding model string (e.g., "text-embedding-ada-002" or "azure/text-embedding-3-large")
        prisma_client: The Prisma client instance
        llm_router: The LiteLLM router instance (optional, will be imported if not provided)
        
    Returns:
        Dictionary with api_key, api_base, and api_version if model found, None otherwise
    """
    if not embedding_model:
        return None
    
    # Import llm_router if not provided
    if llm_router is None:
        try:
            from litellm.proxy.proxy_server import llm_router
        except ImportError:
            llm_router = None
    
    # First try to resolve from router (config-defined models)
    if llm_router is not None:
        router_config = _resolve_embedding_config_from_router(
            embedding_model=embedding_model,
            llm_router=llm_router
        )
        if router_config:
            verbose_proxy_logger.debug(
                f"Resolved embedding config from router for model {embedding_model}"
            )
            return router_config
    
    # Fall back to database
    if prisma_client is not None:
        db_config = await _resolve_embedding_config_from_db(
            embedding_model=embedding_model,
            prisma_client=prisma_client
        )
        if db_config:
            verbose_proxy_logger.debug(
                f"Resolved embedding config from database for model {embedding_model}"
            )
            return db_config
    
    verbose_proxy_logger.debug(
        f"Could not resolve embedding config for model {embedding_model} from router or database"
    )
    return None


########################################################
# Helper Functions
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


async def create_vector_store_in_db(
    vector_store_id: str,
    custom_llm_provider: str,
    prisma_client,
    vector_store_name: Optional[str] = None,
    vector_store_description: Optional[str] = None,
    vector_store_metadata: Optional[Dict] = None,
    litellm_params: Optional[Dict] = None,
    litellm_credential_name: Optional[str] = None,
    team_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> LiteLLM_ManagedVectorStore:
    """
    Helper function to create a vector store in the database.
    
    This function handles:
    - Checking if vector store already exists
    - Creating the vector store in the database
    - Adding it to the vector store registry
    
    Returns:
        LiteLLM_ManagedVectorStore: The created vector store object
        
    Raises:
        HTTPException: If vector store already exists or database error occurs
    """
    from litellm.types.router import GenericLiteLLMParams
    
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    # Check if vector store already exists
    existing_vector_store = (
        await prisma_client.db.litellm_managedvectorstorestable.find_unique(
            where={"vector_store_id": vector_store_id}
        )
    )
    if existing_vector_store is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store with ID {vector_store_id} already exists",
        )
    
    # Prepare data for database
    data_to_create: Dict[str, Any] = {
        "vector_store_id": vector_store_id,
        "custom_llm_provider": custom_llm_provider,
    }
    
    if vector_store_name is not None:
        data_to_create["vector_store_name"] = vector_store_name
    if vector_store_description is not None:
        data_to_create["vector_store_description"] = vector_store_description
    if vector_store_metadata is not None:
        data_to_create["vector_store_metadata"] = safe_dumps(vector_store_metadata)
    if litellm_credential_name is not None:
        data_to_create["litellm_credential_name"] = litellm_credential_name
    if team_id is not None:
        data_to_create["team_id"] = team_id
    if user_id is not None:
        data_to_create["user_id"] = user_id
    
    # Handle litellm_params - always provide at least an empty dict
    if litellm_params:
        # Auto-resolve embedding config if embedding model is provided but config is not
        embedding_model = litellm_params.get("litellm_embedding_model")
        if embedding_model and not litellm_params.get("litellm_embedding_config"):
            resolved_config = await _resolve_embedding_config(
                embedding_model=embedding_model,
                prisma_client=prisma_client
            )
            if resolved_config:
                litellm_params["litellm_embedding_config"] = resolved_config
                verbose_proxy_logger.info(
                    f"Auto-resolved embedding config for model {embedding_model}"
                )
        
        litellm_params_dict = GenericLiteLLMParams(
            **litellm_params
        ).model_dump(exclude_none=True)
        data_to_create["litellm_params"] = safe_dumps(litellm_params_dict)
    else:
        # Provide empty dict if no litellm_params provided
        data_to_create["litellm_params"] = safe_dumps({})
    
    # Create in database
    _new_vector_store = (
        await prisma_client.db.litellm_managedvectorstorestable.create(
            data=data_to_create
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
    
    verbose_proxy_logger.info(
        f"Vector store {vector_store_id} created in database successfully"
    )
    
    return new_vector_store


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

    try:
        vector_store_id = vector_store.get("vector_store_id")
        custom_llm_provider = vector_store.get("custom_llm_provider")
        
        if not vector_store_id or not custom_llm_provider:
            raise HTTPException(
                status_code=400,
                detail="vector_store_id and custom_llm_provider are required"
            )
        
        # Extract and validate metadata
        metadata = vector_store.get("vector_store_metadata")
        validated_metadata: Optional[Dict] = None
        if metadata is not None and isinstance(metadata, dict):
            validated_metadata = metadata
        
        new_vector_store = await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider=custom_llm_provider,
            prisma_client=prisma_client,
            vector_store_name=vector_store.get("vector_store_name"),
            vector_store_description=vector_store.get("vector_store_description"),
            vector_store_metadata=validated_metadata,
            litellm_params=vector_store.get("litellm_params"),
            litellm_credential_name=vector_store.get("litellm_credential_name"),
            team_id=user_api_key_dict.team_id,
            user_id=user_api_key_dict.user_id,
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
@router.get(
    "/v1/vector_store/list",
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
    Database is the source of truth - deleted stores are removed from memory, updated stores sync to memory.

    Parameters:
    - page: int - Page number for pagination (default: 1)
    - page_size: int - Number of items per page (default: 100)
    """
    from litellm.proxy.proxy_server import prisma_client

    vector_store_map: Dict[str, LiteLLM_ManagedVectorStore] = {}
    db_vector_store_ids: set = set()

    try:
        # Get vector stores from database first (source of truth)
        vector_stores_from_db = await VectorStoreRegistry._get_vector_stores_from_db(
            prisma_client=prisma_client
        )
        
        # Build map from database vector stores
        for vector_store in vector_stores_from_db:
            vector_store_id = vector_store.get("vector_store_id", None)
            if vector_store_id:
                vector_store_map[vector_store_id] = vector_store
                db_vector_store_ids.add(vector_store_id)
        
        # Process in-memory vector stores
        if litellm.vector_store_registry is not None:
            in_memory_vector_stores = copy.deepcopy(
                litellm.vector_store_registry.vector_stores
            )
            
            vector_stores_to_delete_from_memory: List[str] = []
            
            for vector_store in in_memory_vector_stores:
                vector_store_id = vector_store.get("vector_store_id", None)
                if not vector_store_id:
                    continue
                
                # If vector store is in memory but NOT in database, it was deleted
                if vector_store_id not in db_vector_store_ids:
                    verbose_proxy_logger.info(
                        f"Vector store {vector_store_id} exists in memory but not in database - marking for deletion from cache"
                    )
                    vector_stores_to_delete_from_memory.append(vector_store_id)
                # If not in our map yet, add it (only in-memory, not in DB)
                elif vector_store_id not in vector_store_map:
                    vector_store_map[vector_store_id] = vector_store
            
            # Synchronize in-memory registry with database
            # 1. Remove deleted vector stores from memory
            for vs_id in vector_stores_to_delete_from_memory:
                litellm.vector_store_registry.delete_vector_store_from_registry(
                    vector_store_id=vs_id
                )
                verbose_proxy_logger.debug(
                    f"Removed deleted vector store {vs_id} from in-memory registry"
                )
            
            # 2. Update in-memory registry with database versions (for updates)
            for vector_store in vector_stores_from_db:
                vector_store_id = vector_store.get("vector_store_id", None)
                if vector_store_id:
                    litellm.vector_store_registry.update_vector_store_in_registry(
                        vector_store_id=vector_store_id,
                        updated_data=vector_store
                    )

        # Filter vector stores based on team access
        accessible_vector_stores = [
            vs for vs in vector_store_map.values()
            if _check_vector_store_access(vs, user_api_key_dict)
        ]
        
        total_count = len(accessible_vector_stores)
        total_pages = (total_count + page_size - 1) // page_size

        # Format response using LiteLLM_ManagedVectorStoreListResponse
        response = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=accessible_vector_stores,
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
    Delete a vector store from both database and in-memory registry.

    Parameters:
    - vector_store_id: str - ID of the vector store to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if vector store exists in database or in-memory registry
        db_vector_store_exists = False
        memory_vector_store_exists = False
        vector_store_to_check = None
        
        existing_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                where={"vector_store_id": data.vector_store_id}
            )
        )
        if existing_vector_store is not None:
            db_vector_store_exists = True
            vector_store_to_check = LiteLLM_ManagedVectorStore(
                **existing_vector_store.model_dump()
            )
        
        # Check in-memory registry
        if litellm.vector_store_registry is not None:
            memory_vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if memory_vector_store is not None:
                memory_vector_store_exists = True
                if vector_store_to_check is None:
                    vector_store_to_check = memory_vector_store
        
        # If not found in either location, raise 404
        if not db_vector_store_exists and not memory_vector_store_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )
        
        # Check access control
        if vector_store_to_check and not _check_vector_store_access(
            vector_store_to_check, user_api_key_dict
        ):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to delete this vector store",
            )

        # Delete from database if exists
        if db_vector_store_exists:
            await prisma_client.db.litellm_managedvectorstorestable.delete(
                where={"vector_store_id": data.vector_store_id}
            )

        # Delete from in-memory registry if exists
        if memory_vector_store_exists and litellm.vector_store_registry is not None:
            litellm.vector_store_registry.delete_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )

        return {
            "status": "success",
            "message": f"Vector store {data.vector_store_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting vector store: {str(e)}")
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
                # Check access control
                if not _check_vector_store_access(vector_store, user_api_key_dict):
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: You do not have permission to access this vector store",
                    )
                
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
                    team_id=vector_store.get("team_id") or None,
                    user_id=vector_store.get("user_id") or None,
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
        
        # Check access control for DB vector store
        vector_store_dict = vector_store.model_dump()  # type: ignore[attr-defined]
        vector_store_typed = LiteLLM_ManagedVectorStore(**vector_store_dict)
        if not _check_vector_store_access(vector_store_typed, user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to access this vector store",
            )

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
    """
    Update vector store details in both database and in-memory registry.
    The updated data is immediately synchronized to the in-memory registry.
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.types.router import GenericLiteLLMParams

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        update_data = data.model_dump(exclude_unset=True)
        vector_store_id = update_data.pop("vector_store_id")
        
        # Handle metadata serialization
        if update_data.get("vector_store_metadata") is not None:
            update_data["vector_store_metadata"] = safe_dumps(
                update_data["vector_store_metadata"]
            )
        
        # Handle litellm_params if provided
        if "litellm_params" in update_data:
            _input_litellm_params: dict = update_data.get("litellm_params", {}) or {}
            
            # Auto-resolve embedding config if embedding model is provided but config is not
            embedding_model = _input_litellm_params.get("litellm_embedding_model")
            if embedding_model and not _input_litellm_params.get("litellm_embedding_config"):
                resolved_config = await _resolve_embedding_config(
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
            update_data["litellm_params"] = safe_dumps(litellm_params_dict)

        # Update in database
        updated = await prisma_client.db.litellm_managedvectorstorestable.update(
            where={"vector_store_id": vector_store_id},
            data=update_data,
        )

        updated_vs = LiteLLM_ManagedVectorStore(**updated.model_dump())

        # Immediately update in-memory registry to keep it in sync
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=updated_vs,
            )
            verbose_proxy_logger.debug(
                f"Updated vector store {vector_store_id} in both database and in-memory registry"
            )

        return {
            "status": "success",
            "message": f"Vector store {vector_store_id} updated successfully",
            "vector_store": updated_vs
        }
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
