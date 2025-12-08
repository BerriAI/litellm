# litellm/proxy/vector_stores/vector_store_registry.py
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, get_args

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import remove_items_at_indices
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreIndex,
    LiteLLM_ManagedVectorStoreListResponse,
    LiteLLM_VectorStoreConfig,
    VectorStoreToolParams,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any

class VectorStoreIndexRegistry:
    def __init__(
        self, vector_store_indexes: List[LiteLLM_ManagedVectorStoreIndex] = []
    ):
        self.vector_store_indexes: List[LiteLLM_ManagedVectorStoreIndex] = (
            vector_store_indexes
        )

    def get_vector_store_indexes(self) -> List[LiteLLM_ManagedVectorStoreIndex]:
        """
        Returns the vector store indexes
        """
        return self.vector_store_indexes

    def get_vector_store_index_by_name(
        self, vector_store_index_name: str
    ) -> Optional[LiteLLM_ManagedVectorStoreIndex]:
        """
        Returns the vector store index by name
        """
        for vector_store_index in self.vector_store_indexes:
            if vector_store_index.index_name == vector_store_index_name:
                return vector_store_index
        return None

    def upsert_vector_store_index(
        self, vector_store_index: LiteLLM_ManagedVectorStoreIndex
    ):
        """
        Adds a vector store index to the registry.

        If it already exists, it will be updated.
        """
        for i, _vector_store_index in enumerate[LiteLLM_ManagedVectorStoreIndex](
            self.vector_store_indexes
        ):
            if _vector_store_index.index_name == vector_store_index.index_name:
                self.vector_store_indexes[i] = vector_store_index
                return
        self.vector_store_indexes.append(vector_store_index)

    def delete_vector_store_index(self, vector_store_index: str):
        """
        Deletes a vector store index from the registry
        """
        self.vector_store_indexes = [
            index for index in self.vector_store_indexes if index != vector_store_index
        ]

    def is_vector_store_index(self, vector_store_index_name: str) -> bool:
        """
        Returns True if the vector store index is in the registry
        """
        for vector_store_index in self.vector_store_indexes:
            if vector_store_index.index_name == vector_store_index_name:
                return True
        return False

    #########################################################
    ########### DB management helpers for vector stores ###########
    #########################################################

    @staticmethod
    async def _get_vector_store_indexes_from_db(
        prisma_client: Optional[PrismaClient],
    ) -> List[LiteLLM_ManagedVectorStoreIndex]:
        """
        Get vector stores from the database
        """
        vector_stores_from_db: List[LiteLLM_ManagedVectorStoreIndex] = []
        if prisma_client is not None:
            _vector_stores_from_db = (
                await prisma_client.db.litellm_managedvectorstoreindextable.find_many(
                    order={"created_at": "desc"},
                )
            )
            for vector_store in _vector_stores_from_db:
                _dict_vector_store = dict(vector_store)
                _litellm_managed_vector_store = LiteLLM_ManagedVectorStoreIndex(
                    **_dict_vector_store
                )
                vector_stores_from_db.append(_litellm_managed_vector_store)
        return vector_stores_from_db


class VectorStoreRegistry:
    def __init__(self, vector_stores: List[LiteLLM_ManagedVectorStore] = []):
        self.vector_stores: List[LiteLLM_ManagedVectorStore] = vector_stores
        self.vector_store_ids_to_vector_store_map: Dict[
            str, LiteLLM_ManagedVectorStore
        ] = {}

    def _extract_tool_params(self, tool: Dict) -> VectorStoreToolParams:
        """
        Extract supported parameters from a tool definition.
        
        Dynamically extracts all parameters defined in VECTOR_STORE_OPENAI_PARAMS.
        """
        # Get the list of supported param names from the Literal type
        supported_params = get_args(VECTOR_STORE_OPENAI_PARAMS)
        
        # Extract only the params that exist in the tool
        kwargs = {param: tool.get(param) for param in supported_params if param in tool}
        
        return VectorStoreToolParams(**kwargs)

    def get_vector_store_ids_to_run(
        self, non_default_params: Dict, tools: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Returns the vector store ids to run

        vector_store_ids can be provided in two ways:
        """
        vector_store_ids: List[str] = []

        # 1. check if vector_store_ids is provided in the non_default_params
        vector_store_ids_param = non_default_params.get("vector_store_ids")
        if isinstance(vector_store_ids_param, list):
            vector_store_ids.extend(vector_store_ids_param)

        # 2. check if vector_store_ids is provided as a tool in the request
        vector_store_ids = self._get_vector_store_ids_from_tool_calls(
            tools=tools, vector_store_ids=vector_store_ids
        )

        return list(dict.fromkeys(vector_store_ids))

    def get_and_pop_recognised_vector_store_tools(
        self, tools: Optional[List[Dict]] = None, vector_store_ids: Optional[List[str]] = None
    ) -> Dict[str, VectorStoreToolParams]:
        """
        Returns and pops recognized vector store tools from the tools list.
        
        Args:
            tools: The tools to extract and remove vector store IDs from
            vector_store_ids: Mutable list to append found vector_store_ids to
        
        Returns:
            Dict mapping vector_store_id to its extracted tool parameters
        """
        params_by_id: Dict[str, VectorStoreToolParams] = {}
        
        if not tools:
            return params_by_id
        
        if vector_store_ids is None:
            vector_store_ids = []
        
        tools_to_remove: List[int] = []
        
        for i, tool in enumerate(tools):
            tool_vector_store_ids = tool.get("vector_store_ids", [])
            if not tool_vector_store_ids:
                continue
            
            # Check if all vector_store_ids are recognized in the registry
            recognised = all(
                any(vs.get("vector_store_id") == vs_id for vs in self.vector_stores)
                for vs_id in tool_vector_store_ids
            )
            
            if recognised:
                tools_to_remove.append(i)
                vector_store_ids.extend(tool_vector_store_ids)
                
                # Extract and store params for each vector store
                tool_params = self._extract_tool_params(tool)
                for vs_id in tool_vector_store_ids:
                    params_by_id[vs_id] = tool_params
        
        # Remove recognized tools from the original list
        remove_items_at_indices(items=tools, indices=tools_to_remove)
        
        return params_by_id

    def get_vector_store_to_run(
        self, non_default_params: Dict, tools: Optional[List[Dict]] = None
    ) -> Optional[LiteLLM_ManagedVectorStore]:
        """
        Returns the vector store to run

         vectore_stores can be run in two ways:
            1. vector_store_ids is provided in the non_default_params
            2. vector_store_ids is provided as a tool in the request


        This will return the first vector store found in the registry.
        """
        vector_store_ids = self.get_vector_store_ids_to_run(
            non_default_params=non_default_params, tools=tools
        )

        # check if the vector store ids are in the registry
        if len(vector_store_ids) <= 0:
            return None

        for vector_store_id in vector_store_ids:
            for vector_store in self.vector_stores:
                if vector_store.get("vector_store_id") == vector_store_id:
                    return vector_store
        return None

    def get_litellm_managed_vector_store_from_registry(
        self, vector_store_id: str
    ) -> Optional[LiteLLM_ManagedVectorStore]:
        """
        Returns the vector store from the registry
        """
        for vector_store in self.vector_stores:
            if vector_store.get("vector_store_id") == vector_store_id:
                return vector_store
        return None

    async def get_litellm_managed_vector_store_from_registry_or_db(
        self, vector_store_id: str, prisma_client: Optional[PrismaClient] = None
    ) -> Optional[LiteLLM_ManagedVectorStore]:
        """
        Returns the vector store from the registry, falling back to database if not found.
        This ensures synchronization across multiple instances.
        """
        # First check in-memory registry
        vector_store = self.get_litellm_managed_vector_store_from_registry(vector_store_id)
        if vector_store is not None:
            return vector_store
        
        # Fall back to database if not found in memory
        if prisma_client is not None:
            try:
                vector_stores_from_db = await self._get_vector_stores_from_db(
                    prisma_client=prisma_client
                )
                for db_vector_store in vector_stores_from_db:
                    if db_vector_store.get("vector_store_id") == vector_store_id:
                        # Add to in-memory registry for future use
                        self.add_vector_store_to_registry(vector_store=db_vector_store)
                        return db_vector_store
            except Exception as e:
                verbose_logger.debug(
                    f"Error fetching vector store from database: {str(e)}"
                )
        
        return None

    def get_litellm_managed_vector_store_from_registry_by_name(
        self, vector_store_name: str
    ) -> Optional[LiteLLM_ManagedVectorStore]:
        """
        Returns the vector store from the registry by name
        """
        for vector_store in self.vector_stores:
            if vector_store.get("vector_store_name") == vector_store_name:
                return vector_store
        return None

    def pop_vector_stores_to_run(
        self, non_default_params: Dict, tools: Optional[List[Dict]] = None
    ) -> List[LiteLLM_ManagedVectorStore]:
        """
        Pops the vector stores to run with their tool parameters merged.
        
        Primary function to use for vector store pre call hook.
        
        Args:
            non_default_params: Parameters dict to pop vector_store_ids from
            tools: Optional list of tools to extract vector store params from
        
        Returns:
            List of vector stores with tool parameters merged into litellm_params
        """
        # Pop vector_store_ids from params
        vector_store_ids: List[str] = non_default_params.pop("vector_store_ids", None) or []
        
        # Extract params from tools and collect IDs
        params_by_id = self.get_and_pop_recognised_vector_store_tools(
            tools=tools,
            vector_store_ids=vector_store_ids
        )
        
        vector_stores_to_run: List[LiteLLM_ManagedVectorStore] = []
        
        for vector_store_id in vector_store_ids:
            for vector_store in self.vector_stores:
                if vector_store.get("vector_store_id") == vector_store_id:
                    # Create a copy to avoid modifying the registry
                    vector_store_copy = vector_store.copy()
                    
                    # Merge tool params if they exist
                    if vector_store_id in params_by_id:
                        existing_params = vector_store_copy.get("litellm_params", {}) or {}
                        tool_params_dict = params_by_id[vector_store_id].to_dict()
                        # Tool params take precedence over existing params
                        tool_params_dict.update(existing_params)
                        vector_store_copy["litellm_params"] = tool_params_dict
                    
                    vector_stores_to_run.append(vector_store_copy)
                    break
        
        return vector_stores_to_run

    async def pop_vector_stores_to_run_with_db_fallback(
        self, 
        non_default_params: Dict, 
        tools: Optional[List[Dict]] = None,
        prisma_client: Optional[PrismaClient] = None
    ) -> List[LiteLLM_ManagedVectorStore]:
        """
        Pops the vector stores to run with their tool parameters merged.
        Falls back to database if vector stores are not found in memory.
        This ensures synchronization across multiple instances.
        
        Primary function to use for vector store pre call hook.
        
        Args:
            non_default_params: Parameters dict to pop vector_store_ids from
            tools: Optional list of tools to extract vector store params from
            prisma_client: Optional database client for fallback lookup
        
        Returns:
            List of vector stores with tool parameters merged into litellm_params
        """
        # Pop vector_store_ids from params
        vector_store_ids: List[str] = non_default_params.pop("vector_store_ids", None) or []
        
        # Extract params from tools and collect IDs
        params_by_id = self.get_and_pop_recognised_vector_store_tools(
            tools=tools,
            vector_store_ids=vector_store_ids
        )
        
        vector_stores_to_run: List[LiteLLM_ManagedVectorStore] = []
        
        for vector_store_id in vector_store_ids:
            vector_store = None
            
            # First check in-memory registry
            for vs in self.vector_stores:
                if vs.get("vector_store_id") == vector_store_id:
                    vector_store = vs
                    break
            
            # Verify vector store still exists in database (if we have DB access)
            # This ensures deleted vector stores are removed from cache
            if vector_store is not None and prisma_client is not None:
                try:
                    # Check if it still exists in database
                    db_vector_store = await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                        where={"vector_store_id": vector_store_id}
                    )
                    if db_vector_store is None:
                        # Vector store was deleted from database, remove from cache
                        verbose_logger.debug(
                            f"Vector store {vector_store_id} found in memory but deleted from database, removing from cache"
                        )
                        self.delete_vector_store_from_registry(vector_store_id=vector_store_id)
                        vector_store = None
                except Exception as e:
                    verbose_logger.debug(
                        f"Error verifying vector store {vector_store_id} in database: {str(e)}"
                    )
            
            # Fall back to database if not found in memory (or was deleted)
            if vector_store is None and prisma_client is not None:
                try:
                    vector_store = await self.get_litellm_managed_vector_store_from_registry_or_db(
                        vector_store_id=vector_store_id,
                        prisma_client=prisma_client
                    )
                except Exception as e:
                    verbose_logger.debug(
                        f"Error fetching vector store {vector_store_id} from database: {str(e)}"
                    )
            
            if vector_store is not None:
                # Create a copy to avoid modifying the registry
                vector_store_copy = vector_store.copy()
                
                # Merge tool params if they exist
                if vector_store_id in params_by_id:
                    existing_params = vector_store_copy.get("litellm_params", {}) or {}
                    tool_params_dict = params_by_id[vector_store_id].to_dict()
                    # Tool params take precedence over existing params
                    tool_params_dict.update(existing_params)
                    vector_store_copy["litellm_params"] = tool_params_dict
                
                vector_stores_to_run.append(vector_store_copy)
        
        return vector_stores_to_run

    def _get_vector_store_ids_from_tool_calls(
        self, tools: Optional[List[Dict]] = None, vector_store_ids: List[str] = []
    ) -> List[str]:
        """
        Returns the vector store ids from the tool calls
        """
        if tools:
            for tool in tools:
                if "vector_store_ids" in tool:
                    vector_store_ids.extend(tool["vector_store_ids"])
        return vector_store_ids

    def load_vector_stores_from_config(self, vector_stores_config: List[Dict]):
        """
        Loads vector stores from the litellm proxy config.yaml
        """
        for vector_store_config in vector_stores_config:
            # cast to VectorStoreConfig
            litellm_vector_store_config = LiteLLM_VectorStoreConfig(
                **vector_store_config
            )
            vector_store_name = litellm_vector_store_config.get("vector_store_name")
            vector_store_litellm_params: Dict[str, Any] = (
                litellm_vector_store_config.get("litellm_params") or {}
            )

            vector_store_id = vector_store_litellm_params.get("vector_store_id")
            if vector_store_id is None:
                raise ValueError(
                    f"vector_store_id is required for initializing vector store, got vector_store_id={vector_store_id}"
                )
            custom_llm_provider = vector_store_litellm_params.get("custom_llm_provider")
            if custom_llm_provider is None:
                raise ValueError(
                    f"custom_llm_provider is required for initializing vector store, got custom_llm_provider={custom_llm_provider}"
                )

            litellm_managed_vector_store = LiteLLM_ManagedVectorStore(
                vector_store_id=vector_store_id,
                custom_llm_provider=custom_llm_provider,
                litellm_params=vector_store_litellm_params,
                vector_store_name=vector_store_name,
                vector_store_description=vector_store_litellm_params.get(
                    "vector_store_description"
                ),
                vector_store_metadata=vector_store_litellm_params.get(
                    "vector_store_metadata"
                ),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            self.vector_stores.append(litellm_managed_vector_store)

        verbose_logger.debug(
            "all loaded vector stores = %s",
            json.dumps(self.vector_stores, indent=4, default=str),
        )

    def list_all_vector_stores(self) -> LiteLLM_ManagedVectorStoreListResponse:
        """
        List all vector stores in the required format

        Returns:
            LiteLLM_ManagedVectorStoreListResponse: A standardized response with vector store data
        """
        # Prepare the response
        response = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=self.vector_stores,
            total_count=len(self.vector_stores),
            current_page=1,
            total_pages=1,
        )

        return response

    def add_vector_store_to_registry(self, vector_store: LiteLLM_ManagedVectorStore):
        """
        Add a vector store to the registry

        Only add the vector store if it is not already in the registry
        """
        vector_store_id = vector_store.get("vector_store_id")
        for _vector_store in self.vector_stores:
            if _vector_store.get("vector_store_id") == vector_store_id:
                return
        self.vector_stores.append(vector_store)

    def delete_vector_store_from_registry(self, vector_store_id: str):
        """
        Delete a vector store from the registry
        """
        self.vector_stores = [
            vector_store
            for vector_store in self.vector_stores
            if vector_store.get("vector_store_id") != vector_store_id
        ]

    def update_vector_store_in_registry(
        self, vector_store_id: str, updated_data: LiteLLM_ManagedVectorStore
    ):
        """Update or add a vector store in the registry"""
        for i, vector_store in enumerate(self.vector_stores):
            if vector_store.get("vector_store_id") == vector_store_id:
                self.vector_stores[i] = updated_data
                return
        self.vector_stores.append(updated_data)

    #########################################################
    ########### DB management helpers for vector stores ###########
    #########################################################

    @staticmethod
    async def _get_vector_stores_from_db(
        prisma_client: Optional[PrismaClient],
    ) -> List[LiteLLM_ManagedVectorStore]:
        """
        Get vector stores from the database
        """
        vector_stores_from_db: List[LiteLLM_ManagedVectorStore] = []
        if prisma_client is not None:
            _vector_stores_from_db = (
                await prisma_client.db.litellm_managedvectorstorestable.find_many(
                    order={"created_at": "desc"},
                )
            )
            for vector_store in _vector_stores_from_db:
                _dict_vector_store = dict(vector_store)
                _litellm_managed_vector_store = LiteLLM_ManagedVectorStore(
                    **_dict_vector_store
                )
                vector_stores_from_db.append(_litellm_managed_vector_store)
        return vector_stores_from_db

    def get_credentials_for_vector_store(self, vector_store_id: str) -> Dict[str, Any]:
        """
        Get the credentials for a vector store

        Returns a dictionary of unpacked credentials for the vector store to use for the request
        """
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

        for vector_store in self.vector_stores:
            if vector_store.get("vector_store_id") == vector_store_id:
                credentials = vector_store.get("litellm_credential_name")
                if credentials:
                    return CredentialAccessor.get_credential_values(credentials)
        return {}
