# litellm/proxy/vector_stores/vector_store_registry.py
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    LiteLLM_VectorStoreConfig,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any


class VectorStoreRegistry:
    def __init__(self, vector_stores: List[LiteLLM_ManagedVectorStore] = []):
        self.vector_stores: List[LiteLLM_ManagedVectorStore] = vector_stores
        self.vector_store_ids_to_vector_store_map: Dict[
            str, LiteLLM_ManagedVectorStore
        ] = {}

    def get_vector_store_ids_to_run(
        self, non_default_params: Dict, tools: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Returns the vector store ids to run

        vector_store_ids can be provided in two ways:
        """
        vector_store_ids: List[str] = []

        # 1. check if vector_store_ids is provided in the non_default_params
        vector_store_ids = non_default_params.get("vector_store_ids", None) or []

        # 2. check if vector_store_ids is provided as a tool in the request
        vector_store_ids = self._get_vector_store_ids_from_tool_calls(
            tools=tools, vector_store_ids=vector_store_ids
        )

        return vector_store_ids

    def pop_vector_store_ids_to_run(
        self, non_default_params: Dict, tools: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Pops the vector store ids from the non_default_params and tools
        """
        vector_store_ids: List[str] = []

        # 1. check if vector_store_ids is provided in the non_default_params
        vector_store_ids = non_default_params.pop("vector_store_ids", None) or []

        # 2. check if vector_store_ids is provided as a tool in the request
        vector_store_ids = self._get_vector_store_ids_from_tool_calls(
            tools=tools, vector_store_ids=vector_store_ids
        )
        return vector_store_ids

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
        for vector_store in self.vector_stores:
            if vector_store.get("vector_store_id") == vector_store_id:
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
