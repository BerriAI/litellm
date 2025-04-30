# litellm/proxy/vector_stores/vector_store_registry.py
import json
import time
from typing import Any, Dict, List

from litellm._logging import verbose_logger
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    LiteLLM_VectorStoreConfig,
    SupportedVectorStoreIntegrations,
)

from .vector_store_initializers import initialize_bedrock_vector_store

# mapping of the vector store custom_llm_provider to the vector store initializer
VECTOR_STORE_REGISTRY = {
    SupportedVectorStoreIntegrations.BEDROCK.value: initialize_bedrock_vector_store,
}


class VectorStoreManager:
    def __init__(self):
        self.vector_stores: List[LiteLLM_ManagedVectorStore] = []

    def load_vector_stores_from_config(self, vector_stores_config: List[Dict]):
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
                created_at=vector_store_litellm_params.get("created_at")
                or int(time.time()),
                updated_at=vector_store_litellm_params.get("updated_at")
                or int(time.time()),
                # store any extra / additional litellm params
                litellm_params=vector_store_litellm_params,
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


global_vector_store_manager = VectorStoreManager()
