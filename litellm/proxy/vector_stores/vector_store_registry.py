# litellm/proxy/vector_stores/vector_store_registry.py
import json
from typing import Dict, List

from litellm._logging import verbose_logger
from litellm.types.vector_stores import (
    SupportedVectorStoreIntegrations,
    VectorStoreConfig,
    VectorStoreLiteLLMParams,
)

from .vector_store_initializers import initialize_bedrock_vector_store

# mapping of the vector store custom_llm_provider to the vector store initializer
VECTOR_STORE_REGISTRY = {
    SupportedVectorStoreIntegrations.BEDROCK.value: initialize_bedrock_vector_store,
}


class VectorStoreManager:
    def __init__(self):
        self.vector_stores: List[VectorStoreConfig] = []

    def load_vector_stores_from_config(self, vector_stores_config: List[Dict]):
        for vector_store_config in vector_stores_config:
            # cast to VectorStoreConfig
            vector_store_config = VectorStoreConfig(
                vector_store_name=vector_store_config["vector_store_name"],
                litellm_params=VectorStoreLiteLLMParams(
                    **vector_store_config.get("litellm_params", {})
                ),
            )
            self.vector_stores.append(vector_store_config)

        verbose_logger.debug(
            "all loaded vector stores = %s",
            json.dumps(self.vector_stores, indent=4, default=str),
        )

    def list_all_vector_stores(self):
        pass


global_vector_store_manager = VectorStoreManager()
