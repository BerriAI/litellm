from base_vector_store_test import BaseVectorStoreTest

class TestAzureOpenAIVectorStore(BaseVectorStoreTest):
    def get_base_request_args(self) -> dict:
        """
        This is a real vector store on Azure
        """
        return {
            "vector_store_id": "vs_pRkcTzdBLLtzfoGtOxAgYFGl",
            "custom_llm_provider": "azure",
        }