from base_vector_store_test import BaseVectorStoreTest

class TestOpenAIVectorStore(BaseVectorStoreTest):
    def get_base_request_args(self) -> dict:
        """
        This is a real vector store on OpenAI
        """
        return {
            "vector_store_id": "vs_685b14b1a1b88191bc27e04f1917fddd",
            "custom_llm_provider": "openai",
        }
    

    def get_base_create_vector_store_args(self) -> dict:
        """
        This is a real vector store on OpenAI
        """
        return {
            "custom_llm_provider": "openai",
        }