from base_vector_store_test import BaseVectorStoreTest
import os
import pytest

class TestAzureOpenAIVectorStore(BaseVectorStoreTest):
    def get_base_request_args(self) -> dict:
        """Must return the base request args"""
        return {}

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_search_vector_store(self, sync_mode):
        pass


    def get_base_create_vector_store_args(self) -> dict:
        """
        This is a real vector store on Azure
        """
        return {
            "custom_llm_provider": "azure",
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": "2025-04-01-preview",
        }