import pytest
import litellm
from litellm.llms.pg_vector.vector_stores.transformation import PGVectorStoreConfig
from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams

class TestPGVectorStoreConfig:
    def test_pg_vector_inheritance(self):
        """
        Verify that PGVectorStoreConfig inherits from OpenAIVectorStoreConfig.
        """
        config = PGVectorStoreConfig()
        assert isinstance(config, OpenAIVectorStoreConfig)

    def test_pg_vector_url_construction(self):
        """
        Verify that PGVectorStoreConfig correctly constructs the URL.
        """
        config = PGVectorStoreConfig()
        api_base = "http://localhost:8080"
        url = config.get_complete_url(api_base=api_base, litellm_params={})
        assert url == "http://localhost:8080/v1/vector_stores"

    def test_pg_vector_validate_environment(self):
        """
        Verify that PGVectorStoreConfig correctly sets the Authorization header.
        """
        config = PGVectorStoreConfig()
        headers = {}
        litellm_params = GenericLiteLLMParams(api_key="test-key")
        
        result_headers = config.validate_environment(headers=headers, litellm_params=litellm_params)
        
        assert result_headers["Authorization"] == "Bearer test-key"
        assert result_headers["Content-Type"] == "application/json"

    def test_pg_vector_in_openai_compatible_providers(self):
        """
        Test that pg_vector is in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS.
        This is required for file uploads and vector store files functionality.
        """
        from litellm.types.utils import OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS
        
        assert litellm.LlmProviders.PG_VECTOR.value in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS

    def test_pg_vector_store_files_config(self):
        """
        Test that get_provider_vector_store_files_config returns OpenAIVectorStoreFilesConfig for pg_vector.
        """
        from litellm.utils import ProviderConfigManager
        from litellm.llms.openai.vector_store_files.transformation import OpenAIVectorStoreFilesConfig
        
        config = ProviderConfigManager.get_provider_vector_store_files_config(litellm.LlmProviders.PG_VECTOR)
        assert isinstance(config, OpenAIVectorStoreFilesConfig)
