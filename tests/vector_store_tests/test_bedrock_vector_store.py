"""
Test Bedrock Vector Store helper functions and transformation.
"""
import pytest
from unittest.mock import Mock
import httpx

from tests.vector_store_tests.base_vector_store_test import BaseVectorStoreTest
from litellm.llms.bedrock.vector_stores.transformation import BedrockVectorStoreConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class TestBedrockVectorStore(BaseVectorStoreTest):
    """
    Test the Bedrock vector store transformation functionality.
    """
    
    def get_base_create_vector_store_args(self) -> dict:
        """Must return the base create vector store args"""
        return {}
    
    def get_base_request_args(self):
        return {
            "vector_store_id": "T37J8R4WTM",
            "custom_llm_provider": "bedrock",
            "query": "what happens after we add a model"
        }

    def test_get_file_id_from_metadata(self):
        """Test that file_id is correctly extracted from metadata."""
        config = BedrockVectorStoreConfig()
        
        # Test with source URI
        metadata_with_uri = {
            "x-amz-bedrock-kb-source-uri": "https://www.litellm.ai",
            "x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co"
        }
        file_id = config._get_file_id_from_metadata(metadata_with_uri)
        assert file_id == "https://www.litellm.ai"
        
        # Test without source URI but with chunk ID
        metadata_without_uri = {
            "x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co"
        }
        file_id = config._get_file_id_from_metadata(metadata_without_uri)
        assert file_id == "bedrock-kb-1%3A0%3AjNYPg5YByRuP5PdK96co"
        
        # Test with empty metadata
        file_id = config._get_file_id_from_metadata({})
        assert file_id == "bedrock-kb-unknown"

    def test_get_filename_from_metadata(self):
        """Test that filename is correctly extracted from metadata."""
        config = BedrockVectorStoreConfig()
        
        # Test with source URI containing path
        metadata_with_path = {
            "x-amz-bedrock-kb-source-uri": "https://docs.litellm.ai/tutorial/setup.html"
        }
        filename = config._get_filename_from_metadata(metadata_with_path)
        assert filename == "setup.html"
        
        # Test with source URI without path (domain only)
        metadata_domain_only = {
            "x-amz-bedrock-kb-source-uri": "https://www.litellm.ai"
        }
        filename = config._get_filename_from_metadata(metadata_domain_only)
        assert filename == "www.litellm.ai"
        
        # Test without source URI but with data source ID
        metadata_without_uri = {
            "x-amz-bedrock-kb-data-source-id": "CCEJIRXXFI"
        }
        filename = config._get_filename_from_metadata(metadata_without_uri)
        assert filename == "bedrock-kb-document-CCEJIRXXFI"
        
        # Test with empty metadata
        filename = config._get_filename_from_metadata({})
        assert filename == "bedrock-kb-document-unknown"

    def test_get_attributes_from_metadata(self):
        """Test that attributes are correctly extracted from metadata."""
        config = BedrockVectorStoreConfig()
        
        # Test with full metadata
        metadata = {
            "x-amz-bedrock-kb-source-uri": "https://www.litellm.ai",
            "x-amz-bedrock-kb-chunk-id": "1%3A0%3AjNYPg5YByRuP5PdK96co",
            "x-amz-bedrock-kb-data-source-id": "CCEJIRXXFI"
        }
        attributes = config._get_attributes_from_metadata(metadata)
        assert attributes == metadata
        assert attributes is not metadata  # Should be a copy
        
        # Test with empty metadata
        attributes = config._get_attributes_from_metadata({})
        assert attributes == {}
        
        # Test with None
        attributes = config._get_attributes_from_metadata(None)
        assert attributes == {} 


@pytest.mark.asyncio
async def test_bedrock_search_with_router():
    from litellm.router import Router
    # init router
    _router = Router(model_list=[])
    search_response = await _router.avector_store_search(
        query="what happens after we add a model",
        vector_store_id="T37J8R4WTM",
        custom_llm_provider="bedrock",
    )
    print(search_response)