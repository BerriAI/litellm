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



@pytest.mark.asyncio
async def test_bedrock_search_with_credentials_managed_registry():
    """
    Test that the vector store search uses the credential accessor from the registry
    when AWS environment variables are not set, ensuring credentials are managed properly.
    """
    from unittest.mock import patch, MagicMock
    from litellm.router import Router
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
    from litellm.types.utils import CredentialItem
    from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
    from datetime import datetime, timezone
    import litellm

    # Store original registry and credential list
    original_registry = getattr(litellm, "vector_store_registry", None)
    original_credential_list = getattr(litellm, "credential_list", [])
    
    try:
        # Set up test AWS credentials in the credential system
        test_credentials = CredentialItem(
            credential_name="bedrock-litellm-website-knowledgebase",
            credential_info={
                "provider": "aws",
                "description": "Test AWS credentials for bedrock"
            },
            credential_values={
                "aws_access_key_id": "test_access_key",
                "aws_secret_access_key": "test_secret_key", 
                "aws_region_name": "us-east-1",
            }
        )
        
        # Set up the credential list
        litellm.credential_list = [test_credentials]
        
        # Create vector store with credential reference
        vector_store = LiteLLM_ManagedVectorStore(
            vector_store_id="T37J8R4WTM",
            custom_llm_provider="bedrock",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            litellm_credential_name="bedrock-litellm-website-knowledgebase",
        )
        
        # Set up registry
        registry = VectorStoreRegistry([vector_store])
        litellm.vector_store_registry = registry
        
        # Verify credentials can be retrieved from registry
        retrieved_credentials = registry.get_credentials_for_vector_store("T37J8R4WTM")
        assert retrieved_credentials, "Should retrieve credentials from registry"
        assert retrieved_credentials.get("aws_access_key_id") == "test_access_key"
        assert retrieved_credentials.get("aws_secret_access_key") == "test_secret_key"
        assert retrieved_credentials.get("aws_region_name") == "us-east-1"
        
        # Create router and perform search
        _router = Router(model_list=[])
        
        # Mock the credential injection process to verify it's called
        with patch.object(registry, 'get_credentials_for_vector_store', wraps=registry.get_credentials_for_vector_store) as mock_get_creds:
            # Mock the actual search call to avoid making real API calls
            with patch('litellm.vector_stores.main.base_llm_http_handler.vector_store_search_handler') as mock_handler:
                mock_handler.return_value = {
                    "data": [
                        {
                            "id": "test_result",
                            "text": "Mock search result",
                            "score": 0.9,
                            "metadata": {}
                        }
                    ]
                }
                
                search_response = await _router.avector_store_search(
                    query="what happens after we add a model",
                    vector_store_id="T37J8R4WTM",
                    custom_llm_provider="bedrock",
                )
                
                # Verify the search was called
                mock_handler.assert_called_once()
                call_kwargs = mock_handler.call_args[1]
                
                # Verify that the credential accessor was called with the correct vector store ID
                mock_get_creds.assert_called_with("T37J8R4WTM")
                
                # Verify the credentials were injected into the search call
                litellm_params = call_kwargs.get("litellm_params", {})
                
                # The key test: verify that credentials from the registry were used
                # Since we have a registry with credentials, they should be present in the params
                assert hasattr(litellm_params, 'aws_access_key_id'), "aws_access_key_id should be in litellm_params"
                assert hasattr(litellm_params, 'aws_secret_access_key'), "aws_secret_access_key should be in litellm_params"
                assert hasattr(litellm_params, 'aws_region_name'), "aws_region_name should be in litellm_params"
                
                # Verify we got the expected response
                assert search_response["data"][0]["id"] == "test_result"
                
                print(f"✅ Test passed: Credential accessor was called with vector store ID: T37J8R4WTM")
                print(f"✅ Retrieved credentials: {retrieved_credentials}")
                print(f"✅ Credentials were injected into search call")
                print(f"✅ Search completed successfully using registry credentials")
    
    finally:
        # Restore original state
        litellm.vector_store_registry = original_registry
        litellm.credential_list = original_credential_list