"""
Test RAGFlow Vector Store helper functions and transformation.
"""
import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

sys.path.insert(0, os.path.abspath("../.."))
import litellm

from tests.vector_store_tests.base_vector_store_test import BaseVectorStoreTest
from litellm.llms.ragflow.vector_stores.transformation import RAGFlowVectorStoreConfig
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.vector_stores import VectorStoreCreateOptionalRequestParams


class TestRAGFlowVectorStore(BaseVectorStoreTest):
    """
    Test the RAGFlow vector store transformation functionality.
    """
    
    def get_base_create_vector_store_args(self) -> dict:
        """Must return the base create vector store args"""
        return {
            "custom_llm_provider": "ragflow",
            "api_key": os.getenv("RAGFLOW_API_KEY", "test-api-key"),
            "api_base": os.getenv("RAGFLOW_API_BASE", "http://localhost:9380")
        }
    
    def get_base_request_args(self):
        # RAGFlow doesn't support search, so we'll skip search tests
        return {
            "vector_store_id": "test-dataset-id",
            "custom_llm_provider": "ragflow",
            "query": "test query"
        }

    def test_get_auth_credentials(self):
        """Test that auth credentials are correctly extracted."""
        config = RAGFlowVectorStoreConfig()
        
        # Test with api_key in params
        litellm_params = {"api_key": "test-api-key-123"}
        credentials = config.get_auth_credentials(litellm_params)
        assert "headers" in credentials
        assert credentials["headers"]["Authorization"] == "Bearer test-api-key-123"
        
        # Test with missing api_key (should raise ValueError)
        with pytest.raises(ValueError, match="api_key is required"):
            config.get_auth_credentials({})

    def test_get_complete_url(self):
        """Test that complete URL is correctly constructed."""
        config = RAGFlowVectorStoreConfig()
        
        # Test with api_base in params
        litellm_params = {"api_base": "http://custom-host:9999"}
        url = config.get_complete_url(api_base=None, litellm_params=litellm_params)
        assert url == "http://custom-host:9999/api/v1/datasets"
        
        # Test with api_base parameter
        url = config.get_complete_url(api_base="http://test-host:8888", litellm_params={})
        assert url == "http://test-host:8888/api/v1/datasets"
        
        # Test with default (no api_base provided)
        with patch.dict(os.environ, {}, clear=True):
            url = config.get_complete_url(api_base=None, litellm_params={})
            assert url == "http://localhost:9380/api/v1/datasets"
        
        # Test with trailing slash removal
        url = config.get_complete_url(api_base="http://test-host:8888/", litellm_params={})
        assert url == "http://test-host:8888/api/v1/datasets"

    def test_validate_environment(self):
        """Test environment validation and header setting."""
        config = RAGFlowVectorStoreConfig()
        from litellm.types.router import GenericLiteLLMParams
        
        # Test with api_key in litellm_params
        litellm_params = GenericLiteLLMParams(api_key="test-key")
        headers = config.validate_environment({}, litellm_params)
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        
        # Test with missing api_key
        with pytest.raises(ValueError, match="RAGFLOW_API_KEY"):
            config.validate_environment({}, GenericLiteLLMParams())

    def test_get_vector_store_endpoints_by_type(self):
        """Test that endpoints are correctly configured (empty for management only)."""
        config = RAGFlowVectorStoreConfig()
        endpoints = config.get_vector_store_endpoints_by_type()
        assert endpoints["read"] == []
        assert endpoints["write"] == []

    def test_transform_create_vector_store_request_basic(self):
        """Test basic dataset creation request transformation."""
        config = RAGFlowVectorStoreConfig()
        
        params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-dataset"
        }
        
        url, body = config.transform_create_vector_store_request(
            params, "http://localhost:9380/api/v1/datasets"
        )
        
        assert url == "http://localhost:9380/api/v1/datasets"
        assert body["name"] == "test-dataset"
        assert body["chunk_method"] == "naive"  # Default chunk method

    def test_transform_create_vector_store_request_with_metadata(self):
        """Test dataset creation with RAGFlow-specific metadata."""
        config = RAGFlowVectorStoreConfig()
        
        params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-dataset-advanced",
            "metadata": {
                "description": "Test dataset",
                "embedding_model": "BAAI/bge-large-zh-v1.5@BAAI",
                "permission": "me",
                "chunk_method": "naive",
                "parser_config": {
                    "chunk_token_num": 512,
                    "delimiter": "\n"
                }
            }
        }
        
        url, body = config.transform_create_vector_store_request(
            params, "http://localhost:9380/api/v1/datasets"
        )
        
        assert body["name"] == "test-dataset-advanced"
        assert body["description"] == "Test dataset"
        assert body["embedding_model"] == "BAAI/bge-large-zh-v1.5@BAAI"
        assert body["permission"] == "me"
        assert body["chunk_method"] == "naive"
        assert "parser_config" in body
        assert body["parser_config"]["chunk_token_num"] == 512

    def test_transform_create_vector_store_request_missing_name(self):
        """Test that missing name raises ValueError."""
        config = RAGFlowVectorStoreConfig()
        
        params: VectorStoreCreateOptionalRequestParams = {}
        
        with pytest.raises(ValueError, match="name is required"):
            config.transform_create_vector_store_request(
                params, "http://localhost:9380/api/v1/datasets"
            )

    def test_transform_create_vector_store_request_mutually_exclusive(self):
        """Test that chunk_method and pipeline_id are mutually exclusive."""
        config = RAGFlowVectorStoreConfig()
        
        params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-dataset",
            "metadata": {
                "chunk_method": "naive",
                "pipeline_id": "d0bebe30ae2211f0970942010a8e0005"
            }
        }
        
        with pytest.raises(ValueError, match="mutually exclusive"):
            config.transform_create_vector_store_request(
                params, "http://localhost:9380/api/v1/datasets"
            )

    def test_transform_create_vector_store_request_with_pipeline(self):
        """Test dataset creation with ingestion pipeline."""
        config = RAGFlowVectorStoreConfig()
        
        params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-pipeline-dataset",
            "metadata": {
                "parse_type": 2,
                "pipeline_id": "d0bebe30ae2211f0970942010a8e0005"
            }
        }
        
        url, body = config.transform_create_vector_store_request(
            params, "http://localhost:9380/api/v1/datasets"
        )
        
        assert body["name"] == "test-pipeline-dataset"
        assert body["parse_type"] == 2
        assert body["pipeline_id"] == "d0bebe30ae2211f0970942010a8e0005"
        assert "chunk_method" not in body

    def test_transform_create_vector_store_response_success(self):
        """Test successful response transformation."""
        config = RAGFlowVectorStoreConfig()
        
        # Mock RAGFlow response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "id": "3b4de7d4241d11f0a6a79f24fc270c7f",
                "name": "test-dataset",
                "create_time": 1745836841611,
                "chunk_method": "naive",
                "embedding_model": "BAAI/bge-large-zh-v1.5@BAAI"
            }
        }
        
        response = config.transform_create_vector_store_response(mock_response)
        
        assert response["id"] == "3b4de7d4241d11f0a6a79f24fc270c7f"
        assert response["name"] == "test-dataset"
        assert response["object"] == "vector_store"
        assert response["status"] == "completed"
        assert response["created_at"] == 1745836841  # Converted from milliseconds
        assert response["bytes"] == 0
        assert "file_counts" in response

    def test_transform_create_vector_store_response_error(self):
        """Test error response transformation."""
        config = RAGFlowVectorStoreConfig()
        
        # Mock RAGFlow error response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_response.json.return_value = {
            "code": 101,
            "message": "Dataset name 'test-dataset' already exists"
        }
        
        with pytest.raises(Exception):  # Should raise BaseLLMException
            config.transform_create_vector_store_response(mock_response)

    def test_transform_create_vector_store_response_missing_id(self):
        """Test response with missing dataset ID."""
        config = RAGFlowVectorStoreConfig()
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "name": "test-dataset"
                # Missing "id"
            }
        }
        
        with pytest.raises(ValueError, match="missing dataset id"):
            config.transform_create_vector_store_response(mock_response)

    def test_transform_search_vector_store_request_not_implemented(self):
        """Test that search operations raise NotImplementedError."""
        config = RAGFlowVectorStoreConfig()
        logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        
        with pytest.raises(NotImplementedError, match="management only"):
            config.transform_search_vector_store_request(
                vector_store_id="test-id",
                query="test query",
                vector_store_search_optional_params={},
                api_base="http://localhost:9380",
                litellm_logging_obj=logging_obj,
                litellm_params={}
            )

    def test_transform_search_vector_store_response_not_implemented(self):
        """Test that search response transformation raises NotImplementedError."""
        config = RAGFlowVectorStoreConfig()
        logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_response = Mock(spec=httpx.Response)
        
        with pytest.raises(NotImplementedError, match="management only"):
            config.transform_search_vector_store_response(mock_response, logging_obj)

    def _validate_vector_store_create_response(self, response):
        """Override to handle RAGFlow-specific response format."""
        # RAGFlow IDs are hex strings (not OpenAI-style vs_* format)
        # So we override the base validation to not check for vs_ prefix
        assert isinstance(response, dict), f"Response should be a dict, got {type(response)}"
        assert "id" in response, "Missing required field 'id' in create response"
        assert "object" in response, "Missing required field 'object' in create response"
        assert "created_at" in response, "Missing required field 'created_at' in create response"
        
        assert response["object"] == "vector_store", \
            f"Expected object to be 'vector_store', got '{response['object']}'"
        
        assert isinstance(response["id"], str), \
            f"id should be a string, got {type(response['id'])}"
        assert len(response["id"]) > 0, "id should not be empty"
        # RAGFlow IDs are hex strings, not OpenAI-style vs_* format
        
        assert isinstance(response["created_at"], int), \
            f"created_at should be an integer, got {type(response['created_at'])}"
        assert response["created_at"] > 0, "created_at should be a positive timestamp"
        
        print(f"✅ RAGFlow create response validation passed: Dataset '{response['id']}' created successfully")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_create_vector_store(self, sync_mode):
        """Override to handle RAGFlow-specific connection errors."""
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_request_args = self.get_base_create_vector_store_args()
        
        # Skip if no API key is set
        if not os.getenv("RAGFLOW_API_KEY") and not base_request_args.get("api_key"):
            pytest.skip("RAGFLOW_API_KEY not set, skipping integration test")
        
        # Extract custom_llm_provider from base args if present
        create_args = base_request_args
        try: 
            if sync_mode:
                response = litellm.vector_stores.create(
                    name=f"test-ragflow-{int(__import__('time').time())}",
                    **create_args
                )
            else:
                response = await litellm.vector_stores.acreate(
                    name=f"test-ragflow-{int(__import__('time').time())}",
                    **create_args
                )
        except litellm.InternalServerError: 
            pytest.skip("Skipping test due to litellm.InternalServerError")
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Check if it's a connection error
            if (isinstance(e, (ConnectionError, OSError)) or 
                "connection" in error_str or 
                "connect" in error_str or
                "APIConnectionError" in error_type):
                pytest.skip(f"Skipping test due to connection error (RAGFlow instance may not be running): {e}")
            
            # If this is an authentication or permission error, skip the test
            if "authentication" in error_str or "permission" in error_str or "unauthorized" in error_str:
                pytest.skip(f"Skipping test due to authentication/permission error: {e}")
            
            # Re-raise if it's not a handled error
            raise
        
        print("litellm create response=", json.dumps(response, indent=4, default=str))
        
        # Validate response structure
        self._validate_vector_store_create_response(response)

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_search_vector_store(self, sync_mode):
        """Override search test - RAGFlow doesn't support search."""
        pytest.skip("RAGFlow vector stores support dataset management only, not search")

