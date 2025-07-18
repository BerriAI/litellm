"""
Unit tests for PG Vector Store transformation.

This test file mirrors litellm/llms/pg_vector/vector_stores/transformation.py
and contains mocked tests for the PGVectorStoreConfig class.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from litellm.llms.pg_vector.vector_stores.transformation import PGVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams


class TestPGVectorStoreConfig:
    """Test the PG Vector Store transformation configuration."""
    
    def test_validate_environment_with_api_key_in_params(self):
        """
        Test that validate_environment works when api_key is provided in litellm_params.
        
        This test validates that API key from params is correctly set in headers.
        """
        config = PGVectorStoreConfig()
        litellm_params = GenericLiteLLMParams(api_key="test_pg_vector_key_123")
        headers = {}
        
        result_headers = config.validate_environment(headers, litellm_params)
        
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == "Bearer test_pg_vector_key_123"
        assert result_headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """
        Test that validate_environment raises ValueError when no API key is provided.
        
        This test validates that proper error handling occurs for missing credentials.
        """
        config = PGVectorStoreConfig()
        litellm_params = GenericLiteLLMParams()
        headers = {}
        
        with pytest.raises(ValueError) as exc_info:
            config.validate_environment(headers, litellm_params)
        
        assert "PG Vector API key is required" in str(exc_info.value)

    def test_get_complete_url_with_api_base(self):
        """
        Test that get_complete_url correctly formats the URL with api_base.
        
        This test validates URL construction for PG Vector endpoints.
        """
        config = PGVectorStoreConfig()
        api_base = "https://my-pg-vector-service.example.com"
        litellm_params = {}
        
        result_url = config.get_complete_url(api_base, litellm_params)
        
        assert result_url == "https://my-pg-vector-service.example.com/v1/vector_stores"

    def test_get_complete_url_removes_trailing_slashes(self):
        """
        Test that get_complete_url handles trailing slashes correctly.
        
        This test validates that URLs are normalized properly.
        """
        config = PGVectorStoreConfig()
        api_base = "https://my-pg-vector-service.example.com/"
        litellm_params = {}
        
        result_url = config.get_complete_url(api_base, litellm_params)
        
        assert result_url == "https://my-pg-vector-service.example.com/v1/vector_stores"

    def test_get_complete_url_missing_api_base(self):
        """
        Test that get_complete_url raises ValueError when no API base is provided.
        
        This test validates that proper error handling occurs for missing API base.
        """
        config = PGVectorStoreConfig()
        litellm_params = {}
        
        with pytest.raises(ValueError) as exc_info:
            config.get_complete_url(None, litellm_params)
        
        assert "PG Vector API base URL is required" in str(exc_info.value)

    def test_inheritance_from_openai_config(self):
        """
        Test that PGVectorStoreConfig correctly inherits from OpenAIVectorStoreConfig.
        
        This test validates that PG Vector config inherits OpenAI-compatible methods.
        """
        from litellm.llms.openai.vector_stores.transformation import (
            OpenAIVectorStoreConfig,
        )
        
        config = PGVectorStoreConfig()
        
        # Test that it's an instance of the parent class
        assert isinstance(config, OpenAIVectorStoreConfig)
        
        # Test that inherited methods are available
        assert hasattr(config, 'transform_search_vector_store_request')
        assert hasattr(config, 'transform_search_vector_store_response')
        assert hasattr(config, 'transform_create_vector_store_request')
        assert hasattr(config, 'transform_create_vector_store_response')

    def test_openai_compatible_methods_available(self):
        """
        Test that OpenAI-compatible transformation methods are available.
        
        Since PG Vector is OpenAI-compatible, it should inherit all transformation methods.
        """
        config = PGVectorStoreConfig()
        
        # Test that transformation methods are callable
        assert callable(getattr(config, 'transform_search_vector_store_request', None))
        assert callable(getattr(config, 'transform_search_vector_store_response', None))
        assert callable(getattr(config, 'transform_create_vector_store_request', None))
        assert callable(getattr(config, 'transform_create_vector_store_response', None))

    def test_config_methods_with_mock_data(self):
        """
        Test configuration with mock data to ensure basic functionality.
        
        This test validates that the config works with typical parameters.
        """
        config = PGVectorStoreConfig()
        
        # Test with valid parameters
        litellm_params = GenericLiteLLMParams(api_key="test_key")
        headers = config.validate_environment({}, litellm_params)
        url = config.get_complete_url("https://example.com", {})
        
        # Verify results
        assert headers["Authorization"] == "Bearer test_key"
        assert url == "https://example.com/v1/vector_stores"

    def test_environment_variable_support(self):
        """
        Test that environment variables are supported for configuration.
        
        This test validates that the config properly reads from environment variables.
        """
        import os
        from unittest.mock import patch
        
        config = PGVectorStoreConfig()
        
        # Test API key from environment variable
        with patch.dict(os.environ, {'PG_VECTOR_API_KEY': 'env_api_key_123'}):
            litellm_params = GenericLiteLLMParams()  # No API key in params
            
            headers = config.validate_environment({}, litellm_params)
            
            assert headers["Authorization"] == "Bearer env_api_key_123"
            assert headers["Content-Type"] == "application/json"
        
        # Test API base from environment variable
        with patch.dict(os.environ, {'PG_VECTOR_API_BASE': 'https://env-pg-vector.example.com'}):
            url = config.get_complete_url(None, {})
            
            assert url == "https://env-pg-vector.example.com/v1/vector_stores"
        
        # Test that params take precedence over environment variables
        with patch.dict(os.environ, {'PG_VECTOR_API_KEY': 'env_key'}):
            litellm_params = GenericLiteLLMParams(api_key="param_key")
            
            headers = config.validate_environment({}, litellm_params)
            
            # Param key should take precedence over environment variable
            assert headers["Authorization"] == "Bearer param_key"
            assert headers["Content-Type"] == "application/json"

    @patch('litellm.llms.custom_httpx.http_handler.HTTPHandler.post')
    def test_pg_vector_search_request_construction(self, mock_post):
        """
        Test that PG Vector search constructs the correct URL and request body.
        
        This test validates the complete request construction for PG Vector search
        operations, including URL, headers, and request body.
        """
        import litellm

        # Clear any existing vector store registry to prevent interference with test data
        original_registry = getattr(litellm, 'vector_store_registry', None)
        litellm.vector_store_registry = None
        
        try:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "object": "vector_store.search_results.page",
                "search_query": ["what are remote working hours for BerriAI"],
                "data": [
                    {
                        "file_id": "file_123",
                        "filename": "remote_work_policy.txt",
                        "score": 0.95,
                        "attributes": {"department": "HR"},
                        "content": [
                            {
                                "type": "text",
                                "text": "Remote working hours are flexible from 9 AM to 5 PM"
                            }
                        ]
                    }
                ]
            }
            mock_post.return_value = mock_response
            
            # Test parameters - use a different vector store ID than test registry
            api_base = "http://localhost:8001"
            api_key = "sk-1234"
            vector_store_id = "pg-vector-test-store-123"  # Different from test registry IDs
            query = "what are remote working hours for BerriAI"
            
            # Call litellm vector store search
            exception_raised = None
            response = None
            try:
                response = litellm.vector_stores.search(
                    query=query,
                    vector_store_id=vector_store_id,
                    api_base=api_base,
                    api_key=api_key,
                    custom_llm_provider="pg_vector",
                    mock_response=None  # Explicitly disable LiteLLM's automatic mocking
                )
                print(f"✅ Search completed successfully: {response}")
            except Exception as e:
                exception_raised = e
                print(f"❌ Exception raised during search: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            # Print debug information
            print(f"🔍 Mock post called: {mock_post.called}")
            print(f"🔍 Mock post call count: {mock_post.call_count}")
            if mock_post.call_args:
                print(f"🔍 Mock post call args: {mock_post.call_args}")
            
            # For now, let's check if there was an exception that prevented the call
            if exception_raised:
                print(f"🔍 Exception details: {exception_raised}")
                # If there's a specific exception we expect during testing, we might allow it
                # but we should still verify the mock was called before the exception
            
            # Validate that the mock was called correctly
            assert mock_post.called, f"HTTPHandler.post should have been called. Exception: {exception_raised}"
            
            # Get the call arguments
            call_args, call_kwargs = mock_post.call_args
            
            # Validate URL
            expected_url = f"{api_base}/v1/vector_stores/{vector_store_id}/search"
            actual_url = call_kwargs.get('url')
            assert actual_url == expected_url, f"Expected URL {expected_url}, got {actual_url}"
            
            # Validate headers
            headers = call_kwargs.get('headers', {})
            assert headers.get("Authorization") == f"Bearer {api_key}"
            assert headers.get("Content-Type") == "application/json"
            
            # Validate request body - it should be in 'data' parameter as JSON string
            json_data_str = call_kwargs.get('data', '{}')
            import json
            json_data = json.loads(json_data_str) if isinstance(json_data_str, str) else json_data_str
            assert json_data.get("query") == query
            
            print("✅ PG Vector search request validation passed:")
            print(f"   URL: {actual_url}")
            print(f"   Headers: {headers}")
            print(f"   Body: {json_data}") 
        finally:
            # Restore original registry
            litellm.vector_store_registry = original_registry 