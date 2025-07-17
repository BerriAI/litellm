"""
Unit tests for PG Vector Store transformation.

This test file mirrors litellm/llms/pg_vector/vector_stores/transformation.py
and contains mocked tests for the PGVectorStoreConfig class.
"""

from unittest.mock import Mock

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
        
        assert result_url == "https://my-pg-vector-service.example.com/vector_stores"

    def test_get_complete_url_removes_trailing_slashes(self):
        """
        Test that get_complete_url handles trailing slashes correctly.
        
        This test validates that URLs are normalized properly.
        """
        config = PGVectorStoreConfig()
        api_base = "https://my-pg-vector-service.example.com/"
        litellm_params = {}
        
        result_url = config.get_complete_url(api_base, litellm_params)
        
        assert result_url == "https://my-pg-vector-service.example.com/vector_stores"

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
        assert url == "https://example.com/vector_stores"

    @pytest.mark.serial
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
            
            assert url == "https://env-pg-vector.example.com/vector_stores"
        
        # Test that params take precedence over environment variables
        with patch.dict(os.environ, {'PG_VECTOR_API_KEY': 'env_key'}):
            litellm_params = GenericLiteLLMParams(api_key="param_key")
            
            headers = config.validate_environment({}, litellm_params)
            
            # Param key should take precedence over environment variable
            assert headers["Authorization"] == "Bearer param_key"
            assert headers["Content-Type"] == "application/json" 