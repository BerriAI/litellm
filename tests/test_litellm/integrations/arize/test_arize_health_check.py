"""
Test Arize health check functionality and proxy integration.
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import pytest

import litellm
from litellm.integrations.arize.arize import ArizeLogger
from litellm.types.utils import StandardCallbackDynamicParams


class TestArizeHealthCheck:
    """Test Arize health check functionality."""

    @pytest.mark.asyncio
    async def test_arize_health_check_with_credentials(self):
        """Test Arize health check returns healthy when credentials are available."""
        
        with patch.dict(os.environ, {
            "ARIZE_SPACE_KEY": "test-space-key",
            "ARIZE_API_KEY": "test-api-key",
            "ARIZE_ENDPOINT": "https://otlp.arize.com/v1"
        }):
            arize_logger = ArizeLogger()
            response = await arize_logger.async_health_check()
            
            assert response["status"] == "healthy"
            assert "configured properly" in response["message"]

    @pytest.mark.asyncio
    async def test_arize_health_check_missing_space_key(self):
        """Test Arize health check returns unhealthy when space key is missing."""
        
        with patch.dict(os.environ, {
            "ARIZE_API_KEY": "test-api-key"
        }, clear=True):
            arize_logger = ArizeLogger()
            response = await arize_logger.async_health_check()
            
            assert response["status"] == "unhealthy"
            assert "ARIZE_SPACE_KEY" in response["error_message"]

    @pytest.mark.asyncio
    async def test_arize_health_check_missing_api_key(self):
        """Test Arize health check returns unhealthy when API key is missing."""
        
        with patch.dict(os.environ, {
            "ARIZE_SPACE_KEY": "test-space-key"
        }, clear=True):
            arize_logger = ArizeLogger()
            response = await arize_logger.async_health_check()
            
            assert response["status"] == "unhealthy"
            assert "ARIZE_API_KEY" in response["error_message"]

    @pytest.mark.asyncio
    async def test_arize_health_check_missing_both_keys(self):
        """Test Arize health check when both keys are missing."""
        
        with patch.dict(os.environ, {}, clear=True):
            arize_logger = ArizeLogger()
            response = await arize_logger.async_health_check()
            
            assert response["status"] == "unhealthy"
            assert "ARIZE_SPACE_KEY" in response["error_message"]


class TestArizeIntegrationWithProxy:
    """Test Arize integration with LiteLLM completion requests."""

    @pytest.mark.asyncio
    async def test_arize_logging_with_completion(self):
        """Test that Arize logging works with actual completion requests."""
        
        with patch.dict(os.environ, {
            "ARIZE_SPACE_KEY": "test-space-key", 
            "ARIZE_API_KEY": "test-api-key",
            "ARIZE_ENDPOINT": "https://otlp.arize.com/v1"
        }):
            # Create ArizeLogger instance
            arize_logger = ArizeLogger()
            
            # Store original callbacks
            original_callbacks = litellm.success_callback.copy() if litellm.success_callback else []
            
            try:
                # Add ArizeLogger to callbacks
                litellm.success_callback = [arize_logger]
                
                # Make completion request
                response = await litellm.acompletion(
                    model="openai/litellm-mock-response-model",
                    messages=[{"role": "user", "content": "Test message for Arize health check"}],
                    mock_response="This is a test response that validates Arize integration.",
                    user="test-arize-health"
                )
                
                # Verify response is valid
                assert response is not None
                print(f"Response type: {type(response)}")
                print("✅ Arize completion request completed successfully")
                
                # Give time for async logging
                await asyncio.sleep(0.1)
                
                print("✅ Arize completion logging test successful")
                
            finally:
                # Restore original callbacks
                litellm.success_callback = original_callbacks

    def test_arize_get_config(self):
        """Test ArizeLogger.get_arize_config() method."""
        
        with patch.dict(os.environ, {
            "ARIZE_SPACE_KEY": "test-space-123",
            "ARIZE_API_KEY": "test-api-456", 
            "ARIZE_ENDPOINT": "https://custom.arize.com/v1",
            "ARIZE_PROJECT_NAME": "custom-project",
        }):
            config = ArizeLogger.get_arize_config()
            
            assert config.space_key == "test-space-123"
            assert config.api_key == "test-api-456"
            assert config.endpoint == "https://custom.arize.com/v1"
            assert config.protocol == "otlp_grpc"
            assert config.project_name == "custom-project"

    def test_arize_get_config_defaults(self):
        """Test ArizeLogger.get_arize_config() with default endpoint."""
        
        with patch.dict(os.environ, {
            "ARIZE_SPACE_KEY": "test-space-default",
            "ARIZE_API_KEY": "test-api-default",
            "ARIZE_PROJECT_NAME": "default-project",
        }, clear=True):
            config = ArizeLogger.get_arize_config()
            
            assert config.space_key == "test-space-default"
            assert config.api_key == "test-api-default"
            assert config.endpoint == "https://otlp.arize.com/v1"  # Default endpoint
            assert config.protocol == "otlp_grpc"  # Default protocol
            assert config.project_name == "default-project"

    def test_arize_construct_dynamic_headers(self):
        """Test dynamic OTEL headers construction for team/key logging."""
        
        arize_logger = ArizeLogger()
        
        dynamic_params = StandardCallbackDynamicParams(
            arize_space_key="dynamic-space-123",
            arize_api_key="dynamic-api-456"
        )
        
        headers = arize_logger.construct_dynamic_otel_headers(dynamic_params)
        
        assert headers is not None
        assert headers["arize-space-id"] == "dynamic-space-123"
        assert headers["api_key"] == "dynamic-api-456"

    def test_arize_construct_dynamic_headers_space_id_fallback(self):
        """Test dynamic headers with arize_space_id parameter (fallback)."""
        
        arize_logger = ArizeLogger()
        
        dynamic_params = StandardCallbackDynamicParams(
            arize_space_id="fallback-space-789",  # Using space_id instead of space_key
            arize_api_key="fallback-api-999"
        )
        
        headers = arize_logger.construct_dynamic_otel_headers(dynamic_params)
        
        assert headers is not None
        assert headers["arize-space-id"] == "fallback-space-789"
        assert headers["api_key"] == "fallback-api-999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
