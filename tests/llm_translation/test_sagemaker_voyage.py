"""
Test SageMaker Voyage AI embedding implementation.

This module tests the SageMaker Voyage AI embedding functionality,
including AWS authentication, endpoint URL construction, request/response transformation,
and integration with the main litellm embedding function.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import httpx
import litellm
from litellm import embedding
from litellm.llms.voyage.embedding.sagemaker_transformation import (
    SageMakerVoyageEmbeddingConfig,
    SageMakerVoyageError,
)
from litellm.types.utils import EmbeddingResponse
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class TestSageMakerVoyageEmbeddingConfig:
    """Test the SageMaker Voyage embedding configuration class."""
    
    @pytest.fixture
    def config(self):
        """Create a SageMakerVoyageEmbeddingConfig instance for testing."""
        return SageMakerVoyageEmbeddingConfig()

    @pytest.fixture
    def mock_credentials(self):
        """Mock AWS credentials for testing."""
        return {
            "access_key": "mock_access_key",
            "secret_key": "mock_secret_key",
            "token": "mock_token",
            "region": "us-east-1",
        }

    @pytest.fixture
    def sample_optional_params(self):
        """Sample optional parameters for testing."""
        return {
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "mock_key",
            "aws_secret_access_key": "mock_secret",
            "input_type": "query",
            "truncation": True,
        }

    def test_extract_model_name(self, config):
        """Test model name extraction from full model paths."""
        # Test new hierarchical format
        new_model = "sagemaker/voyage/voyage-3"
        expected = "voyage-3"
        assert config._extract_model_name(new_model) == expected
        
        # Test old format (backward compatibility)
        old_model = "sagemaker_voyage/voyage-3"
        assert config._extract_model_name(old_model) == expected
        
        # Test just model name
        model_name = "voyage-3-lite"
        assert config._extract_model_name(model_name) == model_name

    def test_get_endpoint_name_default(self, config, sample_optional_params):
        """Test default endpoint name construction."""
        model = "sagemaker/voyage/voyage-3"
        endpoint_name = config._get_endpoint_name(model, sample_optional_params)
        assert endpoint_name == "voyage-3"

    def test_get_endpoint_name_custom(self, config):
        """Test custom endpoint name from optional parameters."""
        model = "sagemaker/voyage/voyage-3"
        optional_params = {"sagemaker_endpoint_name": "my-custom-endpoint"}
        endpoint_name = config._get_endpoint_name(model, optional_params)
        assert endpoint_name == "my-custom-endpoint"

    @patch.object(SageMakerVoyageEmbeddingConfig, "_get_aws_region_name")
    def test_get_complete_url(self, mock_region, config, sample_optional_params):
        """Test SageMaker runtime URL construction."""
        mock_region.return_value = "us-east-1"
        model = "sagemaker/voyage/voyage-3"
        
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params=sample_optional_params,
            litellm_params={},
        )
        
        expected_url = "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/voyage-3/invocations"
        assert url == expected_url

    @patch.object(SageMakerVoyageEmbeddingConfig, "get_credentials")
    @patch.object(SageMakerVoyageEmbeddingConfig, "_get_aws_region_name")
    def test_validate_environment(
        self, mock_region, mock_credentials_method, config, mock_credentials, sample_optional_params
    ):
        """Test AWS environment validation."""
        mock_credentials_method.return_value = mock_credentials
        mock_region.return_value = "us-east-1"
        
        headers = config.validate_environment(
            headers={},
            model="sagemaker/voyage/voyage-3",
            messages=[],
            optional_params=sample_optional_params,
            litellm_params={},
        )
        
        # Check that AWS credentials are called with correct parameters
        mock_credentials_method.assert_called_once()
        
        # Check that credentials and region are stored in optional_params
        assert "aws_credentials" in sample_optional_params
        assert "aws_region_name" in sample_optional_params
        assert sample_optional_params["aws_credentials"] == mock_credentials
        assert sample_optional_params["aws_region_name"] == "us-east-1"
        
        # Check headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    def test_transform_embedding_request(self, config, sample_optional_params):
        """Test embedding request transformation."""
        model = "sagemaker/voyage/voyage-3"
        input_texts = ["Hello world", "Test text"]
        
        request = config.transform_embedding_request(
            model=model,
            input=input_texts,
            optional_params=sample_optional_params,
            headers={},
        )
        
        # Should use parent Voyage transformation but with extracted model name
        expected_model = "voyage-3"
        assert request["model"] == expected_model
        assert request["input"] == input_texts
        assert request["input_type"] == "query"
        assert request["truncation"] is True

    def test_transform_embedding_response_success(self, config):
        """Test successful embedding response transformation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "voyage-3",
            "usage": {"total_tokens": 10},
        }
        
        model_response = EmbeddingResponse()
        logging_obj = MagicMock()
        
        result = config.transform_embedding_response(
            model="sagemaker/voyage/voyage-3",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
        )
        
        # Check that response is properly transformed
        assert result.model == "voyage-3"
        assert result.object == "list"
        assert len(result.data) == 2
        assert result.usage.total_tokens == 10

    def test_transform_embedding_response_error(self, config):
        """Test embedding response transformation with error."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.status_code = 400
        
        model_response = EmbeddingResponse()
        logging_obj = MagicMock()
        
        with pytest.raises(SageMakerVoyageError) as exc_info:
            config.transform_embedding_response(
                model="sagemaker/voyage/voyage-3",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=logging_obj,
            )
        
        assert exc_info.value.status_code == 400
        assert "Invalid JSON response" in str(exc_info.value.message)

    def test_get_error_class(self, config):
        """Test error class creation."""
        error = config.get_error_class(
            error_message="Test error", status_code=500, headers={}
        )
        
        assert isinstance(error, SageMakerVoyageError)
        assert error.status_code == 500
        assert error.message == "Test error"


class TestSageMakerVoyageIntegration:
    """Test integration with main litellm embedding function."""

    @pytest.mark.parametrize("model", [
        "sagemaker/voyage/voyage-3",
        "sagemaker/voyage/voyage-3-lite", 
        "sagemaker/voyage/voyage-code-3",
    ])
    @patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.embedding")
    def test_embedding_function_routing(self, mock_http_handler, model):
        """Test that embedding function correctly routes to SageMaker Voyage handler."""
        # Mock successful response
        mock_response = EmbeddingResponse(
            object="list",
            data=[{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            model=model.split("/")[-1],
            usage={"total_tokens": 5}
        )
        mock_http_handler.return_value = mock_response
        
        # Test embedding call
        response = embedding(
            model=model,
            input=["Test input"],
            aws_region_name="us-east-1",
            input_type="query",
        )
        
        # Verify handler was called
        mock_http_handler.assert_called_once()
        call_args = mock_http_handler.call_args
        assert call_args.kwargs["custom_llm_provider"] == "sagemaker_voyage"
        assert call_args.kwargs["model"] == model

    @patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.embedding")
    def test_embedding_with_aws_params(self, mock_http_handler):
        """Test embedding call with AWS authentication parameters."""
        mock_response = EmbeddingResponse(
            object="list",
            data=[{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            model="voyage-3",
            usage={"total_tokens": 5}
        )
        mock_http_handler.return_value = mock_response
        
        response = embedding(
            model="sagemaker/voyage/voyage-3",
            input=["Test input"],
            aws_region_name="us-west-2",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            sagemaker_endpoint_name="my-voyage-endpoint",
            input_type="document",
        )
        
        # Verify handler was called with correct parameters
        mock_http_handler.assert_called_once()
        call_args = mock_http_handler.call_args
        assert call_args.kwargs["optional_params"]["aws_region_name"] == "us-west-2"
        assert call_args.kwargs["optional_params"]["aws_access_key_id"] == "test_key"
        assert call_args.kwargs["optional_params"]["sagemaker_endpoint_name"] == "my-voyage-endpoint"
        assert call_args.kwargs["optional_params"]["input_type"] == "document"

    @pytest.mark.asyncio
    @patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.embedding")
    async def test_async_embedding(self, mock_http_handler):
        """Test async embedding functionality."""
        mock_response = EmbeddingResponse(
            object="list",
            data=[{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            model="voyage-3",
            usage={"total_tokens": 5}
        )
        mock_http_handler.return_value = mock_response
        
        response = await litellm.aembedding(
            model="sagemaker/voyage/voyage-3",
            input=["Async test input"],
            aws_region_name="us-east-1",
        )
        
        # Verify async handler was called
        mock_http_handler.assert_called_once()
        call_args = mock_http_handler.call_args
        assert call_args.kwargs["custom_llm_provider"] == "sagemaker_voyage"
        assert call_args.kwargs["aembedding"] is True

    def test_unsupported_model_fallback(self):
        """Test error handling for unsupported models."""
        with pytest.raises(Exception):  # Should raise appropriate error
            embedding(
                model="sagemaker/voyage/unsupported-model",
                input=["Test input"],
            )


@pytest.mark.integration
class TestSageMakerVoyageRealWorld:
    """Integration tests (require actual AWS resources - skip by default)."""
    
    @pytest.mark.skip(reason="Requires actual AWS SageMaker endpoint")
    def test_real_sagemaker_voyage_call(self):
        """
        Test actual SageMaker Voyage call.
        
        Note: This test requires:
        1. Valid AWS credentials
        2. A deployed SageMaker Voyage endpoint
        3. Proper IAM permissions
        """
        response = embedding(
            model="sagemaker/voyage/voyage-3",
            input=["Hello world", "Test embedding"],
            aws_region_name="us-east-1",
            sagemaker_endpoint_name="your-voyage-endpoint-name",
            input_type="query",
        )
        
        assert response.object == "list"
        assert len(response.data) == 2
        assert all(isinstance(item.embedding, list) for item in response.data)
        assert response.usage.total_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])