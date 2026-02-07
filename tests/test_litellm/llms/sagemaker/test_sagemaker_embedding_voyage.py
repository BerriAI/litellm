"""
Test cases for SageMaker Voyage embedding model integration

This module tests the factory pattern implementation for Voyage embedding models
deployed on AWS SageMaker endpoints, including parameter handling, request/response
transformation, and model type detection.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm import embedding
from litellm.llms.sagemaker.embedding.transformation import SagemakerEmbeddingConfig
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
from litellm.types.utils import EmbeddingResponse, Usage


class TestSagemakerEmbeddingFactory:
    """Test the factory pattern for SageMaker embedding configurations"""

    def test_get_model_config_voyage_model(self):
        """Test that Voyage models return VoyageEmbeddingConfig"""
        config = SagemakerEmbeddingConfig.get_model_config("voyage-3-5-embedding")
        
        assert isinstance(config, VoyageEmbeddingConfig)
        assert config.get_supported_openai_params("voyage-3-5-embedding") == ["encoding_format", "dimensions"]

    def test_get_model_config_hf_model(self):
        """Test that non-Voyage models return base SagemakerEmbeddingConfig"""
        config = SagemakerEmbeddingConfig.get_model_config("sentence-transformers-model")
        
        assert isinstance(config, SagemakerEmbeddingConfig)
        assert config.get_supported_openai_params("sentence-transformers-model") == []

    def test_get_model_config_case_insensitive(self):
        """Test that model detection is case insensitive"""
        config1 = SagemakerEmbeddingConfig.get_model_config("VOYAGE-3-5-embedding")
        config2 = SagemakerEmbeddingConfig.get_model_config("Voyage-3-5-Embedding")
        config3 = SagemakerEmbeddingConfig.get_model_config("voyage-3-5-embedding")
        
        assert isinstance(config1, VoyageEmbeddingConfig)
        assert isinstance(config2, VoyageEmbeddingConfig)
        assert isinstance(config3, VoyageEmbeddingConfig)


class TestVoyageEmbeddingConfig:
    """Test Voyage-specific embedding configuration"""

    def setup_method(self):
        self.config = VoyageEmbeddingConfig()

    def test_get_supported_openai_params(self):
        """Test supported parameters for Voyage models"""
        params = self.config.get_supported_openai_params("voyage-3-5-embedding")
        assert params == ["encoding_format", "dimensions"]

    def test_map_openai_params_encoding_format(self):
        """Test mapping of encoding_format parameter"""
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "float"},
            optional_params={},
            model="voyage-3-5-embedding",
            drop_params=False
        )
        assert result == {"encoding_format": "float"}

    def test_map_openai_params_dimensions(self):
        """Test mapping of dimensions parameter to output_dimension"""
        result = self.config.map_openai_params(
            non_default_params={"dimensions": 1024},
            optional_params={},
            model="voyage-3-5-embedding",
            drop_params=False
        )
        assert result == {"output_dimension": 1024}

    def test_map_openai_params_unsupported_encoding(self):
        """Test handling of unsupported encoding_format values - VoyageEmbeddingConfig passes through without validation"""
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "invalid"},
            optional_params={},
            model="voyage-3-5-embedding",
            drop_params=False
        )
        assert result == {"encoding_format": "invalid"}

    def test_map_openai_params_drop_unsupported(self):
        """Test that VoyageEmbeddingConfig doesn't drop parameters based on drop_params flag"""
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "invalid", "dimensions": 512},
            optional_params={},
            model="voyage-3-5-embedding",
            drop_params=True
        )
        assert result == {"encoding_format": "invalid", "output_dimension": 512}

    def test_transform_embedding_request(self):
        """Test Voyage request transformation"""
        result = self.config.transform_embedding_request(
            model="voyage-3-5-embedding",
            input=["Hello", "World"],
            optional_params={"encoding_format": "float"},
            headers={}
        )
        expected = {
            "input": ["Hello", "World"],
            "model": "voyage-3-5-embedding",
            "encoding_format": "float"
        }
        assert result == expected

    def test_transform_embedding_response(self):
        """Test Voyage response transformation to OpenAI format"""
        # Mock Voyage response
        voyage_response = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                },
                {
                    "object": "embedding", 
                    "embedding": [0.4, 0.5, 0.6],
                    "index": 1
                }
            ],
            "object": "list",
            "model": "voyage-3-5-embedding",
            "usage": {"total_tokens": 10}
        }
        
        # Create mock httpx Response
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(voyage_response).encode('utf-8'),
            headers={"content-type": "application/json"}
        )
        
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="voyage-3-5-embedding",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={"input": ["Hello", "World"]}
        )
        
        # Verify response structure
        assert result.object == "list"
        assert result.model == "voyage-3-5-embedding"
        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert isinstance(result.usage, Usage)
        assert result.usage.total_tokens == 10


class TestHFSagemakerEmbeddingConfig:
    """Test Hugging Face embedding configuration"""

    def setup_method(self):
        self.config = SagemakerEmbeddingConfig()

    def test_get_supported_openai_params_hf(self):
        """Test that HF models don't support embedding parameters"""
        params = self.config.get_supported_openai_params("sentence-transformers-model")
        assert params == []

    def test_transform_embedding_request_hf(self):
        """Test HF request transformation"""
        result = self.config.transform_embedding_request(
            model="sentence-transformers-model",
            input=["Hello", "World"],
            optional_params={},
            headers={}
        )
        expected = {
            "inputs": ["Hello", "World"]
        }
        assert result == expected

    def test_transform_embedding_response_hf(self):
        """Test HF response transformation to OpenAI format"""
        # Mock HF response
        hf_response = {
            "embedding": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
        }
        
        # Create mock httpx Response
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(hf_response).encode('utf-8'),
            headers={"content-type": "application/json"}
        )
        
        model_response = EmbeddingResponse()
        result = self.config.transform_embedding_response(
            model="sentence-transformers-model",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={"inputs": ["Hello", "World"]}
        )
        
        # Verify response structure
        assert result.object == "list"
        assert result.model == "sentence-transformers-model"
        assert len(result.data) == 2
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert isinstance(result.usage, Usage)


class TestSagemakerEmbeddingIntegration:
    """Integration tests for SageMaker embedding with factory pattern"""

    def test_voyage_embedding_request_format(self):
        """Test that Voyage models use correct request format"""
        with patch('litellm.llms.sagemaker.completion.handler.SagemakerLLM.embedding') as mock_embedding:
            # Mock the actual SageMaker call to avoid AWS credentials
            mock_embedding.return_value = EmbeddingResponse(
                object="list",
                data=[
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
                ],
                model="voyage-3-5-embedding",
                usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
            )
            
            # Test Voyage model
            response = embedding(
                model="sagemaker/voyage-3-5-embedding-endpoint",
                input=["Hello", "World"],
                encoding_format="float",
                dimensions=1024
            )
            
            # Verify the request was made with correct format
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args
            assert call_args[1]["model"] == "voyage-3-5-embedding-endpoint"
            assert call_args[1]["input"] == ["Hello", "World"]
            # Check that the parameters are in the optional_params
            optional_params = call_args[1].get("optional_params", {})
            assert optional_params.get("encoding_format") == "float"
            assert optional_params.get("output_dimension") == 1024  # dimensions is mapped to output_dimension

    def test_hf_embedding_request_format(self):
        """Test that HF models use correct request format"""
        with patch('litellm.llms.sagemaker.completion.handler.SagemakerLLM.embedding') as mock_embedding:
            # Mock the actual SageMaker call to avoid AWS credentials
            mock_embedding.return_value = EmbeddingResponse(
                object="list",
                data=[
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
                ],
                model="sentence-transformers-model",
                usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)
            )
            
            # Test HF model with drop_params=True to ignore unsupported parameters
            response = embedding(
                model="sagemaker/sentence-transformers-endpoint",
                input=["Hello", "World"],
                encoding_format="float",  # Should be ignored
                dimensions=1024,  # Should be ignored
                drop_params=True
            )
            
            # Verify the request was made
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args
            assert call_args[1]["model"] == "sentence-transformers-endpoint"
            assert call_args[1]["input"] == ["Hello", "World"]
            # HF models should ignore these parameters in optional_params
            optional_params = call_args[1].get("optional_params", {})
            assert "encoding_format" not in optional_params or optional_params["encoding_format"] is None
            assert "dimensions" not in optional_params or optional_params["dimensions"] is None

    def test_parameter_validation_voyage(self):
        """Test parameter validation for Voyage models"""
        # Test valid parameters
        config = VoyageEmbeddingConfig()
        result = config.map_openai_params(
            non_default_params={"encoding_format": "float", "dimensions": 512},
            optional_params={},
            model="voyage-3-5-embedding",
            drop_params=False
        )
        assert result == {"encoding_format": "float", "output_dimension": 512}

    def test_parameter_validation_hf(self):
        """Test parameter validation for HF models"""
        # Test that HF models ignore embedding parameters
        config = SagemakerEmbeddingConfig()
        result = config.map_openai_params(
            non_default_params={"encoding_format": "float", "dimensions": 512},
            optional_params={},
            model="sentence-transformers-model",
            drop_params=False
        )
        assert result == {}  # HF models should ignore these parameters


class TestErrorHandling:
    """Test error handling in the embedding integration"""

    def test_voyage_response_missing_data(self):
        """Test handling of Voyage response missing data field"""
        config = VoyageEmbeddingConfig()
        
        # Mock response without data field
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps({"object": "list"}).encode('utf-8'),
            headers={"content-type": "application/json"}
        )
        
        model_response = EmbeddingResponse()
        
        # VoyageEmbeddingConfig doesn't validate for missing data field, it just sets it to None
        result = config.transform_embedding_response(
            model="voyage-3-5-embedding",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=None,
            request_data={"input": ["Hello"]}
        )
        assert result.data is None

    def test_hf_response_missing_embedding(self):
        """Test handling of HF response missing embedding field"""
        config = SagemakerEmbeddingConfig()
        
        # Mock response without embedding field
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps({"object": "list"}).encode('utf-8'),
            headers={"content-type": "application/json"}
        )
        
        model_response = EmbeddingResponse()
        
        with pytest.raises(Exception, match="HF response missing 'embedding' field"):
            config.transform_embedding_response(
                model="sentence-transformers-model",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=None,
                request_data={"inputs": ["Hello"]}
            )


if __name__ == "__main__":
    pytest.main([__file__])
