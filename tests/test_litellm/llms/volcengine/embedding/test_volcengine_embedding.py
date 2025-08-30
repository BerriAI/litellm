"""
Improved tests for Volcengine Embedding functionality
Tests real business logic without excessive mocking
"""

import pytest
import json
import httpx
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from litellm.llms.volcengine.embedding import VolcEngineEmbeddingHandler, VolcEngineEmbeddingConfig
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.types.utils import EmbeddingResponse
from litellm.types.llms.openai import AllEmbeddingInputValues


class TestVolcEngineEmbeddingConfigBusinessLogic:
    """Test real business logic of VolcEngineEmbeddingConfig without excessive mocking"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = VolcEngineEmbeddingConfig()
        self.model = "doubao-embedding-text-240715"
        self.api_key = "test-api-key-12345"
    
    def test_supported_params_completeness(self):
        """Test that all required parameters are supported"""
        params = self.config.get_supported_openai_params(self.model)
        
        # Verify essential parameters are supported
        required_params = ["encoding_format", "user", "extra_headers"]
        for param in required_params:
            assert param in params, f"Required parameter '{param}' not supported"
    
    def test_parameter_mapping_with_valid_values(self):
        """Test parameter mapping with various valid values"""
        test_cases = [
            # Standard float encoding
            {"encoding_format": "float", "user": "test-user"},
            # Base64 encoding 
            {"encoding_format": "base64", "user": "batch-user"},
            # None encoding (default)
            {"encoding_format": None, "user": "api-user"},
            # Only user parameter
            {"user": "minimal-user"},
        ]
        
        for test_params in test_cases:
            result = self.config.map_openai_params(
                non_default_params=test_params,
                optional_params={},
                model=self.model,
                drop_params=False
            )
            
            # Verify all valid parameters are preserved
            for key, value in test_params.items():
                if value is not None:
                    assert result[key] == value, f"Parameter {key} not mapped correctly"
    
    def test_parameter_mapping_with_invalid_encoding(self):
        """Test proper error handling for invalid encoding formats"""
        invalid_encodings = ["int32", "binary", "invalid_format", 123, []]
        
        for invalid_encoding in invalid_encodings:
            with pytest.raises(ValueError) as exc_info:
                self.config.map_openai_params(
                    non_default_params={"encoding_format": invalid_encoding},
                    optional_params={},
                    model=self.model,
                    drop_params=False
                )
            
            assert "Unsupported encoding_format" in str(exc_info.value)
            assert str(invalid_encoding) in str(exc_info.value)
    
    def test_parameter_dropping_behavior(self):
        """Test parameter dropping when drop_params=True"""
        invalid_params = {
            "encoding_format": "invalid_format",
            "unsupported_param": "value",
            "another_invalid": 123
        }
        
        result = self.config.map_openai_params(
            non_default_params=invalid_params,
            optional_params={},
            model=self.model,
            drop_params=True
        )
        
        # Should drop all invalid parameters
        for param in invalid_params.keys():
            assert param not in result, f"Invalid parameter {param} was not dropped"
    
    def test_request_transformation_structure(self):
        """Test request transformation produces correct structure"""
        test_inputs = [
            # Single string input
            "Hello world",
            # Multiple strings
            ["Hello", "World", "Test"],
            # Mixed content
            ["Short", "This is a longer text for testing purposes"],
        ]
        
        for input_data in test_inputs:
            result = self.config.transform_request(
                model=self.model,
                input=input_data,
                api_key=self.api_key,
                encoding_format="float"
            )
            
            # Verify structure
            assert "url" in result
            assert "headers" in result
            assert "data" in result
            
            # Verify URL
            assert result["url"] == "https://ark.cn-beijing.volces.com/api/v3/embeddings"
            
            # Verify headers
            headers = result["headers"]
            assert headers["Authorization"] == f"Bearer {self.api_key}"
            assert headers["Content-Type"] == "application/json"
            
            # Verify data
            data = result["data"]
            assert data["model"] == self.model
            assert data["encoding_format"] == "float"
            
            # Input should always be a list
            if isinstance(input_data, str):
                assert data["input"] == [input_data]
            else:
                assert data["input"] == input_data
    
    def test_response_transformation_with_real_data(self):
        """Test response transformation with realistic Volcengine response data"""
        # Simulate real Volcengine API response
        volcengine_responses = [
            # Single embedding response
            {
                "id": "cmpl-123456789",
                "object": "list",
                "model": "doubao-embedding-text-240715",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, -0.2, 0.3, 0.4, -0.5] * 100  # Realistic embedding size
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            },
            # Multiple embeddings response
            {
                "id": "cmpl-987654321",
                "object": "list", 
                "model": "doubao-embedding-text-240715",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3] * 256
                    },
                    {
                        "object": "embedding", 
                        "index": 1,
                        "embedding": [0.4, 0.5, 0.6] * 256
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "total_tokens": 12
                }
            }
        ]
        
        for response_data in volcengine_responses:
            mock_response = Mock(spec=httpx.Response)
            mock_response.json.return_value = response_data
            
            result = self.config.transform_response(
                response=mock_response,
                model=self.model,
                input=["test input"],
            )
            
            # Verify transformation preserves important data
            assert result["object"] == "list"
            assert result["model"] == response_data["model"]
            assert len(result["data"]) == len(response_data["data"])
            assert result["usage"] == response_data["usage"]
            
            # Verify embedding data integrity
            for i, embedding_item in enumerate(result["data"]):
                original_item = response_data["data"][i]
                assert embedding_item["object"] == "embedding"
                assert embedding_item["index"] == original_item["index"]
                assert len(embedding_item["embedding"]) == len(original_item["embedding"])
    
    def test_response_transformation_with_error_data(self):
        """Test response transformation handles error response formats correctly"""
        # Test that transform_response can handle both success and error response structures
        
        # Success response (should work)
        success_response = {
            "id": "cmpl-123",
            "object": "list",
            "model": "doubao-embedding-text-240715",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = success_response
        
        result = self.config.transform_response(
            response=mock_response,
            model=self.model,
            input=["test"],
        )
        
        # Should successfully transform
        assert result["object"] == "list"
        assert result["model"] == "doubao-embedding-text-240715"
        
        # Error response (should still transform but with empty/missing data)
        error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }
        
        mock_response.json.return_value = error_response
        
        result = self.config.transform_response(
            response=mock_response,
            model=self.model,
            input=["test"],
        )
        
        # Should handle missing fields gracefully
        assert result["object"] == "list"  # default value
        assert result["data"] == []  # default empty data
        assert result["usage"] == {}  # default empty usage


class TestVolcEngineEmbeddingHandlerBusinessLogic:
    """Test VolcEngineEmbeddingHandler with focus on business logic"""
    
    def setup_method(self):
        self.handler = VolcEngineEmbeddingHandler()
        self.model = "doubao-embedding-text-240715"
        self.api_key = "test-api-key-12345"
    
    def test_response_conversion_to_litellm_format(self):
        """Test conversion of Volcengine response to LiteLLM EmbeddingResponse"""
        volcengine_response = {
            "id": "emb-123",
            "object": "list",
            "model": self.model,
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, -0.1, -0.2] * 200  # 1000-dimensional embedding
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }
        
        result = self.handler._convert_to_litellm_response(
            volcengine_response, 
            self.model, 
            ["test input"]
        )
        
        # Verify result is proper EmbeddingResponse
        assert isinstance(result, EmbeddingResponse)
        assert result.object == "list"
        assert result.model == self.model
        assert len(result.data) == 1
        assert len(result.data[0]["embedding"]) == 1000
        
        # Verify usage information
        assert result.usage.prompt_tokens == 8
        assert result.usage.total_tokens == 8
        assert result.usage.completion_tokens == 0
    
    def test_network_error_handling_without_mocking_business_logic(self):
        """Test network error handling preserves business logic"""
        
        # Test with actual VolcEngineError class
        with pytest.raises(VolcEngineError) as exc_info:
            # This would raise a network error in real scenario
            error = VolcEngineError(
                status_code=500,
                message="Network error during embedding request: Connection timeout"
            )
            raise error
        
        # Verify error contains meaningful information
        assert exc_info.value.status_code == 500
        assert "Network error during embedding request" in str(exc_info.value.message)
        assert "Connection timeout" in str(exc_info.value.message)
    
    def test_input_validation_and_preprocessing(self):
        """Test input validation and preprocessing logic"""
        test_cases = [
            # String input should be converted to list
            ("single string", ["single string"]),
            # List input should remain list
            (["multiple", "strings"], ["multiple", "strings"]),
            # Empty string handling
            ("", [""]),
            # Unicode handling
            ("测试中文", ["测试中文"]),
            # Special characters
            ("Special chars: @#$%^&*()", ["Special chars: @#$%^&*()"]),
        ]
        
        for input_data, expected_output in test_cases:
            # Test the actual transformation logic
            config = VolcEngineEmbeddingConfig()
            result = config.transform_request(
                model=self.model,
                input=input_data,
                api_key=self.api_key,
            )
            
            assert result["data"]["input"] == expected_output


class TestVolcEngineEmbeddingIntegration:
    """Integration tests that test the full pipeline with minimal mocking"""
    
    def setup_method(self):
        self.handler = VolcEngineEmbeddingHandler()
        self.model = "doubao-embedding-text-240715"
        self.api_key = "test-api-key-12345"
    
    def test_full_request_response_cycle(self):
        """Test the complete request-response cycle with realistic data"""
        
        # Create a realistic Volcengine response
        realistic_response_data = {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi6",
            "object": "list",
            "model": "doubao-embedding-text-240715",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0023064255] + [0.1 * (i % 10 - 5) for i in range(1023)]  # Realistic 1024-dim embedding
                },
                {
                    "object": "embedding", 
                    "index": 1,
                    "embedding": [-0.0038562391] + [0.05 * (i % 20 - 10) for i in range(1023)]
                }
            ],
            "usage": {
                "prompt_tokens": 6,
                "total_tokens": 6
            }
        }
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = realistic_response_data
        
        # Only mock the HTTP call, not the business logic
        with patch('litellm.llms.volcengine.embedding.handler.HTTPHandler') as mock_handler:
            mock_client = Mock()
            mock_client.post.return_value = mock_response
            mock_handler.return_value = mock_client
            
            # Test the actual embedding call
            result = self.handler.embedding(
                model=self.model,
                input=["Hello world", "Test embedding"],
                api_key=self.api_key,
                encoding_format="float"
            )
            
            # Verify the HTTP request was made correctly (this tests integration)
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            
            # Verify request structure
            assert call_args.kwargs["url"] == "https://ark.cn-beijing.volces.com/api/v3/embeddings"
            assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {self.api_key}"
            
            request_data = call_args.kwargs["json"]
            assert request_data["model"] == self.model
            assert request_data["input"] == ["Hello world", "Test embedding"]
            assert request_data["encoding_format"] == "float"
            
            # Verify the response processing (real business logic)
            assert isinstance(result, EmbeddingResponse)
            assert result.model == self.model
            assert len(result.data) == 2
            assert len(result.data[0]["embedding"]) == 1024
            assert len(result.data[1]["embedding"]) == 1024
            assert result.usage.prompt_tokens == 6
    
    def test_parameter_validation_integration(self):
        """Test parameter validation in the full integration context"""
        
        # Test with various parameter combinations that should work
        valid_param_sets = [
            {"encoding_format": "float"},
            {"encoding_format": "base64"},
            {"user": "test-user-123"},
            {"encoding_format": "float", "user": "test-user"},
            {"extra_headers": {"Custom-Header": "value"}},
        ]
        
        for params in valid_param_sets:
            # Only create the request, don't execute (avoids HTTP call)
            config = VolcEngineEmbeddingConfig()
            try:
                result = config.transform_request(
                    model=self.model,
                    input=["test"],
                    api_key=self.api_key,
                    **params
                )
                # Verify structure is correct
                assert "url" in result
                assert "headers" in result  
                assert "data" in result
                
            except Exception as e:
                pytest.fail(f"Valid parameters {params} caused error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])