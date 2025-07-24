"""
Simple unit tests for CustomOpenAPISpec class.

Tests basic functionality of OpenAPI schema generation.
"""

from unittest.mock import Mock, patch

import pytest

from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec


class TestCustomOpenAPISpec:
    """Test suite for CustomOpenAPISpec class."""

    @pytest.fixture
    def base_openapi_schema(self):
        """Base OpenAPI schema for testing."""
        return {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/v1/chat/completions": {
                    "post": {
                        "summary": "Chat completions"
                    }
                },
                "/v1/embeddings": {
                    "post": {
                        "summary": "Embeddings"
                    }
                },
                "/v1/responses": {
                    "post": {
                        "summary": "Responses API"
                    }
                }
            }
        }

    @patch('litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema')
    def test_add_chat_completion_request_schema(self, mock_add_schema, base_openapi_schema):
        """Test that chat completion schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema
        
        with patch('litellm.proxy._types.ProxyChatCompletionRequest') as mock_model:
            result = CustomOpenAPISpec.add_chat_completion_request_schema(base_openapi_schema)
            
            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="ProxyChatCompletionRequest",
                paths=CustomOpenAPISpec.CHAT_COMPLETION_PATHS,
                operation_name="chat completion"
            )
            assert result == base_openapi_schema

    @patch('litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema')
    def test_add_embedding_request_schema(self, mock_add_schema, base_openapi_schema):
        """Test that embedding schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema
        
        with patch('litellm.types.embedding.EmbeddingRequest') as mock_model:
            result = CustomOpenAPISpec.add_embedding_request_schema(base_openapi_schema)
            
            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="EmbeddingRequest",
                paths=CustomOpenAPISpec.EMBEDDING_PATHS,
                operation_name="embedding"
            )
            assert result == base_openapi_schema

    @patch('litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema')
    def test_add_responses_api_request_schema(self, mock_add_schema, base_openapi_schema):
        """Test that responses API schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema
        
        with patch('litellm.types.llms.openai.ResponsesAPIRequestParams') as mock_model:
            result = CustomOpenAPISpec.add_responses_api_request_schema(base_openapi_schema)
            
            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="ResponsesAPIRequestParams",
                paths=CustomOpenAPISpec.RESPONSES_API_PATHS,
                operation_name="responses API"
            )
            assert result == base_openapi_schema 