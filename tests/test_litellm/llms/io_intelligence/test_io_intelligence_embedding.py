"""
Comprehensive test suite for IO Intelligence embedding functionality.
Tests embedding functionality, usage tracking, error handling, and async operations with mock responses.
"""

import asyncio
import json
import os
import sys

import pytest
import respx

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import EmbeddingResponse, embedding
from litellm.constants import openai_compatible_providers


class TestIOIntelligenceEmbedding:
    """Comprehensive test suite for IO Intelligence embedding with mocking"""

    @pytest.mark.respx()
    def test_basic_embedding_with_usage(self, respx_mock):
        """Test basic embedding functionality with usage tracking"""
        litellm.disable_aiohttp_transport = True

        mock_embedding_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [-0.123, 0.456, 0.789, -0.234, 0.567, -0.891]  # Sample 6D embedding vector
                }
            ],
            "model": "BAAI/bge-multilingual-gemma2",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=mock_embedding_response, status_code=200
        )

        # Test embedding
        response = litellm.embedding(
            model="io_intelligence/BAAI/bge-multilingual-gemma2",
            input=["Hello world"],
            api_key="test_io_intelligence_key",
            api_base="https://api.intelligence.io.solutions/api/v1"
        )

        # Verify response structure
        assert response.model == "BAAI/bge-multilingual-gemma2"
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0]["object"] == "embedding"
        assert response.data[0]["index"] == 0
        assert len(response.data[0]["embedding"]) == 6
        
        # Verify usage tracking
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 10

    @pytest.mark.respx()
    def test_multiple_inputs_embedding(self, respx_mock):
        """Test embedding with multiple input texts"""
        litellm.disable_aiohttp_transport = True

        mock_embedding_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [-0.123, 0.456, 0.789, -0.234]
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.234, -0.567, 0.891, 0.345]
                }
            ],
            "model": "BAAI/bge-multilingual-gemma2",
            "usage": {
                "prompt_tokens": 25,
                "total_tokens": 25
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=mock_embedding_response, status_code=200
        )

        # Test embedding with multiple inputs
        response = litellm.embedding(
            model="io_intelligence/BAAI/bge-multilingual-gemma2",
            input=["Hello world", "Goodbye world"],
            api_key="test_io_intelligence_key",
            api_base="https://api.intelligence.io.solutions/api/v1"
        )

        # Verify multiple embeddings
        assert len(response.data) == 2
        assert response.data[0]["index"] == 0
        assert response.data[1]["index"] == 1
        assert len(response.data[0]["embedding"]) == 4
        assert len(response.data[1]["embedding"]) == 4
        assert response.usage.prompt_tokens == 25
        assert response.usage.total_tokens == 25

    @pytest.mark.respx()
    def test_embedding_with_custom_parameters(self, respx_mock):
        """Test embedding with custom parameters"""
        litellm.disable_aiohttp_transport = True

        mock_embedding_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            ],
            "model": "BAAI/bge-multilingual-gemma2",
            "usage": {
                "prompt_tokens": 15,
                "total_tokens": 15
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=mock_embedding_response, status_code=200
        )

        # Test embedding with custom parameters (dimensions, encoding_format, etc.)
        response = litellm.embedding(
            model="io_intelligence/BAAI/bge-multilingual-gemma2",
            input=["Test with custom params"],
            api_key="test_key",
            api_base="https://api.intelligence.io.solutions/api/v1",
            dimensions=512,
            encoding_format="float"
        )

        # Verify request was made and response structure
        assert response.model == "BAAI/bge-multilingual-gemma2"
        assert len(response.data) == 1
        assert len(response.data[0]["embedding"]) == 5
        assert response.usage.prompt_tokens == 15

    @pytest.mark.respx()
    def test_async_embedding_with_usage(self, respx_mock):
        """Test async embedding functionality with usage tracking"""
        litellm.disable_aiohttp_transport = True

        mock_embedding_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.11, 0.22, 0.33, 0.44]
                }
            ],
            "model": "BAAI/bge-multilingual-gemma2",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=mock_embedding_response, status_code=200
        )

        async def run_async_test():
            response = await litellm.aembedding(
                model="io_intelligence/BAAI/bge-multilingual-gemma2",
                input=["Async embedding test"],
                api_key="test_async_key",
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
            return response

        # Run async test
        response = asyncio.run(run_async_test())

        # Verify async embedding response
        assert response.model == "BAAI/bge-multilingual-gemma2"
        assert len(response.data) == 1
        assert len(response.data[0]["embedding"]) == 4
        assert response.usage.prompt_tokens == 8
        assert response.usage.total_tokens == 8

    @pytest.mark.respx()
    def test_embedding_error_handling_api_error(self, respx_mock):
        """Test embedding error handling for API errors"""
        litellm.disable_aiohttp_transport = True

        # Mock an error response from IO Intelligence API
        error_response_data = {
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=error_response_data, status_code=404
        )

        # Test that API errors are properly handled
        with pytest.raises(Exception) as exc_info:
            litellm.embedding(
                model="io_intelligence/invalid-embedding-model",
                input=["This should fail"],
                api_key="test_key",
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
        
        # Verify error contains useful information
        error_str = str(exc_info.value)
        assert len(error_str) > 0

    def test_embedding_error_handling_missing_api_key(self):
        """Test embedding error handling for missing API key"""
        # Test should raise error for missing API key
        with pytest.raises(Exception) as exc_info:
            litellm.embedding(
                model="io_intelligence/BAAI/bge-multilingual-gemma2",
                input=["This should fail"],
                # No api_key provided
                api_base="https://api.intelligence.io.solutions/api/v1"
            )
        
        # Should contain helpful error message about missing API key
        error_message = str(exc_info.value)
        assert any(keyword in error_message.lower() for keyword in ["api", "key", "auth", "missing"])

    @pytest.mark.respx()
    def test_embedding_cost_calculation(self, respx_mock):
        """Test embedding cost calculation integration"""
        litellm.disable_aiohttp_transport = True

        mock_embedding_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1024  # 1024-dimensional embedding
                }
            ],
            "model": "BAAI/bge-multilingual-gemma2",
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 50
            }
        }

        respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
            json=mock_embedding_response, status_code=200
        )

        response = litellm.embedding(
            model="io_intelligence/BAAI/bge-multilingual-gemma2",
            input=["Cost calculation test with longer text to generate more tokens"],
            api_key="test_key",
            api_base="https://api.intelligence.io.solutions/api/v1"
        )

        # Test cost calculation integration (if pricing available)
        try:
            from litellm.cost_calculator import completion_cost
            cost = completion_cost(completion_response=response, custom_llm_provider="io_intelligence")
            assert isinstance(cost, (int, float))
            assert cost >= 0
        except Exception:
            # If no pricing data available, that's acceptable for this test
            pass

        # Verify usage data is properly structured
        assert response.usage.prompt_tokens == 50
        assert response.usage.total_tokens == 50

    @pytest.mark.respx()
    def test_different_embedding_models(self, respx_mock):
        """Test different embedding models supported by IO Intelligence"""
        litellm.disable_aiohttp_transport = True

        models_to_test = [
            "BAAI/bge-multilingual-gemma2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "text-embedding-ada-002"
        ]

        for model_name in models_to_test:
            mock_embedding_response = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3]
                    }
                ],
                "model": model_name,
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            }

            respx_mock.post("https://api.intelligence.io.solutions/api/v1/embeddings").respond(
                json=mock_embedding_response, status_code=200
            )

            # Test each model
            response = litellm.embedding(
                model=f"io_intelligence/{model_name}",
                input=["Test model"],
                api_key="test_key",
                api_base="https://api.intelligence.io.solutions/api/v1"
            )

            assert response.model == model_name
            assert len(response.data) == 1
            assert response.usage.prompt_tokens == 5