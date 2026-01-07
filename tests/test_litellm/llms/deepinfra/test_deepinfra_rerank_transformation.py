"""
Tests for DeepInfra rerank transformation functionality.
Based on the test patterns from other rerank providers and the current DeepInfra implementation.
"""
import json
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.deepinfra.rerank.transformation import DeepinfraRerankConfig
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankResponse,
)


class TestDeepinfraRerankTransform:
    def setup_method(self):
        self.config = DeepinfraRerankConfig()
        self.model = "deepinfra/Qwen/Qwen3-Reranker-0.6B"

    def test_get_complete_url(self):
        """Test URL generation for DeepInfra rerank API."""
        # Test basic URL generation
        api_base = "https://api.deepinfra.com"
        model = "Qwen/Qwen3-Reranker-0.6B"
        url = self.config.get_complete_url(api_base, model)
        assert url == "https://api.deepinfra.com/inference/Qwen/Qwen3-Reranker-0.6B"

        # Test URL with slash at the end
        api_base_with_slash = "https://api.deepinfra.com/"
        url = self.config.get_complete_url(api_base_with_slash, model)
        assert url == "https://api.deepinfra.com/inference/Qwen/Qwen3-Reranker-0.6B"

        # Test URL with openai replacement
        api_base_openai = "https://api.deepinfra.com/openai"
        url = self.config.get_complete_url(api_base_openai, model)
        assert url == "https://api.deepinfra.com/inference/Qwen/Qwen3-Reranker-0.6B"

        # Test error when api_base is None
        with pytest.raises(ValueError, match="Deepinfra API Base is required"):
            self.config.get_complete_url(None, model)


    def test_map_cohere_rerank_params_basic(self):
        """Test basic parameter mapping for DeepInfra rerank."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={"documents": ["doc1", "doc2"]},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
        )
        assert params["queries"] == [
            "test query",
            "test query",
        ]  # DeepInfra requires queries to match documents length
        assert params["documents"] == ["doc1", "doc2"]

    def test_map_cohere_rerank_params_with_non_default(self):
        """Test parameter mapping with DeepInfra-specific parameters."""
        non_default_params = {
            "queries": ["custom query"],
            "documents": ["doc1", "doc2", "doc3"],
            "service_tier": "premium",
            "instruction": "custom instruction",
            "webhook": "https://webhook.example.com",
        }

        params = self.config.map_cohere_rerank_params(
            non_default_params=non_default_params,
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
        )

        # queries should override the query parameter (custom queries take precedence)
        assert params["queries"] == ["custom query"]
        assert params["documents"] == ["doc1", "doc2", "doc3"]
        assert params["service_tier"] == "premium"
        assert params["instruction"] == "custom instruction"
        assert params["webhook"] == "https://webhook.example.com"

    def test_transform_rerank_request(self):
        """Test request transformation for DeepInfra format."""
        optional_params = OptionalRerankParams(
            queries=["test query"],
            documents=["doc1", "doc2"],
            service_tier="default",
        )

        request_body = self.config.transform_rerank_request(
            model=self.model, optional_rerank_params=optional_params, headers={}
        )

        assert request_body["queries"] == ["test query"]
        assert request_body["documents"] == ["doc1", "doc2"]
        assert request_body["service_tier"] == "default"

    def test_transform_rerank_request_missing_documents(self):
        """Test that transform_rerank_request handles missing documents gracefully."""
        optional_params = OptionalRerankParams(queries=["test query"])

        # The current implementation doesn't validate documents, it just returns the params
        result = self.config.transform_rerank_request(
            model=self.model, optional_rerank_params=optional_params, headers={}
        )
        assert result == optional_params

    def test_transform_rerank_response_success(self):
        """Test successful response transformation."""
        # Mock DeepInfra response format
        response_data = {
            "scores": [0.9, 0.7, 0.3],
            "input_tokens": 42,
            "request_id": "test-request-123",
            "inference_status": {
                "status": "success",
                "runtime_ms": 150,
                "cost": 0.0001,
                "tokens_generated": 0,
                "tokens_input": 42,
            },
        }

        # Create mock httpx response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        # Create mock logging object
        mock_logging = MagicMock()

        model_response = RerankResponse()

        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        # Verify response structure
        assert result.id == "test-request-123"
        assert len(result.results) == 3
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 0.9
        assert result.results[1]["index"] == 1
        assert result.results[1]["relevance_score"] == 0.7
        assert result.results[2]["index"] == 2
        assert result.results[2]["relevance_score"] == 0.3

        # Verify metadata
        assert result.meta["tokens"]["input_tokens"] == 42
        assert result.meta["tokens"]["output_tokens"] == 0
        assert result.meta["billed_units"]["total_tokens"] == 42

        # Verify hidden params
        assert result._hidden_params["status"] == "success"
        assert result._hidden_params["runtime_ms"] == 150
        assert result._hidden_params["cost"] == 0.0001
        assert result._hidden_params["tokens_generated"] == 0
        assert result._hidden_params["tokens_input"] == 42
        assert result._hidden_params["model"] == self.model

        # Verify logging was called
        mock_logging.post_call.assert_called_once_with(
            original_response=mock_response.text
        )

    def test_transform_rerank_response_minimal(self):
        """Test response transformation with minimal data."""
        response_data = {
            "scores": [0.8, 0.2],
            "input_tokens": 20,
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        mock_logging = MagicMock()
        model_response = RerankResponse()

        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        # Should generate UUID when request_id is missing
        assert result.id is not None
        assert len(result.id) > 0

        # Should handle missing inference_status gracefully
        assert result._hidden_params["status"] == "unknown"
        assert result._hidden_params["runtime_ms"] == 0
        assert result._hidden_params["cost"] == 0.0

    def test_transform_rerank_response_error_fallback(self):
        """Test error handling and fallback in response transformation."""
        # Create a response that will cause JSON parsing to fail
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text = "Invalid JSON response"

        mock_logging = MagicMock()
        model_response = RerankResponse()

        # The current implementation should handle JSON parsing errors gracefully
        # by falling back to the parent implementation
        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )
        # Should return the original model_response when fallback occurs
        assert result == model_response

    def test_get_supported_cohere_rerank_params(self):
        """Test getting supported parameters for DeepInfra rerank."""
        supported_params = self.config.get_supported_cohere_rerank_params(self.model)
        assert "query" in supported_params
        assert "documents" in supported_params
        assert len(supported_params) == 2

    def test_query_replication_for_deepinfra_requirement(self):
        """Test that queries are replicated to match documents length as required by DeepInfra."""
        # Test with different document lengths
        test_cases = [
            (["doc1"], ["query1"]),
            (["doc1", "doc2"], ["query1", "query1"]),
            (["doc1", "doc2", "doc3"], ["query1", "query1", "query1"]),
        ]

        for documents, expected_queries in test_cases:
            params = self.config.map_cohere_rerank_params(
                non_default_params={},
                model=self.model,
                drop_params=False,
                query="query1",
                documents=documents,
            )
            assert (
                params["queries"] == expected_queries
            ), f"Failed for {len(documents)} documents"
            assert len(params["queries"]) == len(
                documents
            ), "Queries length must match documents length"

    def test_get_error_class_basic(self):
        """Test error class generation for basic error."""
        error_message = "Authentication failed"
        status_code = 401
        headers = {"content-type": "application/json"}

        with pytest.raises(Exception) as exc_info:
            self.config.get_error_class(error_message, status_code, headers)

        # The method should raise a BaseLLMException
        assert exc_info.value.args[0] == error_message

    def test_get_error_class_with_detail(self):
        """Test error class generation with DeepInfra error format."""
        error_data = {"detail": {"error": "Model not found"}}
        error_message = json.dumps(error_data)
        status_code = 404
        headers = {"content-type": "application/json"}

        with pytest.raises(Exception) as exc_info:
            self.config.get_error_class(error_message, status_code, headers)

        # Should extract the nested error message
        assert "Model not found" in str(exc_info.value)

    def test_get_error_class_with_string_detail(self):
        """Test error class generation with string detail."""
        error_data = {"detail": "Service unavailable"}
        error_message = json.dumps(error_data)
        status_code = 503
        headers = {"content-type": "application/json"}

        with pytest.raises(Exception) as exc_info:
            self.config.get_error_class(error_message, status_code, headers)

        # Should extract the string detail
        assert "Service unavailable" in str(exc_info.value)

    def test_get_error_class_invalid_json(self):
        """Test error class generation with invalid JSON."""
        error_message = "Invalid JSON error message"
        status_code = 500
        headers = {"content-type": "application/json"}

        with pytest.raises(Exception) as exc_info:
            self.config.get_error_class(error_message, status_code, headers)

        # Should use the original error message when JSON parsing fails
        assert "Invalid JSON error message" in str(exc_info.value)
