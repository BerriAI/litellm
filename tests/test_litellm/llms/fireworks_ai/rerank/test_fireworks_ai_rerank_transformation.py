"""
Tests for Fireworks AI rerank transformation functionality.
"""
import json
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.fireworks_ai.rerank.transformation import FireworksAIRerankConfig
from litellm.types.rerank import RerankResponse


class TestFireworksAIRerankTransform:
    def setup_method(self):
        self.config = FireworksAIRerankConfig()
        self.model = "fireworks_ai/fireworks/qwen3-reranker-8b"

    def test_get_complete_url(self):
        """Test URL generation for Fireworks AI rerank API."""
        # Test basic URL generation
        api_base = None
        model = "fireworks/qwen3-reranker-8b"
        url = self.config.get_complete_url(api_base, model)
        assert url == "https://api.fireworks.ai/inference/v1/rerank"

        # Test URL with custom api_base
        api_base = "https://api.fireworks.ai/inference/v1"
        url = self.config.get_complete_url(api_base, model)
        assert url == "https://api.fireworks.ai/inference/v1/rerank"

        # Test URL with trailing slash
        api_base_with_slash = "https://api.fireworks.ai/inference/v1/"
        url = self.config.get_complete_url(api_base_with_slash, model)
        assert url == "https://api.fireworks.ai/inference/v1/rerank"

    def test_map_cohere_rerank_params_basic(self):
        """Test basic parameter mapping for Fireworks AI rerank."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            top_n=3,
            return_documents=True,
        )
        assert params["query"] == "test query"
        assert params["documents"] == ["doc1", "doc2"]
        assert params["top_n"] == 3
        assert params["return_documents"] is True

    def test_map_cohere_rerank_params_ignores_unsupported(self):
        """Test that unsupported params are silently ignored."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            rank_fields=["field1"],  # Not supported by Fireworks AI
            max_chunks_per_doc=5,  # Not supported by Fireworks AI
            max_tokens_per_doc=100,  # Not supported by Fireworks AI
        )
        assert params["query"] == "test query"
        assert params["documents"] == ["doc1", "doc2"]
        # Unsupported params should not be in the result
        assert "rank_fields" not in params
        assert "max_chunks_per_doc" not in params
        assert "max_tokens_per_doc" not in params

    def test_transform_rerank_request(self):
        """Test request transformation for Fireworks AI format."""
        optional_params = {
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "France is a country in Europe.",
            ],
            "top_n": 2,
            "return_documents": True,
        }

        request_body = self.config.transform_rerank_request(
            model=self.model, optional_rerank_params=optional_params, headers={}
        )

        # Model should be transformed to include "fireworks/" prefix
        assert request_body["model"] == "fireworks/qwen3-reranker-8b"
        assert request_body["query"] == "What is the capital of France?"
        assert request_body["documents"] == optional_params["documents"]
        assert request_body["top_n"] == 2
        assert request_body["return_documents"] is True

    def test_transform_rerank_request_model_prefix_handling(self):
        """Test that model prefix is handled correctly."""
        # Test with fireworks_ai/ prefix
        optional_params = {
            "query": "test",
            "documents": ["doc1"],
        }
        request_body = self.config.transform_rerank_request(
            model="fireworks_ai/fireworks/qwen3-reranker-8b",
            optional_rerank_params=optional_params,
            headers={},
        )
        assert request_body["model"] == "fireworks/qwen3-reranker-8b"

        # Test with model already having fireworks/ prefix
        request_body = self.config.transform_rerank_request(
            model="fireworks/qwen3-reranker-8b",
            optional_rerank_params=optional_params,
            headers={},
        )
        assert request_body["model"] == "fireworks/qwen3-reranker-8b"

    def test_transform_rerank_request_missing_query(self):
        """Test that transform_rerank_request raises error for missing query."""
        optional_params = {
            "documents": ["doc1"],
        }

        with pytest.raises(ValueError, match="query is required"):
            self.config.transform_rerank_request(
                model=self.model, optional_rerank_params=optional_params, headers={}
            )

    def test_transform_rerank_request_missing_documents(self):
        """Test that transform_rerank_request raises error for missing documents."""
        optional_params = {
            "query": "test query",
        }

        with pytest.raises(ValueError, match="documents is required"):
            self.config.transform_rerank_request(
                model=self.model, optional_rerank_params=optional_params, headers={}
            )

    def test_transform_rerank_response_success(self):
        """Test successful response transformation."""
        # Mock Fireworks AI response format (uses "data" not "results", and document is a string)
        response_data = {
            "object": "list",
            "model": "accounts/fireworks/models/qwen3-reranker-8b",
            "data": [
                {
                    "index": 0,
                    "relevance_score": 0.95,
                    "document": "Paris is the capital of France.",
                },
                {
                    "index": 1,
                    "relevance_score": 0.75,
                    "document": "France is a country in Europe.",
                },
            ],
            "usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
        }

        # Create mock httpx response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

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
        # Fireworks AI doesn't return "id", so it uses "model" as the id
        assert result.id == "accounts/fireworks/models/qwen3-reranker-8b"
        assert len(result.results) == 2
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 0.95
        assert result.results[0]["document"]["text"] == "Paris is the capital of France."
        assert result.results[1]["index"] == 1
        assert result.results[1]["relevance_score"] == 0.75
        assert result.results[1]["document"]["text"] == "France is a country in Europe."

        # Verify metadata
        assert result.meta["tokens"]["input_tokens"] == 50
        assert result.meta["tokens"]["output_tokens"] == 50
        assert result.meta["billed_units"]["search_units"] == 100

    def test_transform_rerank_response_without_documents(self):
        """Test response transformation when return_documents is False."""
        response_data = {
            "object": "list",
            "model": "accounts/fireworks/models/qwen3-reranker-8b",
            "data": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.75},
            ],
            "usage": {
                "total_tokens": 50,
                "prompt_tokens": 30,
                "completion_tokens": 20,
            },
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        # Fireworks AI doesn't return "id", so it uses "model" as the id
        assert result.id == "accounts/fireworks/models/qwen3-reranker-8b"
        assert len(result.results) == 2
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 0.95
        # Document should not be present
        assert "document" not in result.results[0]

    def test_transform_rerank_response_missing_id(self):
        """Test response transformation when id is missing (should use model name or generate UUID)."""
        response_data = {
            "object": "list",
            "model": "accounts/fireworks/models/qwen3-reranker-8b",
            "data": [
                {"index": 0, "relevance_score": 0.95},
            ],
            "usage": {"total_tokens": 10},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        # Should use model name when id is missing
        assert result.id == "accounts/fireworks/models/qwen3-reranker-8b"

    def test_transform_rerank_response_missing_results(self):
        """Test that missing results raises ValueError."""
        response_data = {
            "object": "list",
            "model": "accounts/fireworks/models/qwen3-reranker-8b",
            "usage": {"total_tokens": 10},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        with pytest.raises(ValueError, match="No results found"):
            self.config.transform_rerank_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
            )

    def test_transform_rerank_response_invalid_json(self):
        """Test error handling for invalid JSON response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.status_code = 500
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        with pytest.raises(Exception) as exc_info:
            self.config.transform_rerank_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
            )

        # Should raise an error with appropriate message
        assert "Failed to parse response" in str(exc_info.value)

    def test_get_supported_cohere_rerank_params(self):
        """Test getting supported parameters for Fireworks AI rerank."""
        supported_params = self.config.get_supported_cohere_rerank_params(self.model)
        assert "query" in supported_params
        assert "documents" in supported_params
        assert "top_n" in supported_params
        assert "return_documents" in supported_params
        assert len(supported_params) == 4

    def test_validate_environment_missing_api_key(self):
        """Test that validate_environment raises error when API key is missing."""
        from unittest.mock import patch

        # Mock _get_api_key to return None
        with patch.object(self.config, "_get_api_key", return_value=None):
            with pytest.raises(ValueError, match="FIREWORKS_API_KEY is not set"):
                self.config.validate_environment(
                    headers={},
                    model=self.model,
                    api_key=None,
                )

    def test_validate_environment_with_api_key(self):
        """Test that validate_environment works with API key."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            api_key="test-api-key",
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

