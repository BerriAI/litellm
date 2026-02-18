"""
Tests for IBM watsonx.ai rerank transformation functionality.
"""
import json
import re
import uuid
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.watsonx.common_utils import (
    WatsonXAIError,
)
from litellm.llms.watsonx.rerank.transformation import IBMWatsonXRerankConfig
from litellm.types.rerank import RerankResponse


class TestIBMWatsonXRerankTransform:
    def setup_method(self):
        self.config = IBMWatsonXRerankConfig()
        self.model = "watsonx/cross-encoder/ms-marco-minilm-l-12-v2"

    def test_get_complete_url(self):
        """Test URL generation for IBM watsonx.ai rerank API."""

        api_base = "https://us-south.ml.cloud.ibm.com"
        model = "watsonx/cross-encoder/ms-marco-minilm-l-12-v2"
        url = self.config.get_complete_url(api_base, model)
        assert url == "https://us-south.ml.cloud.ibm.com/ml/v1/text/rerank?version=2024-03-13"

    def test_map_cohere_rerank_params_basic(self):
        """Test basic parameter mapping for IBM watsonx.ai rerank."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={
                "query": "hello",
                "documents": ["hello", "world"],
                "top_n": 2,
                "return_documents": True,
                "max_tokens_per_doc": 100,
            },
            model="test",
            drop_params=False,
            query="hello",
            documents=["hello", "world"],
        )
        assert params["query"] == "hello"
        assert params["inputs"] == [{"text": "hello"}, {"text": "world"}]
        assert params["parameters"]["return_options"]["top_n"] == 2
        assert params["parameters"]["return_options"]["inputs"] is True
        assert params["parameters"]["truncate_input_tokens"] == 100

    def test_transform_rerank_request(self):
        """Test request transformation for IBM watsonx.ai format."""
        optional_params = {
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "France is a country in Europe.",
            ],
            "top_n": 2,
            "return_documents": True,
            "project_id": uuid.uuid4(),
        }

        request_body = self.config.transform_rerank_request(
            model="cross-encoder/ms-marco-minilm-l-12-v2", optional_rerank_params=optional_params, headers={}
        )

        assert request_body["model_id"] == "cross-encoder/ms-marco-minilm-l-12-v2"
        assert request_body["project_id"] is not None
        assert request_body["query"] == "What is the capital of France?"
        assert request_body["documents"] == optional_params["documents"]
        assert request_body["top_n"] == 2
        assert request_body["return_documents"] is True
        
    def test_transform_rerank_response_success(self):
        """Test successful response transformation."""
        # Mock IBM watsonx.ai response format
        response_data = {
            "model_id": self.model,
            "results": [
                {
                    "index": 0,
                    "score": 6.53515625,
                    "input": {"text": "Python is great for beginners due to simple syntax."},
                },
                {"index": 1, "score": -7.1875, "input": {"text": "JavaScript runs in browsers and is versatile."}},
            ],
            "input_token_count": 62,
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
        # IBM watsonx.ai doesn't return "id", so it uses "model" as the id
        assert result.id == "watsonx/cross-encoder/ms-marco-minilm-l-12-v2"
        assert len(result.results) == 2
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 6.53515625
        assert result.results[0]["document"]["text"] == "Python is great for beginners due to simple syntax."
        assert result.results[1]["index"] == 1
        assert result.results[1]["relevance_score"] == -7.1875
        assert result.results[1]["document"]["text"] == "JavaScript runs in browsers and is versatile."

        # # Verify metadata
        assert result.meta["tokens"]["input_tokens"] == 62

    def test_transform_rerank_response_without_documents(self):
        """Test response transformation when return_documents is False."""
        response_data = {
            "model_id": self.model,
            "results": [
                {
                    "index": 0,
                    "score": 6.53515625,
                },
                {
                    "index": 1,
                    "score": -7.1875,
                },
            ],
            "input_token_count": 62,
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

        # Verify response structure
        # IBM watsonx.ai doesn't return "id", so it uses "model" as the id
        assert result.id == "watsonx/cross-encoder/ms-marco-minilm-l-12-v2"
        assert len(result.results) == 2

        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 6.53515625
        assert "document" not in result.results[0]

        assert result.results[1]["index"] == 1
        assert result.results[1]["relevance_score"] == -7.1875
        assert "document" not in result.results[1]

    def test_transform_rerank_response_missing_results(self):
        """Test that missing results raises ValueError."""
        response_data = {
            "model": self.model,
            "usage": {"total_tokens": 10},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        expected_error_msg = re.escape("No results found")

        with pytest.raises(ValueError, match=expected_error_msg):
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

        expected_error_msg = re.escape("Failed to parse response")

        with pytest.raises(Exception, match=expected_error_msg):
            self.config.transform_rerank_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
            )

    def test_get_supported_cohere_rerank_params(self):
        """Test getting supported parameters for IBM watsonx.ai rerank."""
        supported_params = self.config.get_supported_cohere_rerank_params(self.model)
        assert "query" in supported_params
        assert "documents" in supported_params
        assert "top_n" in supported_params
        assert "return_documents" in supported_params
        assert "max_tokens_per_doc" in supported_params
        assert len(supported_params) == 5
