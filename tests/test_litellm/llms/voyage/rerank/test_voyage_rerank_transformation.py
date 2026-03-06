"""
Tests for Voyage AI rerank transformation functionality.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.voyage.rerank.transformation import VoyageRerankConfig
from litellm.types.rerank import RerankResponse


class TestVoyageRerankTransform:
    def setup_method(self):
        self.config = VoyageRerankConfig()
        self.model = "rerank-2.5"

    def test_get_complete_url_default(self):
        """Test URL generation with default api_base."""
        url = self.config.get_complete_url(api_base=None, model=self.model)
        assert url == "https://api.voyageai.com/v1/rerank"

    def test_get_complete_url_custom_base(self):
        """Test URL generation with custom api_base."""
        api_base = "https://custom.api.com"
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == "https://custom.api.com/v1/rerank"

    def test_get_complete_url_with_trailing_slash(self):
        """Test URL generation with trailing slash in api_base."""
        api_base = "https://custom.api.com/"
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == "https://custom.api.com/v1/rerank"

    def test_get_complete_url_with_v1_suffix(self):
        """Test URL generation when api_base already has /v1."""
        api_base = "https://custom.api.com/v1"
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == "https://custom.api.com/v1/rerank"

    def test_get_complete_url_already_complete(self):
        """Test URL generation when api_base already has /v1/rerank."""
        api_base = "https://custom.api.com/v1/rerank"
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == "https://custom.api.com/v1/rerank"

    def test_map_cohere_rerank_params_basic(self):
        """Test basic parameter mapping for Voyage AI rerank."""
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
        # Voyage uses top_k instead of top_n
        assert params["top_k"] == 3
        assert params["return_documents"] is True

    def test_map_cohere_rerank_params_ignores_unsupported(self):
        """Test that unsupported params are silently ignored."""
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            rank_fields=["field1"],  # Not supported by Voyage AI
            max_chunks_per_doc=5,  # Not supported by Voyage AI
            max_tokens_per_doc=100,  # Not supported by Voyage AI
        )
        assert params["query"] == "test query"
        assert params["documents"] == ["doc1", "doc2"]
        # Unsupported params should not be in the result
        assert "rank_fields" not in params
        assert "max_chunks_per_doc" not in params
        assert "max_tokens_per_doc" not in params

    def test_transform_rerank_request(self):
        """Test request transformation for Voyage AI format."""
        optional_params = {
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "France is a country in Europe.",
            ],
            "top_k": 2,
            "return_documents": True,
        }

        request_body = self.config.transform_rerank_request(
            model=self.model, optional_rerank_params=optional_params, headers={}
        )

        assert request_body["model"] == "rerank-2.5"
        assert request_body["query"] == "What is the capital of France?"
        assert request_body["documents"] == optional_params["documents"]
        assert request_body["top_k"] == 2
        assert request_body["return_documents"] is True

    def test_transform_rerank_response_success(self):
        """Test successful response transformation."""
        # Mock Voyage AI response format
        response_data = {
            "object": "list",
            "data": [
                {"relevance_score": 0.88671875, "index": 0},
                {"relevance_score": 0.353515625, "index": 2},
                {"relevance_score": 0.33984375, "index": 1},
            ],
            "model": "rerank-2.5",
            "usage": {"total_tokens": 30},
        }

        # Create mock httpx response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
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
        assert len(result.results) == 3
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 0.88671875
        assert result.results[1]["index"] == 2
        assert result.results[1]["relevance_score"] == 0.353515625
        assert result.results[2]["index"] == 1
        assert result.results[2]["relevance_score"] == 0.33984375

        # Verify metadata
        assert result.meta["tokens"]["input_tokens"] == 30
        assert result.meta["billed_units"]["total_tokens"] == 30

    def test_transform_rerank_response_with_documents(self):
        """Test response transformation when return_documents is True."""
        response_data = {
            "object": "list",
            "data": [
                {
                    "relevance_score": 0.95,
                    "index": 0,
                    "document": "Paris is the capital of France.",
                },
                {
                    "relevance_score": 0.75,
                    "index": 1,
                    "document": "France is a country in Europe.",
                },
            ],
            "model": "rerank-2.5",
            "usage": {"total_tokens": 50},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.headers = {}

        mock_logging = MagicMock()
        model_response = RerankResponse()

        result = self.config.transform_rerank_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        assert len(result.results) == 2
        assert result.results[0]["document"]["text"] == "Paris is the capital of France."
        assert result.results[1]["document"]["text"] == "France is a country in Europe."

    def test_transform_rerank_response_missing_data(self):
        """Test that missing data raises ValueError."""
        response_data = {
            "object": "list",
            "model": "rerank-2.5",
            "usage": {"total_tokens": 10},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
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

    def test_transform_rerank_response_error_status(self):
        """Test error handling for non-200 status code."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
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

        assert "Unauthorized" in str(exc_info.value)

    def test_transform_rerank_response_invalid_json(self):
        """Test error handling for invalid JSON response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.status_code = 200
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

        assert "Failed to parse response" in str(exc_info.value)

    def test_get_supported_cohere_rerank_params(self):
        """Test getting supported parameters for Voyage AI rerank."""
        supported_params = self.config.get_supported_cohere_rerank_params(self.model)
        assert "query" in supported_params
        assert "documents" in supported_params
        assert "top_n" in supported_params
        assert "return_documents" in supported_params

    @patch("litellm.llms.voyage.rerank.transformation.get_secret_str")
    def test_validate_environment_missing_api_key(self, mock_get_secret_str):
        """Test that validate_environment raises error when API key is missing."""
        # Mock get_secret_str to return None for both environment variables
        mock_get_secret_str.return_value = None
        with pytest.raises(ValueError, match="Voyage AI API key is required"):
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
        assert headers["content-type"] == "application/json"

    def test_calculate_rerank_cost(self):
        """Test cost calculation for Voyage AI rerank."""
        from litellm.types.rerank import RerankBilledUnits

        billed_units = RerankBilledUnits(total_tokens=1000)
        model_info = {"input_cost_per_token": 0.00000005}  # $0.05 per 1M tokens

        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model=self.model,
            billed_units=billed_units,
            model_info=model_info,
        )

        assert abs(prompt_cost - 0.00005) < 1e-10  # 1000 * 0.00000005
        assert completion_cost == 0.0

    def test_calculate_rerank_cost_missing_info(self):
        """Test cost calculation returns 0 when info is missing."""
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model=self.model,
            billed_units=None,
            model_info=None,
        )

        assert prompt_cost == 0.0
        assert completion_cost == 0.0
