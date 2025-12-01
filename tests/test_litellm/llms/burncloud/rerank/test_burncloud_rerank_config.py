"""Unit tests for BurnCloud rerank transformation."""

import unittest
from typing import Dict
from unittest.mock import patch, Mock

import httpx

from litellm.llms.burncloud.rerank.transformation import BurnCloudRerankConfig
from litellm.types.rerank import RerankResponse


class TestBurnCloudRerankConfig(unittest.TestCase):
    """Test suite for BurnCloudRerankConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudRerankConfig()
        self.test_model = "rerank-model"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_query = "What is the capital of France?"
        self.test_documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Madrid is the capital of Spain."
        ]

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        result = self.config.get_complete_url(
            api_base=self.test_api_base,
            model=self.test_model
        )

        expected = f"{self.test_api_base}/v1/rerank"
        self.assertEqual(result, expected)

    def test_get_complete_url_with_v1_base(self) -> None:
        """Test get_complete_url when API base ends with /v1."""
        api_base_with_v1 = f"{self.test_api_base}/v1"

        result = self.config.get_complete_url(
            api_base=api_base_with_v1,
            model=self.test_model
        )

        expected = f"{api_base_with_v1}/rerank"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        with patch("litellm.llms.burncloud.rerank.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                api_base=None,
                model=self.test_model
            )

            expected = f"{self.test_api_base}/v1/rerank"
            self.assertEqual(result, expected)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}
        optional_params: Dict = {}

        with patch("litellm.llms.burncloud.rerank.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                api_key=None,
                optional_params=optional_params
            )

            expected = {
                "Authorization": f"Bearer {self.test_api_key}",
                "accept": "application/json",
                "content-type": "application/json",
            }
            self.assertEqual(result, expected)

    def test_validate_environment_with_existing_auth_header(self) -> None:
        """Test environment validation with existing Authorization header."""
        custom_auth = "Custom token"
        headers = {"Authorization": custom_auth}
        optional_params: Dict = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.test_model,
            api_key=self.test_api_key,
            optional_params=optional_params
        )

        expected = {
            "Authorization": custom_auth,
            "accept": "application/json",
            "content-type": "application/json",
        }
        self.assertEqual(result, expected)

    def test_transform_rerank_response_success(self) -> None:
        """Test successful transformation of rerank response."""
        mock_response = Mock(spec=httpx.Response)
        response_data = {
            "id": "test-id",
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.95,
                    "document": "Paris is the capital of France."
                },
                {
                    "index": 1,
                    "relevance_score": 0.12
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 15
            }
        }
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.text = str(response_data)

        model_response = RerankResponse()
        # Create a mock logging object instead of instantiating the real one
        mock_logging_obj = Mock()
        request_data = {
            "model": self.test_model,
            "query": self.test_query,
            "documents": self.test_documents
        }
        optional_params: Dict = {}
        litellm_params: Dict = {}

        result = self.config.transform_rerank_response(
            model=self.test_model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            api_key=self.test_api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params
        )

        self.assertIsInstance(result, RerankResponse)
        self.assertEqual(result.id, "test-id")
        self.assertEqual(len(result.results), 2)
        self.assertIsNotNone(result.meta)

    def test_transform_rerank_response_json_error(self) -> None:
        """Test transformation when JSON parsing fails."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.status_code = 400
        mock_response.text = "Invalid JSON response"

        model_response = RerankResponse()
        # Create a mock logging object instead of instantiating the real one
        mock_logging_obj = Mock()
        request_data = {}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with self.assertRaises(Exception):
            self.config.transform_rerank_response(
                model=self.test_model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging_obj,
                api_key=self.test_api_key,
                request_data=request_data,
                optional_params=optional_params,
                litellm_params=litellm_params
            )
