"""Unit tests for BurnCloud embedding transformation."""

import unittest
from typing import Dict, List, Union
from unittest.mock import patch

import httpx

from litellm.llms.burncloud.common_utils import BurnCloudError
from litellm.llms.burncloud.embedding.transformation import BurnCloudEmbeddingConfig
from litellm.types.llms.openai import AllMessageValues, AllEmbeddingInputValues


class TestBurnCloudEmbeddingConfig(unittest.TestCase):
    """Test suite for BurnCloudEmbeddingConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudEmbeddingConfig()
        self.test_model = "text-embedding-ada-002"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Hello"}
        ]
        self.test_input: AllEmbeddingInputValues = "Hello world"

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        result = self.config.get_complete_url(
            api_base=self.test_api_base,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{self.test_api_base}/v1/embeddings"
        self.assertEqual(result, expected)

    def test_get_complete_url_with_v1_base(self) -> None:
        """Test get_complete_url when API base ends with /v1."""
        api_base_with_v1 = f"{self.test_api_base}/v1"

        result = self.config.get_complete_url(
            api_base=api_base_with_v1,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{api_base_with_v1}/embeddings"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        with patch("litellm.llms.burncloud.embedding.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                api_base=None,
                api_key=self.test_api_key,
                model=self.test_model,
                optional_params={},
                litellm_params={}
            )

            expected = f"{self.test_api_base}/v1/embeddings"
            self.assertEqual(result, expected)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.embedding.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                messages=self.test_messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                api_base=None
            )

            expected = {
                "Authorization": f"Bearer {self.test_api_key}",
                "Content-Type": "application/json",
            }
            self.assertEqual(result, expected)

    def test_validate_environment_with_existing_auth_header(self) -> None:
        """Test environment validation with existing Authorization header."""
        custom_auth = "Custom token"
        headers = {"Authorization": custom_auth}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.test_model,
            messages=self.test_messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self.test_api_key,
            api_base=None
        )

        expected = {
            "Authorization": custom_auth,
            "Content-Type": "application/json",
        }
        self.assertEqual(result, expected)

    def test_get_supported_openai_params(self) -> None:
        """Test that supported OpenAI parameters are returned."""
        result = self.config.get_supported_openai_params(self.test_model)

        expected = ["dimensions", "encoding_format"]
        self.assertEqual(result, expected)

    def test_map_openai_params_with_dimensions(self) -> None:
        """Test mapping of dimensions parameter."""
        non_default_params = {"dimensions": 128}
        optional_params: Dict = {}
        drop_params = False

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        expected = {"dimensions": 128}
        self.assertEqual(result, expected)

    def test_map_openai_params_with_encoding_format(self) -> None:
        """Test mapping of encoding_format parameter."""
        non_default_params = {"encoding_format": "base64"}
        optional_params: Dict = {}
        drop_params = False

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        expected = {"encoding_format": "base64"}
        self.assertEqual(result, expected)

    def test_map_openai_params_with_both_params(self) -> None:
        """Test mapping of both supported parameters."""
        non_default_params = {
            "dimensions": 256,
            "encoding_format": "float"
        }
        optional_params: Dict = {}
        drop_params = False

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        self.assertEqual(result, non_default_params)

    def test_transform_embedding_request_with_text_input(self) -> None:
        """Test transformation of embedding request with text input."""
        optional_params = {"dimensions": 128}

        result = self.config.transform_embedding_request(
            model=self.test_model,
            input=self.test_input,
            optional_params=optional_params,
            headers={}
        )

        expected = {
            "model": self.test_model,
            "dimensions": 128,
            "input": [self.test_input]
        }
        self.assertEqual(result, expected)

    def test_transform_embedding_request_with_list_input(self) -> None:
        """Test transformation of embedding request with list input."""
        input_list: AllEmbeddingInputValues = ["Hello", "World"]
        optional_params: Dict = {}

        result = self.config.transform_embedding_request(
            model=self.test_model,
            input=input_list,
            optional_params=optional_params,
            headers={}
        )

        expected = {
            "model": self.test_model,
            "input": input_list
        }
        self.assertEqual(result, expected)

    def test_get_error_class(self) -> None:
        """Test creation of error class."""
        error_message = "Test error"
        status_code = 400
        headers: Union[dict, httpx.Headers] = {"Content-Type": "application/json"}

        result = self.config.get_error_class(error_message, status_code, headers)

        self.assertIsInstance(result, BurnCloudError)
        self.assertEqual(result.status_code, status_code)
        self.assertEqual(result.message, error_message)
        self.assertEqual(result.headers, headers)
