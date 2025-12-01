"""Unit tests for BurnCloud image generation transformation."""

import unittest
from typing import Dict, List, Optional
from unittest.mock import patch, Mock

import httpx

from litellm.llms.burncloud.image_generation.transformation import (
    BurnCloudImageGenerationConfig,
)
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse


class TestBurnCloudImageGenerationConfig(unittest.TestCase):
    """Test suite for BurnCloudImageGenerationConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudImageGenerationConfig()
        self.test_model = "dall-e-3"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_messages: List[AllMessageValues] = [
            {"role": "user", "content": "Generate an image of a cat"}
        ]
        self.test_prompt = "A beautiful sunset"

    def test_get_supported_openai_params(self) -> None:
        """Test that all supported OpenAI parameters are returned."""
        result = self.config.get_supported_openai_params(self.test_model)

        expected: List[OpenAIImageGenerationOptionalParams] = [
            "n",
            "response_format",
            "style",
            "quality",
            "size",
            "user",
        ]
        self.assertEqual(result, expected)

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        result = self.config.get_complete_url(
            api_base=self.test_api_base,
            api_key=self.test_api_key,
            model=self.test_model,
            optional_params={},
            litellm_params={}
        )

        expected = f"{self.test_api_base}/v1/images/generations"
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

        expected = f"{api_base_with_v1}/images/generations"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        with patch("litellm.llms.burncloud.image_generation.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                api_base=None,
                api_key=self.test_api_key,
                model=self.test_model,
                optional_params={},
                litellm_params={}
            )

            expected = f"{self.test_api_base}/v1/images/generations"
            self.assertEqual(result, expected)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}
        optional_params: Dict = {}
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.image_generation.transformation.get_secret_str") as mock_get_secret:
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

    def test_map_openai_params_with_supported_params(self) -> None:
        """Test mapping of supported OpenAI parameters."""
        non_default_params = {
            "n": 2,
            "size": "1024x1024",
            "quality": "hd"
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

    def test_map_openai_params_with_unsupported_params_and_drop(self) -> None:
        """Test mapping with unsupported parameters and drop_params=True."""
        non_default_params = {
            "unsupported_param": "value",
            "n": 1
        }
        optional_params: Dict = {}
        drop_params = True

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        expected = {"n": 1}
        self.assertEqual(result, expected)

    def test_map_openai_params_with_unsupported_params_and_no_drop(self) -> None:
        """Test mapping with unsupported parameters and drop_params=False raises error."""
        non_default_params = {
            "unsupported_param": "value",
            "n": 1
        }
        optional_params: Dict = {}
        drop_params = False

        with self.assertRaises(ValueError) as context:
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.test_model,
                drop_params=drop_params
            )

        self.assertIn("Parameter unsupported_param is not supported", str(context.exception))

    def test_transform_image_generation_response(self) -> None:
        """Test transformation of image generation response."""
        mock_response = Mock(spec=httpx.Response)
        response_data = {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image.jpg"
                }
            ]
        }
        mock_response.json.return_value = response_data

        model_response = ImageResponse()
        mock_logging_obj = Mock()
        request_data = {"prompt": self.test_prompt, "model": self.test_model}
        optional_params = {"size": "1024x1024", "quality": "standard", "response_format": "url"}
        litellm_params: Dict = {}
        encoding = None
        api_key = self.test_api_key
        json_mode = False

        with patch("litellm.llms.burncloud.image_generation.transformation.convert_to_model_response_object") as mock_convert:
            mock_convert.return_value = ImageResponse(**{
                "created": 1234567890,
                "data": [{"url": "https://example.com/image.jpg"}]
            })

            result = self.config.transform_image_generation_response(
                model=self.test_model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging_obj,
                request_data=request_data,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
                api_key=api_key,
                json_mode=json_mode
            )

            self.assertIsInstance(result, ImageResponse)
            mock_logging_obj.post_call.assert_called_once()
            mock_convert.assert_called_once()
