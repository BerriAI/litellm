"""Unit tests for BurnCloud video transformation."""

import unittest
from typing import Dict, List, Union, Any, Tuple
from unittest.mock import patch, Mock
from io import BufferedReader
import httpx

from litellm.llms.burncloud.common_utils import BurnCloudError
from litellm.llms.burncloud.videos.transformation import BurnCloudVideoConfig
from litellm.types.videos.main import VideoObject
from litellm.types.router import GenericLiteLLMParams


class TestBurnCloudVideoConfig(unittest.TestCase):
    """Test suite for BurnCloudVideoConfig class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = BurnCloudVideoConfig()
        self.test_model = "video-model"
        self.test_api_key = "test-api-key"
        self.test_api_base = "https://api.burncloud.ai"
        self.test_prompt = "Create a video about space exploration"
        self.test_video_id = "video-123"

    def test_get_supported_openai_params(self) -> None:
        """Test that all supported OpenAI parameters are returned."""
        result = self.config.get_supported_openai_params(self.test_model)

        expected = [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]
        self.assertEqual(result, expected)

    def test_map_openai_params(self) -> None:
        """Test mapping of OpenAI parameters."""
        video_create_optional_params = {
            "seconds": 30,
            "size": "1080p"
        }
        drop_params = False

        result = self.config.map_openai_params(
            video_create_optional_params=video_create_optional_params,
            model=self.test_model,
            drop_params=drop_params
        )

        self.assertEqual(result, video_create_optional_params)

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key provided."""
        headers: Dict = {}

        with patch("litellm.llms.burncloud.videos.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_key

            result = self.config.validate_environment(
                headers=headers,
                model=self.test_model,
                api_key=None
            )

            expected = {
                "Authorization": f"Bearer {self.test_api_key}",
            }
            self.assertEqual(result, expected)

    def test_validate_environment_with_existing_auth_header(self) -> None:
        """Test environment validation with existing Authorization header."""
        custom_auth = "Custom token"
        headers = {"Authorization": custom_auth}

        result = self.config.validate_environment(
            headers=headers,
            model=self.test_model,
            api_key=self.test_api_key
        )

        expected = {
            "Authorization": custom_auth,
        }
        self.assertEqual(result, expected)

    def test_get_complete_url_with_valid_base(self) -> None:
        """Test get_complete_url with valid API base."""
        litellm_params: Dict = {}

        result = self.config.get_complete_url(
            model=self.test_model,
            api_base=self.test_api_base,
            litellm_params=litellm_params
        )

        expected = f"{self.test_api_base}/v1/videos"
        self.assertEqual(result, expected)

    def test_get_complete_url_with_v1_base(self) -> None:
        """Test get_complete_url when API base ends with /v1."""
        api_base_with_v1 = f"{self.test_api_base}/v1"
        litellm_params: Dict = {}

        result = self.config.get_complete_url(
            model=self.test_model,
            api_base=api_base_with_v1,
            litellm_params=litellm_params
        )

        expected = f"{api_base_with_v1}/videos"
        self.assertEqual(result, expected)

    def test_get_complete_url_without_api_base(self) -> None:
        """Test get_complete_url when API base is None."""
        litellm_params: Dict = {}

        with patch("litellm.llms.burncloud.videos.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = self.test_api_base

            result = self.config.get_complete_url(
                model=self.test_model,
                api_base=None,
                litellm_params=litellm_params
            )

            expected = f"{self.test_api_base}/v1/videos"
            self.assertEqual(result, expected)

    def test_transform_video_create_request(self) -> None:
        """Test transformation of video create request."""
        video_create_optional_request_params = {
            "model": self.test_model,
            "prompt": self.test_prompt,
            "seconds": 30,
            "extra_headers": {"X-Custom": "value"}
        }
        litellm_params = GenericLiteLLMParams()
        headers: Dict = {}

        result = self.config.transform_video_create_request(
            model=self.test_model,
            prompt=self.test_prompt,
            api_base=self.test_api_base,
            video_create_optional_request_params=video_create_optional_request_params,
            litellm_params=litellm_params,
            headers=headers
        )

        data, files_list, api_base = result
        self.assertIsInstance(data, dict)
        self.assertIsInstance(files_list, list)
        self.assertEqual(api_base, self.test_api_base)
        self.assertEqual(data["model"], self.test_model)
        self.assertEqual(data["prompt"], self.test_prompt)
        self.assertEqual(data["seconds"], 30)

    def test_transform_video_create_response(self) -> None:
        """Test transformation of video create response."""
        mock_response = Mock(spec=httpx.Response)
        response_data = {
            "id": "video-123",
            "object": "video",
            "status": "completed",
            "seconds": "5"
        }
        mock_response.json.return_value = response_data

        mock_logging_obj = Mock()
        request_data: Dict = {}

        result = self.config.transform_video_create_response(
            model=self.test_model,
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="burncloud",
            request_data=request_data
        )

        self.assertIsInstance(result, VideoObject)
        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.usage)

    def test_transform_video_content_request(self) -> None:
        """Test transformation of video content request."""
        litellm_params = GenericLiteLLMParams()
        headers: Dict = {}

        result = self.config.transform_video_content_request(
            video_id=self.test_video_id,
            api_base=self.test_api_base,
            litellm_params=litellm_params,
            headers=headers
        )

        url, data = result
        expected_url = f"{self.test_api_base}/{self.test_video_id}/content"
        self.assertEqual(url, expected_url)
        self.assertEqual(data, {})

    def test_transform_video_content_response(self) -> None:
        """Test transformation of video content response."""
        mock_response = Mock(spec=httpx.Response)
        test_content = b"fake video content"
        mock_response.content = test_content

        mock_logging_obj = Mock()

        result = self.config.transform_video_content_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj
        )

        self.assertEqual(result, test_content)

    def test_transform_video_status_retrieve_request(self) -> None:
        """Test transformation of video status retrieve request."""
        litellm_params = GenericLiteLLMParams()
        headers: Dict = {}

        result = self.config.transform_video_status_retrieve_request(
            video_id=self.test_video_id,
            api_base=self.test_api_base,
            litellm_params=litellm_params,
            headers=headers
        )

        url, data = result
        expected_url = f"{self.test_api_base}/{self.test_video_id}"
        self.assertEqual(url, expected_url)
        self.assertEqual(data, {})

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

    def test_add_image_to_files_with_bufferedReader(self) -> None:
        """Test adding BufferedReader image to files."""
        files_list: List[Tuple[str, Any]] = []
        mock_buffer = Mock(spec=BufferedReader)
        mock_buffer.name = "test.mp4"
        image = mock_buffer
        field_name = "input_reference"

        with patch("litellm.llms.burncloud.videos.transformation.ImageEditRequestUtils.get_image_content_type") as mock_get_content_type:
            mock_get_content_type.return_value = "video/mp4"

            self.config._add_image_to_files(
                files_list=files_list,
                image=image,
                field_name=field_name
            )

            self.assertEqual(len(files_list), 1)
            self.assertEqual(files_list[0][0], field_name)

    def test_add_image_to_files_with_bytes(self) -> None:
        """Test adding bytes image to files."""
        files_list: List[Tuple[str, Any]] = []
        image = b"fake video bytes"
        field_name = "input_reference"

        with patch("litellm.llms.burncloud.videos.transformation.ImageEditRequestUtils.get_image_content_type") as mock_get_content_type:
            mock_get_content_type.return_value = "video/mp4"

            self.config._add_image_to_files(
                files_list=files_list,
                image=image,
                field_name=field_name
            )

            self.assertEqual(len(files_list), 1)
            self.assertEqual(files_list[0][0], field_name)
