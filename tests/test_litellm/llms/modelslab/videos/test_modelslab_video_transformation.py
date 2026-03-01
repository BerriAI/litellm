"""
Tests for ModelsLab video generation transformation.
All tests are mocked — no real network calls.
"""
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.modelslab.videos.transformation import ModelsLabVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


class TestModelsLabVideoTransformation:
    """Test ModelsLabVideoConfig transformation class."""

    def setup_method(self):
        self.config = ModelsLabVideoConfig()
        self.config._api_key = "test-api-key"
        self.mock_logging_obj = Mock()

    # -------------------------------------------------------------------------
    # validate_environment
    # -------------------------------------------------------------------------

    def test_validate_environment_no_auth_header(self):
        """Key-in-body auth: only Content-Type in headers, no Authorization."""
        with patch(
            "litellm.llms.modelslab.videos.transformation.get_secret_str",
            return_value="test-key",
        ):
            headers = self.config.validate_environment(
                headers={}, model="i2vgen-xl"
            )
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_validate_environment_raises_without_key(self):
        """Raises ValueError when no API key is available."""
        with patch(
            "litellm.llms.modelslab.videos.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="MODELSLAB_API_KEY"):
                self.config.validate_environment(headers={}, model="i2vgen-xl")

    # -------------------------------------------------------------------------
    # get_supported_openai_params / map_openai_params
    # -------------------------------------------------------------------------

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params("i2vgen-xl")
        assert "prompt" in params
        assert "input_reference" in params
        assert "size" in params
        assert "seconds" in params

    def test_map_openai_params_size_parsing(self):
        """'512x768' → width=512, height=768."""
        result = self.config.map_openai_params(
            video_create_optional_params={"size": "512x768"},
            model="i2vgen-xl",
            drop_params=False,
        )
        assert result["width"] == 512
        assert result["height"] == 768

    def test_map_openai_params_input_reference(self):
        """input_reference maps to init_image."""
        result = self.config.map_openai_params(
            video_create_optional_params={"input_reference": "https://example.com/img.jpg"},
            model="stable-video-diffusion",
            drop_params=False,
        )
        assert result["init_image"] == "https://example.com/img.jpg"

    # -------------------------------------------------------------------------
    # transform_video_create_request
    # -------------------------------------------------------------------------

    def test_transform_video_create_request_text2video(self):
        """text2video: key in body, model_id, prompt; URL is text2video endpoint."""
        data, files, url = self.config.transform_video_create_request(
            model="i2vgen-xl",
            prompt="A cat playing with a ball",
            api_base="https://modelslab.com/api/v6/video/text2video",
            video_create_optional_request_params={"width": 512, "height": 512},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["key"] == "test-api-key"
        assert data["model_id"] == "i2vgen-xl"
        assert data["prompt"] == "A cat playing with a ball"
        assert data["width"] == 512
        assert data["height"] == 512
        assert files == []
        assert "text2video" in url

    def test_transform_video_create_request_img2video(self):
        """img2video: init_image in body; URL is img2video endpoint."""
        data, files, url = self.config.transform_video_create_request(
            model="stable-video-diffusion",
            prompt="Camera panning right",
            api_base="https://modelslab.com/api/v6/video/img2video",
            video_create_optional_request_params={"init_image": "https://example.com/frame.jpg"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert data["key"] == "test-api-key"
        assert data["init_image"] == "https://example.com/frame.jpg"
        assert "img2video" in url

    # -------------------------------------------------------------------------
    # transform_video_create_response
    # -------------------------------------------------------------------------

    def test_transform_video_create_response_success(self):
        """Immediate success response → completed VideoObject with output_url."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "success",
            "request_id": "req_123",
            "output": ["https://cdn.modelslab.com/output/video.mp4"],
        }

        result = self.config.transform_video_create_response(
            model="i2vgen-xl",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="modelslab",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/video.mp4"

    def test_transform_video_create_response_processing_polls(self):
        """Processing status → _poll_sync() is called → returns completed VideoObject."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": "processing",
            "request_id": "req_456",
            "eta": 10,
        }

        poll_result = {
            "status": "success",
            "request_id": "req_456",
            "output": ["https://cdn.modelslab.com/output/video2.mp4"],
        }

        with patch.object(self.config, "_poll_sync", return_value=poll_result) as mock_poll:
            result = self.config.transform_video_create_response(
                model="i2vgen-xl",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
            )
            mock_poll.assert_called_once_with("req_456")

        assert result.status == "completed"
        assert result._hidden_params.get("output_url") == "https://cdn.modelslab.com/output/video2.mp4"

    # -------------------------------------------------------------------------
    # transform_video_status_retrieve_request
    # -------------------------------------------------------------------------

    def test_transform_video_status_retrieve_request(self):
        """Fetch URL includes request_id; body has 'key'."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        video_id = encode_video_id_with_provider("req_789", "modelslab", "i2vgen-xl")

        url, body = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base="https://modelslab.com/api/v6/video",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "req_789" in url
        assert "fetch" in url
        assert body["key"] == "test-api-key"
