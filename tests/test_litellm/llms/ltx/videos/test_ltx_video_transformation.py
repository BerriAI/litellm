"""
Tests for LTX Video generation transformation.
"""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.ltx.videos.transformation import LTXVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


class TestLTXVideoTransformation:
    """Test LTXVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = LTXVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_supported_openai_params(self):
        """Test supported OpenAI parameters list."""
        params = self.config.get_supported_openai_params("ltx-2-fast")
        assert "model" in params
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params
        assert "user" in params
        assert "extra_headers" in params

    def test_map_openai_params_basic(self):
        """Test parameter mapping from OpenAI format to LTX format."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://example.com/image.jpg",
                "seconds": "5",
                "size": "1920x1080",
            },
            model="ltx-2-fast",
            drop_params=False,
        )

        assert mapped["image_uri"] == "https://example.com/image.jpg"
        assert mapped["duration"] == 5
        assert mapped["resolution"] == "1920x1080"

    def test_map_openai_params_passthrough(self):
        """Test that LTX-specific params are passed through."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={
                "fps": 30,
                "generate_audio": False,
                "camera_motion": "dolly_in",
            },
            model="ltx-2-fast",
            drop_params=False,
        )

        assert mapped["fps"] == 30
        assert mapped["generate_audio"] is False
        assert mapped["camera_motion"] == "dolly_in"

    def test_map_openai_params_seconds_int(self):
        """Test seconds conversion when provided as int."""
        mapped = self.config.map_openai_params(
            video_create_optional_params={"seconds": 10},
            model="ltx-2-fast",
            drop_params=False,
        )
        assert mapped["duration"] == 10

    def test_validate_environment(self):
        """Test authentication header setup."""
        headers = self.config.validate_environment(
            headers={},
            model="ltx-2-fast",
            api_key="test-api-key",
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="LTX API key is required"):
            self.config.validate_environment(
                headers={},
                model="ltx-2-fast",
                api_key=None,
            )

    def test_get_complete_url_default(self):
        """Test default API base URL."""
        url = self.config.get_complete_url(
            model="ltx-2-fast",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://api.ltx.video/v1"

    def test_get_complete_url_custom(self):
        """Test custom API base URL."""
        url = self.config.get_complete_url(
            model="ltx-2-fast",
            api_base="https://custom.api.com/v1/",
            litellm_params={},
        )
        assert url == "https://custom.api.com/v1"

    def test_transform_video_create_request_text_to_video(self):
        """Test text-to-video request transformation."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-fast",
            prompt="A serene mountain landscape at sunset",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "duration": 5,
                "resolution": "1920x1080",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "ltx-2-fast"
        assert data["prompt"] == "A serene mountain landscape at sunset"
        assert data["duration"] == 5
        assert data["resolution"] == "1920x1080"
        assert "image_uri" not in data
        assert files == []
        assert url == "https://api.ltx.video/v1/text-to-video"

    def test_transform_video_create_request_image_to_video(self):
        """Test image-to-video request transformation when image_uri is present."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-pro",
            prompt="Animate this image with gentle motion",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "image_uri": "https://example.com/source.jpg",
                "duration": 3,
                "resolution": "1280x720",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "ltx-2-pro"
        assert data["prompt"] == "Animate this image with gentle motion"
        assert data["image_uri"] == "https://example.com/source.jpg"
        assert files == []
        assert url == "https://api.ltx.video/v1/image-to-video"

    def test_transform_video_create_request_with_optional_params(self):
        """Test request with LTX-specific optional parameters."""
        data, files, url = self.config.transform_video_create_request(
            model="ltx-2-3-pro",
            prompt="A cinematic pan across a cityscape",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params={
                "duration": 8,
                "resolution": "1920x1080",
                "fps": 30,
                "generate_audio": True,
                "camera_motion": "dolly_in",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["fps"] == 30
        assert data["generate_audio"] is True
        assert data["camera_motion"] == "dolly_in"
        assert url == "https://api.ltx.video/v1/text-to-video"

    def test_transform_video_create_response_binary(self):
        """Test that binary response produces a completed VideoObject."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"\x00\x00\x00\x1cftypisom"  # fake video bytes

        request_data = {
            "model": "ltx-2-fast",
            "prompt": "test",
            "duration": 5,
            "resolution": "1920x1080",
        }

        result = self.config.transform_video_create_response(
            model="ltx-2-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=request_data,
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.created_at is not None
        assert result.created_at > 0
        assert result.completed_at is not None
        assert result.model == "ltx-2-fast"
        assert result.size == "1920x1080"
        assert result.seconds == "5"
        assert result.id.startswith("video_")
        assert result.usage is not None
        assert result.usage["duration_seconds"] == 5.0

    def test_transform_video_create_response_without_request_data(self):
        """Test response transformation without request data."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"\x00\x00\x00"

        result = self.config.transform_video_create_response(
            model="ltx-2-fast",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider=None,
            request_data=None,
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.id  # should have a UUID

    def test_unsupported_operations(self):
        """Test that unsupported operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.config.transform_video_content_request(
                video_id="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_content_response(
                raw_response=Mock(), logging_obj=self.mock_logging_obj
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_status_retrieve_request(
                video_id="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request(
                video_id="test",
                prompt="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request(
                video_id="test",
                api_base="",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request(
                api_base="", litellm_params=GenericLiteLLMParams(), headers={}
            )

    def test_full_text_to_video_workflow(self):
        """Test complete text-to-video workflow."""
        config = LTXVideoConfig()
        mock_logging_obj = Mock()

        # Step 1: Map params
        mapped = config.map_openai_params(
            video_create_optional_params={
                "seconds": "5",
                "size": "1920x1080",
            },
            model="ltx-2-fast",
            drop_params=False,
        )

        assert mapped["duration"] == 5
        assert mapped["resolution"] == "1920x1080"

        # Step 2: Create request
        data, files, url = config.transform_video_create_request(
            model="ltx-2-fast",
            prompt="A serene mountain landscape",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.ltx.video/v1/text-to-video"
        assert data["prompt"] == "A serene mountain landscape"
        assert data["model"] == "ltx-2-fast"
        assert data["duration"] == 5

        # Step 3: Parse response (binary)
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"fake-video-binary-data"

        video_obj = config.transform_video_create_response(
            model="ltx-2-fast",
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=data,
        )

        assert video_obj.status == "completed"
        assert video_obj.model == "ltx-2-fast"
        assert video_obj.seconds == "5"

    def test_full_image_to_video_workflow(self):
        """Test complete image-to-video workflow."""
        config = LTXVideoConfig()
        mock_logging_obj = Mock()

        # Step 1: Map params
        mapped = config.map_openai_params(
            video_create_optional_params={
                "input_reference": "https://example.com/image.jpg",
                "seconds": "3",
                "size": "1280x720",
            },
            model="ltx-2-3-pro",
            drop_params=False,
        )

        assert mapped["image_uri"] == "https://example.com/image.jpg"

        # Step 2: Create request
        data, files, url = config.transform_video_create_request(
            model="ltx-2-3-pro",
            prompt="Animate the scene with flowing water",
            api_base="https://api.ltx.video/v1",
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.ltx.video/v1/image-to-video"
        assert data["image_uri"] == "https://example.com/image.jpg"

        # Step 3: Parse response
        mock_response = Mock(spec=httpx.Response)
        mock_response.content = b"fake-video-binary-data"

        video_obj = config.transform_video_create_response(
            model="ltx-2-3-pro",
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="ltx",
            request_data=data,
        )

        assert video_obj.status == "completed"
        assert video_obj.size == "1280x720"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
