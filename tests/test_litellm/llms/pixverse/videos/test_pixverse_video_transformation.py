"""
Tests for Pixverse video generation transformation.
"""
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.pixverse.videos.transformation import PixverseVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


class TestPixverseVideoTransformation:
    """Test PixverseVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = PixverseVideoConfig()
        self.mock_logging_obj = Mock()

    def test_determine_endpoint_text_to_video(self):
        """Test endpoint determination for text-to-video (no input reference)."""
        endpoint = self.config._determine_endpoint(None)
        assert endpoint == "/video/text/generate"

        endpoint = self.config._determine_endpoint("")
        assert endpoint == "/video/text/generate"

    def test_determine_endpoint_image_to_video(self):
        """Test endpoint determination for image-to-video."""
        # Test with image URL
        endpoint = self.config._determine_endpoint("https://example.com/image.jpg")
        assert endpoint == "/video/image/generate"

        # Test with image data URI
        endpoint = self.config._determine_endpoint(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        assert endpoint == "/video/image/generate"

    def test_determine_endpoint_video_to_video(self):
        """Test endpoint determination for video-to-video (fusion)."""
        # Test with video URL
        endpoint = self.config._determine_endpoint("https://example.com/video.mp4")
        assert endpoint == "/video/video/generate"

        endpoint = self.config._determine_endpoint("https://example.com/video.mov")
        assert endpoint == "/video/video/generate"

        # Test with video data URI
        endpoint = self.config._determine_endpoint(
            "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
        )
        assert endpoint == "/video/video/generate"

    def test_transform_video_create_request_text_to_video(self):
        """Test video creation request for text-to-video."""
        prompt = "A high quality demo video of litellm ai gateway"
        api_base = "https://app-api.pixverse.ai/openapi/v2"

        data, files, url = self.config.transform_video_create_request(
            model="pixverse",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "resolution": "1280x720",
                "duration": 5.0,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Validate payload structure
        assert data["prompt"] == prompt
        assert data["resolution"] == "1280x720"
        assert data["duration"] == 5.0
        assert "image" not in data
        assert "video" not in data
        assert files == []

        # Validate URL has correct endpoint
        assert url == "https://app-api.pixverse.ai/openapi/v2/video/text/generate"

    def test_transform_video_create_request_image_to_video(self):
        """Test video creation request for image-to-video."""
        prompt = "A high quality demo video"
        api_base = "https://app-api.pixverse.ai/openapi/v2"
        image_url = "https://example.com/image.jpg"

        data, files, url = self.config.transform_video_create_request(
            model="pixverse",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "input_reference": image_url,
                "resolution": "1280x720",
                "duration": 5.0,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Validate payload structure
        assert data["prompt"] == prompt
        assert data["image"] == image_url
        assert "video" not in data
        assert data["resolution"] == "1280x720"
        assert data["duration"] == 5.0
        assert files == []

        # Validate URL has correct endpoint
        assert url == "https://app-api.pixverse.ai/openapi/v2/video/image/generate"

    def test_transform_video_create_request_video_to_video(self):
        """Test video creation request for video-to-video (fusion)."""
        prompt = "Transform this video into a surreal artistic style"
        api_base = "https://app-api.pixverse.ai/openapi/v2"
        video_url = "https://example.com/reference.mp4"

        data, files, url = self.config.transform_video_create_request(
            model="pixverse",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "input_reference": video_url,
                "resolution": "1920x1080",
                "duration": 10.0,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # Validate payload structure
        assert data["prompt"] == prompt
        assert data["video"] == video_url
        assert "image" not in data
        assert data["resolution"] == "1920x1080"
        assert data["duration"] == 10.0
        assert files == []

        # Validate URL has correct endpoint
        assert url == "https://app-api.pixverse.ai/openapi/v2/video/video/generate"

    def test_transform_video_status_with_timestamp_handling(self):
        """Test status retrieval handles Pixverse's ISO 8601 timestamps correctly."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        # Test status request URL construction
        video_id = encode_video_id_with_provider(
            "task_abc123def456", "pixverse", "pixverse"
        )
        api_base = "https://app-api.pixverse.ai/openapi/v2"

        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert (
            url
            == "https://app-api.pixverse.ai/openapi/v2/video/result/task_abc123def456"
        )
        assert params == {}

        # Test status response with ISO 8601 timestamp parsing
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "create_time": "2025-01-15T10:30:00Z",
                "status": 1,  # completed
                "modify_time": "2025-01-15T10:35:30Z",
                "url": "https://media.pixverse.ai/videos/task_abc123def456.mp4",
                "outputWidth": 1280,
                "outputHeight": 720,
            },
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="pixverse",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        # Verify ISO 8601 timestamps are converted to Unix timestamps (integers)
        assert isinstance(result.created_at, int)
        assert result.created_at > 0
        assert isinstance(result.completed_at, int)
        assert result.completed_at > 0
        assert result.id.startswith("video_")

    def test_transform_video_content_extraction(self):
        """Test content retrieval extracts video URL from Pixverse response correctly."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        # Test content request URL
        video_id = encode_video_id_with_provider("task_xyz789", "pixverse", "pixverse")
        api_base = "https://app-api.pixverse.ai/openapi/v2"

        url, params = self.config.transform_video_content_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://app-api.pixverse.ai/openapi/v2/video/result/task_xyz789"

        # Test video URL extraction from response
        response_data = {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "status": 1,  # completed
                "url": "https://media.pixverse.ai/videos/task_xyz789.mp4",
            },
        }
        video_url = self.config._extract_video_url_from_response(response_data)
        assert video_url == "https://media.pixverse.ai/videos/task_xyz789.mp4"

        # Test error handling when video is still processing
        processing_response = {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "status": 3,  # processing
            },
        }
        with pytest.raises(ValueError, match="still processing"):
            self.config._extract_video_url_from_response(processing_response)

        # Test error handling when video generation failed
        failed_response = {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "status": 2,  # failed
            },
        }
        with pytest.raises(ValueError, match="Video generation failed"):
            self.config._extract_video_url_from_response(failed_response)

    def test_status_mapping(self):
        """Test Pixverse status mapping to OpenAI format."""
        assert self.config._map_pixverse_status("pending") == "queued"
        assert self.config._map_pixverse_status("processing") == "in_progress"
        assert self.config._map_pixverse_status("completed") == "completed"
        assert self.config._map_pixverse_status("failed") == "failed"
        assert self.config._map_pixverse_status("cancelled") == "failed"
        # Test case insensitivity
        assert self.config._map_pixverse_status("PENDING") == "queued"
        assert self.config._map_pixverse_status("PROCESSING") == "in_progress"

    def test_timestamp_parsing(self):
        """Test ISO 8601 timestamp parsing."""
        # Valid timestamp
        timestamp = "2025-01-15T10:30:00Z"
        result = self.config._parse_pixverse_timestamp(timestamp)
        assert isinstance(result, int)
        assert result > 0

        # Invalid timestamp
        result = self.config._parse_pixverse_timestamp("invalid")
        assert result == 0

        # None timestamp
        result = self.config._parse_pixverse_timestamp(None)
        assert result == 0

    def test_full_video_workflow(self):
        """Test complete video generation workflow from creation to status check."""
        config = PixverseVideoConfig()
        mock_logging_obj = Mock()

        # Step 1: Create video (text-to-video)
        prompt = "A high quality demo video of litellm ai gateway"
        api_base = "https://app-api.pixverse.ai/openapi/v2"
        data, files, url = config.transform_video_create_request(
            model="pixverse",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "resolution": "1280x720",
                "duration": 5.0,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["prompt"] == prompt
        assert url.endswith("/video/text/generate")

        # Step 2: Parse creation response
        mock_create_response = Mock(spec=httpx.Response)
        mock_create_response.json.return_value = {
            "ErrCode": 0,
            "ErrMsg": "success",
            "Resp": {"video_id": 391504480090022, "credits": 45},
        }

        video_obj = config.transform_video_create_response(
            model="pixverse",
            raw_response=mock_create_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="pixverse",
            request_data=data,
        )

        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")

        # Step 3: Check completion status
        mock_status_response = Mock(spec=httpx.Response)
        mock_status_response.json.return_value = {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "create_time": "2025-01-15T10:30:00Z",
                "status": 1,  # completed
                "modify_time": "2025-01-15T10:35:30Z",
                "url": "https://media.pixverse.ai/videos/task_123abc.mp4",
                "outputWidth": 1280,
                "outputHeight": 720,
            },
        }

        status_obj = config.transform_video_status_retrieve_response(
            raw_response=mock_status_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="pixverse",
        )

        assert status_obj.status == "completed"
        assert isinstance(status_obj.created_at, int)
        assert isinstance(status_obj.completed_at, int)
        assert hasattr(status_obj, "_hidden_params")

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping."""
        video_params = {
            "input_reference": "https://example.com/image.jpg",
            "size": "1920x1080",
            "seconds": "10",
        }

        mapped = self.config.map_openai_params(
            video_create_optional_params=video_params,
            model="pixverse",
            drop_params=False,
        )

        assert mapped["input_reference"] == "https://example.com/image.jpg"
        assert mapped["aspect_ratio"] == "16:9"
        assert mapped["quality"] == "1080p"
        assert mapped["duration"] == 10
        # Model should not be hardcoded - removed default model assertion

    def test_determine_endpoint_with_query_params(self):
        """Test endpoint determination with URLs containing query parameters."""
        # Video URL with query parameters
        endpoint = self.config._determine_endpoint(
            "https://cdn.example.com/video.mp4?token=abc123&expires=456789"
        )
        assert endpoint == "/video/video/generate"

        # Image URL with query parameters
        endpoint = self.config._determine_endpoint(
            "https://cdn.example.com/image.jpg?size=large&quality=high"
        )
        assert endpoint == "/video/image/generate"

        # URL with fragment
        endpoint = self.config._determine_endpoint(
            "https://cdn.example.com/video.mov#timestamp=30"
        )
        assert endpoint == "/video/video/generate"

    def test_square_aspect_ratio(self):
        """Test that square dimensions map to 1:1 aspect ratio."""
        # Test 720x720 (square)
        aspect_ratio, quality = self.config._parse_size_to_pixverse_format("720x720")
        assert aspect_ratio == "1:1"
        assert quality == "720p"

        # Test 1080x1080 (square)
        aspect_ratio, quality = self.config._parse_size_to_pixverse_format("1080x1080")
        assert aspect_ratio == "1:1"
        assert quality == "1080p"

        # Verify landscape still works
        aspect_ratio, quality = self.config._parse_size_to_pixverse_format("1920x1080")
        assert aspect_ratio == "16:9"
        assert quality == "1080p"

        # Verify portrait still works
        aspect_ratio, quality = self.config._parse_size_to_pixverse_format("1080x1920")
        assert aspect_ratio == "9:16"
        assert quality == "1080p"

    def test_pixverse_format_to_size_roundtrip(self):
        """Test that _pixverse_format_to_size correctly converts aspect_ratio + quality back to size."""
        # Landscape 16:9
        assert self.config._pixverse_format_to_size("16:9", "720p") == "1280x720"
        assert self.config._pixverse_format_to_size("16:9", "1080p") == "1920x1080"

        # Portrait 9:16
        assert self.config._pixverse_format_to_size("9:16", "720p") == "720x1280"
        assert self.config._pixverse_format_to_size("9:16", "1080p") == "1080x1920"

        # Square 1:1
        assert self.config._pixverse_format_to_size("1:1", "720p") == "720x720"
        assert self.config._pixverse_format_to_size("1:1", "1080p") == "1080x1080"

    def test_model_parameter_mapping(self):
        """Test that model parameter is correctly forwarded to Pixverse."""
        # Test with Pixverse model version
        video_params = {
            "model": "v5.6",
            "size": "1920x1080",
        }
        mapped = self.config.map_openai_params(
            video_create_optional_params=video_params,
            model="pixverse",
            drop_params=False,
        )
        assert mapped["model"] == "v5.6"

        # Test that LiteLLM routing keys are not forwarded
        video_params_routing = {
            "model": "pixverse/v5.6",
            "size": "1920x1080",
        }
        mapped_routing = self.config.map_openai_params(
            video_create_optional_params=video_params_routing,
            model="pixverse",
            drop_params=False,
        )
        assert "model" not in mapped_routing  # Should be filtered out

    def test_unsupported_operations(self):
        """Test that unsupported operations raise NotImplementedError."""
        # Test video remix (not supported)
        with pytest.raises(NotImplementedError, match="remix is not yet supported"):
            self.config.transform_video_remix_request(
                video_id="test_id",
                prompt="test prompt",
                api_base="https://app-api.pixverse.ai/v1",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        # Test video list (not supported)
        with pytest.raises(NotImplementedError, match="listing is not yet supported"):
            self.config.transform_video_list_request(
                api_base="https://app-api.pixverse.ai/v1",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
