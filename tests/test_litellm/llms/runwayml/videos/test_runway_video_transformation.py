"""
Tests for RunwayML video generation transformation.
"""
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.runwayml.videos.transformation import RunwayMLVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


class TestRunwayMLVideoTransformation:
    """Test RunwayMLVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = RunwayMLVideoConfig()
        self.mock_logging_obj = Mock()

    def test_transform_video_create_request(self):
        """Test video creation request validates URL and payload structure."""
        prompt = "A high quality demo video of litellm ai gateway"
        api_base = "https://api.dev.runwayml.com/v1"
        
        data, files, url = self.config.transform_video_create_request(
            model="gen4_turbo",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "promptImage": "https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo",
                "duration": 5,
                "ratio": "1280:720"
            },
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Validate payload structure
        assert data["model"] == "gen4_turbo"
        assert data["promptText"] == prompt
        assert data["promptImage"].startswith("https://")
        assert data["ratio"] == "1280:720"
        assert data["duration"] == 5
        assert files == []
        
        # Validate URL has correct endpoint
        assert url == "https://api.dev.runwayml.com/v1/image_to_video"

    def test_transform_video_status_with_timestamp_handling(self):
        """Test status retrieval handles RunwayML's ISO 8601 timestamps correctly."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        # Test status request URL construction
        video_id = encode_video_id_with_provider(
            "63fd0f13-f29d-4e58-99d3-1cb9efa14a5b",
            "runwayml",
            "gen4_turbo"
        )
        api_base = "https://api.dev.runwayml.com/v1"
        
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        assert url == "https://api.dev.runwayml.com/v1/tasks/63fd0f13-f29d-4e58-99d3-1cb9efa14a5b"
        assert params == {}
        
        # Test status response with ISO 8601 timestamp parsing
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "63fd0f13-f29d-4e58-99d3-1cb9efa14a5b",
            "createdAt": "2025-11-11T21:48:50.448Z",
            "status": "SUCCEEDED",
            "completedAt": "2025-11-11T21:50:15.123Z",
            "output": ["https://dnznrvs05pmza.cloudfront.net/video.mp4"],
            "progress": 100
        }
        
        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="runwayml"
        )
        
        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        # Verify ISO 8601 timestamps are converted to Unix timestamps (integers)
        assert isinstance(result.created_at, int)
        assert result.created_at > 0
        assert isinstance(result.completed_at, int)
        assert result.completed_at > 0
        assert result.progress == 100

    def test_transform_video_content_extraction(self):
        """Test content retrieval extracts video URL from RunwayML response correctly."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        # Test content request URL
        video_id = encode_video_id_with_provider(
            "63fd0f13-f29d-4e58-99d3-1cb9efa14a5b",
            "runwayml",
            "gen4_turbo"
        )
        api_base = "https://api.dev.runwayml.com/v1"
        
        url, params = self.config.transform_video_content_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        assert url == "https://api.dev.runwayml.com/v1/tasks/63fd0f13-f29d-4e58-99d3-1cb9efa14a5b"
        
        # Test video URL extraction from response
        response_data = {
            "id": "test-id",
            "status": "SUCCEEDED",
            "output": ["https://dnznrvs05pmza.cloudfront.net/video.mp4"]
        }
        video_url = self.config._extract_video_url_from_response(response_data)
        assert video_url == "https://dnznrvs05pmza.cloudfront.net/video.mp4"
        
        # Test error handling when video is still processing
        processing_response = {
            "id": "test-id",
            "status": "RUNNING",
            "output": None
        }
        with pytest.raises(ValueError, match="still processing"):
            self.config._extract_video_url_from_response(processing_response)

    def test_full_video_workflow(self):
        """Test complete video generation workflow from creation to status check."""
        config = RunwayMLVideoConfig()
        mock_logging_obj = Mock()
        
        # Step 1: Create video
        prompt = "A high quality demo video of litellm ai gateway"
        api_base = "https://api.dev.runwayml.com/v1"
        data, files, url = config.transform_video_create_request(
            model="gen4_turbo",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={
                "promptImage": "https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo",
                "ratio": "1280:720",
                "duration": 5
            },
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        assert data["model"] == "gen4_turbo"
        assert url.endswith("/image_to_video")
        
        # Step 2: Parse creation response
        mock_create_response = Mock(spec=httpx.Response)
        mock_create_response.json.return_value = {
            "id": "test-video-id-123",
            "createdAt": "2025-11-11T21:48:50.448Z",
            "status": "PENDING"
        }
        
        video_obj = config.transform_video_create_response(
            model="gen4_turbo",
            raw_response=mock_create_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="runwayml",
            request_data=data
        )
        
        assert video_obj.status == "queued"
        assert video_obj.id.startswith("video_")
        
        # Step 3: Check completion status
        mock_status_response = Mock(spec=httpx.Response)
        mock_status_response.json.return_value = {
            "id": "test-video-id-123",
            "createdAt": "2025-11-11T21:48:50.448Z",
            "status": "SUCCEEDED",
            "completedAt": "2025-11-11T21:50:15.123Z",
            "output": ["https://dnznrvs05pmza.cloudfront.net/video.mp4"]
        }
        
        status_obj = config.transform_video_status_retrieve_response(
            raw_response=mock_status_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="runwayml"
        )
        
        assert status_obj.status == "completed"
        assert isinstance(status_obj.created_at, int)
        assert isinstance(status_obj.completed_at, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

