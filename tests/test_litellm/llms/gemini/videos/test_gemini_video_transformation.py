"""
Tests for Gemini (Veo) video generation transformation.
"""
import json
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
import httpx

from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.types.videos.main import VideoObject
from litellm.types.router import GenericLiteLLMParams


class TestGeminiVideoConfig:
    """Test GeminiVideoConfig transformation class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = GeminiVideoConfig()
        self.mock_logging_obj = Mock()

    def test_get_supported_openai_params(self):
        """Test that correct params are supported."""
        params = self.config.get_supported_openai_params("veo-3.0-generate-preview")
        
        assert "model" in params
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params

    def test_validate_environment_with_api_key(self):
        """Test environment validation with API key."""
        headers = {}
        result = self.config.validate_environment(
            headers=headers,
            model="veo-3.0-generate-preview",
            api_key="test-api-key-123"
        )
        
        assert "x-goog-api-key" in result
        assert result["x-goog-api-key"] == "test-api-key-123"
        assert "Content-Type" in result
        assert result["Content-Type"] == "application/json"

    @patch.dict('os.environ', {}, clear=True)
    def test_validate_environment_missing_api_key(self):
        """Test that missing API key raises error."""
        headers = {}
        
        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY is required"):
            self.config.validate_environment(
                headers=headers,
                model="veo-3.0-generate-preview",
                api_key=None
            )

    def test_get_complete_url(self):
        """Test URL construction for video generation."""
        url = self.config.get_complete_url(
            model="gemini/veo-3.0-generate-preview",
            api_base="https://generativelanguage.googleapis.com",
            litellm_params={}
        )
        
        expected = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-preview:predictLongRunning"
        assert url == expected

    def test_get_complete_url_default_api_base(self):
        """Test URL construction with default API base."""
        url = self.config.get_complete_url(
            model="gemini/veo-3.0-generate-preview",
            api_base=None,
            litellm_params={}
        )
        
        assert url.startswith("https://generativelanguage.googleapis.com")
        assert "veo-3.0-generate-preview:predictLongRunning" in url

    def test_transform_video_create_request(self):
        """Test transformation of video creation request."""
        prompt = "A cat playing with a ball of yarn"
        
        data, files = self.config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Check Veo format
        assert "instances" in data
        assert len(data["instances"]) == 1
        assert data["instances"][0]["prompt"] == prompt
        
        # Check no files are uploaded
        assert files is None
    
    def test_transform_video_create_request_with_params(self):
        """Test transformation with optional parameters."""
        prompt = "A cat playing with a ball of yarn"
        
        data, files = self.config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            video_create_optional_request_params={
                "aspectRatio": "16:9",
                "durationSeconds": 8,
                "resolution": "1080p"
            },
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Check Veo format with instances and parameters separated
        instance = data["instances"][0]
        assert instance["prompt"] == prompt
        
        # Parameters should be in a separate object
        assert "parameters" in data
        assert data["parameters"]["aspectRatio"] == "16:9"
        assert data["parameters"]["durationSeconds"] == 8
        assert data["parameters"]["resolution"] == "1080p"
    
    def test_map_openai_params(self):
        """Test parameter mapping from OpenAI format to Veo format."""
        openai_params = {
            "size": "1280x720",
            "seconds": "8",
            "input_reference": "test_image.jpg"
        }
        
        mapped = self.config.map_openai_params(
            video_create_optional_params=openai_params,
            model="veo-3.0-generate-preview",
            drop_params=False
        )
        
        # Check mappings (prompt is not mapped, it's passed separately)
        assert mapped["aspectRatio"] == "16:9"  # 1280x720 is landscape
        assert mapped["durationSeconds"] == 8
        assert mapped["image"] == "test_image.jpg"
    
    def test_convert_size_to_aspect_ratio(self):
        """Test size to aspect ratio conversion."""
        # Landscape
        assert self.config._convert_size_to_aspect_ratio("1280x720") == "16:9"
        assert self.config._convert_size_to_aspect_ratio("1920x1080") == "16:9"
        
        # Portrait
        assert self.config._convert_size_to_aspect_ratio("720x1280") == "9:16"
        assert self.config._convert_size_to_aspect_ratio("1080x1920") == "9:16"
        
        # Invalid (defaults to 16:9)
        assert self.config._convert_size_to_aspect_ratio("invalid") == "16:9"
        # Empty string returns None (no size specified)
        assert self.config._convert_size_to_aspect_ratio("") is None

    def test_transform_video_create_response(self):
        """Test transformation of video creation response."""
        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
            "metadata": {
                "createTime": "2024-11-04T10:00:00.123456Z"
            }
        }
        
        result = self.config.transform_video_create_response(
            model="veo-3.0-generate-preview",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini"
        )
        
        assert isinstance(result, VideoObject)
        # ID is base64 encoded with provider info
        assert result.id.startswith("video_")
        assert result.status == "processing"
        assert result.object == "video"
        assert result.created_at > 0

    def test_transform_video_status_retrieve_request(self):
        """Test transformation of status retrieve request."""
        video_id = "gemini::operations/generate_1234567890::veo-3.0"
        
        url, params = self.config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base="https://generativelanguage.googleapis.com",
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        assert "operations/generate_1234567890" in url
        assert "v1beta" in url
        assert params == {}

    def test_transform_video_status_retrieve_response_processing(self):
        """Test transformation of status response when still processing."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
            "done": False,
            "metadata": {
                "createTime": "2024-11-04T10:00:00.123456Z"
            }
        }
        
        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini"
        )
        
        assert isinstance(result, VideoObject)
        assert result.status == "processing"
        assert result.created_at > 0

    def test_transform_video_status_retrieve_response_completed(self):
        """Test transformation of status response when completed."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
            "done": True,
            "metadata": {
                "createTime": "2024-11-04T10:00:00.123456Z"
            },
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "files/abc123xyz"
                            }
                        }
                    ]
                }
            }
        }
        
        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini"
        )
        
        assert isinstance(result, VideoObject)
        assert result.status == "completed"
        assert result.created_at > 0

    @patch('litellm.module_level_client')
    def test_transform_video_content_request(self, mock_client):
        """Test transformation of content download request."""
        video_id = "gemini::operations/generate_1234567890::veo-3.0"
        
        # Mock the status response
        mock_status_response = Mock(spec=httpx.Response)
        mock_status_response.json.return_value = {
            "name": "operations/generate_1234567890",
            "done": True,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "files/abc123xyz"
                            }
                        }
                    ]
                }
            }
        }
        mock_status_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_status_response
        
        url, params = self.config.transform_video_content_request(
            video_id=video_id,
            api_base="https://generativelanguage.googleapis.com",
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Should return download URL
        assert "files/abc123xyz:download" in url
        assert params == {"alt": "media"}

    def test_transform_video_content_response_bytes(self):
        """Test transformation of content response (returns bytes directly)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.headers = httpx.Headers({
            "content-type": "video/mp4"
        })
        mock_response.content = b"fake_video_data"
        
        result = self.config.transform_video_content_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj
        )
        
        assert result == b"fake_video_data"

    def test_video_remix_not_supported(self):
        """Test that video remix raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video remix is not supported"):
            self.config.transform_video_remix_request(
                video_id="test_id",
                prompt="test prompt",
                api_base="https://test.com",
                litellm_params=GenericLiteLLMParams(),
                headers={}
            )

    def test_video_list_not_supported(self):
        """Test that video list raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video list is not supported"):
            self.config.transform_video_list_request(
                api_base="https://test.com",
                litellm_params=GenericLiteLLMParams(),
                headers={}
            )

    def test_video_delete_not_supported(self):
        """Test that video delete raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Video delete is not supported"):
            self.config.transform_video_delete_request(
                video_id="test_id",
                api_base="https://test.com",
                litellm_params=GenericLiteLLMParams(),
                headers={}
            )


class TestGeminiVideoIntegration:
    """Integration tests for Gemini video generation workflow."""

    def test_full_workflow_mock(self):
        """Test full workflow with mocked responses."""
        config = GeminiVideoConfig()
        mock_logging_obj = Mock()
        
        # Step 1: Create request with parameters
        prompt = "A beautiful sunset over mountains"
        data, files = config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            video_create_optional_request_params={
                "aspectRatio": "16:9",
                "durationSeconds": 8
            },
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Verify instances and parameters structure
        assert data["instances"][0]["prompt"] == prompt
        assert data["parameters"]["aspectRatio"] == "16:9"
        assert data["parameters"]["durationSeconds"] == 8
        
        # Step 2: Parse create response
        mock_create_response = Mock(spec=httpx.Response)
        mock_create_response.json.return_value = {
            "name": "operations/generate_abc123",
            "metadata": {
                "createTime": "2024-11-04T10:00:00.123456Z"
            }
        }
        
        video_obj = config.transform_video_create_response(
            model="veo-3.0-generate-preview",
            raw_response=mock_create_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="gemini"
        )
        
        assert video_obj.status == "processing"
        assert video_obj.id.startswith("video_")
        assert video_obj.created_at > 0
        
        # Step 3: Check status (completed)
        mock_status_response = Mock(spec=httpx.Response)
        mock_status_response.json.return_value = {
            "name": "operations/generate_abc123",
            "done": True,
            "metadata": {
                "createTime": "2024-11-04T10:00:00.123456Z"
            },
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "files/video123"
                            }
                        }
                    ]
                }
            }
        }
        
        status_obj = config.transform_video_status_retrieve_response(
            raw_response=mock_status_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="gemini"
        )
        
        assert status_obj.status == "completed"
        assert status_obj.created_at > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

