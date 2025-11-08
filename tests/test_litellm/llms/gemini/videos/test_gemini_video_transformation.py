"""
Tests for Gemini (Veo) video generation transformation.
"""
import json
import os
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject


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
        api_base = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-preview:predictLongRunning"
        
        data, files, url = self.config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={}
        )
        
        # Check Veo format
        assert "instances" in data
        assert len(data["instances"]) == 1
        assert data["instances"][0]["prompt"] == prompt
        
        # Check no files are uploaded
        assert files == []
        
        # URL should be returned as-is for Gemini
        assert url == api_base
    
    def test_transform_video_create_request_with_params(self):
        """Test transformation with optional parameters."""
        prompt = "A cat playing with a ball of yarn"
        api_base = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-preview:predictLongRunning"
        
        data, files, url = self.config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            api_base=api_base,
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

    def test_map_openai_params_default_duration(self):
        """Test that durationSeconds is omitted when not provided."""
        openai_params = {
            "size": "1280x720",
        }
        
        mapped = self.config.map_openai_params(
            video_create_optional_params=openai_params,
            model="veo-3.0-generate-preview",
            drop_params=False
        )
        
        assert mapped["aspectRatio"] == "16:9"
        assert "durationSeconds" not in mapped

    def test_map_openai_params_with_gemini_specific_params(self):
        """Test that Gemini-specific params are passed through correctly."""
        params_with_gemini_specific = {
            "size": "1280x720",
            "seconds": "8",
            "video": {"bytesBase64Encoded": "abc123", "mimeType": "video/mp4"},
            "negativePrompt": "no people",
            "referenceImages": [{"bytesBase64Encoded": "xyz789"}],
            "personGeneration": "allow"
        }
        
        mapped = self.config.map_openai_params(
            video_create_optional_params=params_with_gemini_specific,
            model="veo-3.1-generate-preview",
            drop_params=False
        )
        
        # Check OpenAI params are mapped
        assert mapped["aspectRatio"] == "16:9"
        assert mapped["durationSeconds"] == 8
        
        # Check Gemini-specific params are passed through
        assert "video" in mapped
        assert mapped["video"]["bytesBase64Encoded"] == "abc123"
        assert mapped["negativePrompt"] == "no people"
        assert mapped["referenceImages"] == [{"bytesBase64Encoded": "xyz789"}]
        assert mapped["personGeneration"] == "allow"

    def test_map_openai_params_with_extra_body(self):
        """Test that extra_body params are merged and extra_body is removed."""
        from litellm.videos.utils import VideoGenerationRequestUtils
        
        params_with_extra_body = {
            "seconds": "4",
            "extra_body": {
                "negativePrompt": "no people",
                "personGeneration": "allow",
                "resolution": "1080p"
            }
        }
        
        mapped = VideoGenerationRequestUtils.get_optional_params_video_generation(
            model="veo-3.0-generate-preview",
            video_generation_provider_config=self.config,
            video_generation_optional_params=params_with_extra_body
        )
        
        # Check OpenAI params are mapped
        assert mapped["durationSeconds"] == 4
        
        # Check extra_body params are merged
        assert mapped["negativePrompt"] == "no people"
        assert mapped["personGeneration"] == "allow"
        assert mapped["resolution"] == "1080p"
        
        # Check extra_body itself is removed
        assert "extra_body" not in mapped
    
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


    def test_transform_video_create_response_with_cost_tracking(self):
        """Test that duration is captured for cost tracking."""
        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
        }
        
        # Request data with durationSeconds in parameters
        request_data = {
            "instances": [{"prompt": "A test video"}],
            "parameters": {
                "durationSeconds": 5,
                "aspectRatio": "16:9"
            }
        }
        
        result = self.config.transform_video_create_response(
            model="gemini/veo-3.0-generate-preview",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
            request_data=request_data
        )
        
        assert isinstance(result, VideoObject)
        assert result.usage is not None, "Usage should be set"
        assert "duration_seconds" in result.usage, "duration_seconds should be in usage"
        assert result.usage["duration_seconds"] == 5.0, f"Expected 5.0, got {result.usage['duration_seconds']}"

    def test_transform_video_create_response_cost_tracking_with_different_durations(self):
        """Test cost tracking with different duration values."""
        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
        }
        
        # Test with 8 seconds
        request_data_8s = {
            "instances": [{"prompt": "Test"}],
            "parameters": {"durationSeconds": 8}
        }
        
        result_8s = self.config.transform_video_create_response(
            model="gemini/veo-3.1-generate-preview",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
            request_data=request_data_8s
        )
        
        assert result_8s.usage["duration_seconds"] == 8.0
        
        # Test with 4 seconds
        request_data_4s = {
            "instances": [{"prompt": "Test"}],
            "parameters": {"durationSeconds": 4}
        }
        
        result_4s = self.config.transform_video_create_response(
            model="gemini/veo-3.1-fast-generate-preview",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
            request_data=request_data_4s
        )
        
        assert result_4s.usage["duration_seconds"] == 4.0

    def test_transform_video_create_response_cost_tracking_no_duration(self):
        """Test that usage defaults to 8 seconds when no duration in request."""
        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_1234567890",
        }
        
        # Request data without durationSeconds (should default to 8 seconds for Google Veo)
        request_data = {
            "instances": [{"prompt": "A test video"}],
            "parameters": {
                "aspectRatio": "16:9"
            }
        }
        
        result = self.config.transform_video_create_response(
            model="gemini/veo-3.0-generate-preview",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="gemini",
            request_data=request_data
        )
        
        assert isinstance(result, VideoObject)
        # When no duration is provided, it defaults to 8 seconds (Google Veo default)
        assert result.usage is not None
        assert "duration_seconds" in result.usage
        assert result.usage["duration_seconds"] == 8.0, "Should default to 8 seconds when not provided (Google Veo default)"

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
        
        # Should return download URL (may or may not include :download suffix)
        assert "files/abc123xyz" in url
        # Params are empty for Gemini file URIs
        assert params == {}

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
        api_base = "https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-preview:predictLongRunning"
        data, files, url = config.transform_video_create_request(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            api_base=api_base,
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


class TestGeminiVideoCostTracking:
    """Test cost tracking for Gemini video generation."""
    
    def test_cost_calculation_with_duration(self):
        """Test that cost is calculated correctly using duration from usage."""
        # Test VEO 2.0 ($0.35/second)
        cost_veo2 = video_generation_cost(
            model="gemini/veo-2.0-generate-001",
            duration_seconds=5.0,
            custom_llm_provider="gemini"
        )
        expected_veo2 = 0.35 * 5.0  # $1.75
        assert abs(cost_veo2 - expected_veo2) < 0.001, f"Expected ${expected_veo2}, got ${cost_veo2}"
        
        # Test VEO 3.0 ($0.75/second)
        cost_veo3 = video_generation_cost(
            model="gemini/veo-3.0-generate-preview",
            duration_seconds=8.0,
            custom_llm_provider="gemini"
        )
        expected_veo3 = 0.75 * 8.0  # $6.00
        assert abs(cost_veo3 - expected_veo3) < 0.001, f"Expected ${expected_veo3}, got ${cost_veo3}"
        
        # Test VEO 3.1 Standard ($0.40/second)
        cost_veo31 = video_generation_cost(
            model="gemini/veo-3.1-generate-preview",
            duration_seconds=10.0,
            custom_llm_provider="gemini"
        )
        expected_veo31 = 0.40 * 10.0  # $4.00
        assert abs(cost_veo31 - expected_veo31) < 0.001, f"Expected ${expected_veo31}, got ${cost_veo31}"
        
        # Test VEO 3.1 Fast ($0.15/second)
        cost_veo31_fast = video_generation_cost(
            model="gemini/veo-3.1-fast-generate-preview",
            duration_seconds=6.0,
            custom_llm_provider="gemini"
        )
        expected_veo31_fast = 0.15 * 6.0  # $0.90
        assert abs(cost_veo31_fast - expected_veo31_fast) < 0.001, f"Expected ${expected_veo31_fast}, got ${cost_veo31_fast}"
    
    def test_cost_calculation_end_to_end(self):
        """Test complete cost tracking flow: request -> response -> cost calculation."""
        config = GeminiVideoConfig()
        mock_logging_obj = Mock()
        
        # Create request with duration
        request_data = {
            "instances": [{"prompt": "A beautiful sunset"}],
            "parameters": {"durationSeconds": 5}
        }
        
        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "name": "operations/generate_test123",
        }
        
        # Transform response
        video_obj = config.transform_video_create_response(
            model="gemini/veo-3.0-generate-preview",
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            custom_llm_provider="gemini",
            request_data=request_data
        )
        
        # Verify usage has duration
        assert video_obj.usage is not None
        assert "duration_seconds" in video_obj.usage
        duration = video_obj.usage["duration_seconds"]
        
        # Calculate cost using the duration from usage
        cost = video_generation_cost(
            model="gemini/veo-3.0-generate-preview",
            duration_seconds=duration,
            custom_llm_provider="gemini"
        )
        
        # Verify cost calculation (VEO 3.0 is $0.75/second)
        expected_cost = 0.75 * 5.0  # $3.75
        assert abs(cost - expected_cost) < 0.001, f"Expected ${expected_cost}, got ${cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

