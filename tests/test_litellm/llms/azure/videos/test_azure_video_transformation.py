import json
import os
import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, Mock, mock_open
import pytest
import httpx

# Add the parent directory to the system path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

import litellm
from litellm.llms.azure.videos.transformation import AzureVideoConfig
from litellm.types.videos.main import VideoObject, VideoResponse, VideoCreateOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams


class TestAzureVideoConfig:
    """Test class for Azure video configuration and transformations."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = AzureVideoConfig()
        self.model = "azure/sora-2"
        self.api_base = "https://your-resource.openai.azure.com"
        self.api_key = "test-api-key"

    def test_get_supported_openai_params(self):
        """Test getting supported OpenAI parameters for video generation."""
        supported_params = self.config.get_supported_openai_params(self.model)
        
        expected_params = [
            "model",
            "prompt", 
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]
        
        assert supported_params == expected_params
        assert len(supported_params) == 7

    def test_map_openai_params(self):
        """Test mapping OpenAI parameters for video generation."""
        video_params = VideoCreateOptionalRequestParams(
            prompt="A beautiful sunset over mountains",
            seconds=10,
            size="1280x720",
            user="test_user"
        )
        
        result = self.config.map_openai_params(
            video_create_optional_params=video_params,
            model=self.model,
            drop_params=False
        )
        
        # Should return the same dict since no mapping is needed
        assert result["prompt"] == "A beautiful sunset over mountains"
        assert result["seconds"] == 10
        assert result["size"] == "1280x720"
        assert result["user"] == "test_user"

    def test_validate_environment_with_api_key(self):
        """Test environment validation with provided API key."""
        headers = {"Content-Type": "application/json"}
        
        result_headers = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key=self.api_key
        )
        
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == f"Bearer {self.api_key}"
        assert result_headers["Content-Type"] == "application/json"

    @patch('litellm.llms.azure.videos.transformation.get_secret_str')
    @patch('litellm.llms.azure.videos.transformation.litellm')
    def test_validate_environment_without_api_key(self, mock_litellm, mock_get_secret):
        """Test environment validation without provided API key."""
        mock_litellm.api_key = None
        mock_litellm.azure_key = None
        mock_get_secret.return_value = "secret-api-key"
        
        headers = {"Content-Type": "application/json"}
        
        result_headers = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key=None
        )
        
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == "Bearer secret-api-key"

    def test_get_complete_url(self):
        """Test URL construction for Azure video API."""
        litellm_params = {
            "api_base": self.api_base,
            "api_version": "2024-02-15-preview"
        }
        
        url = self.config.get_complete_url(
            model=self.model,
            api_base=self.api_base,
            litellm_params=litellm_params
        )
        
        # Should contain the Azure base URL and video endpoint
        assert "/openai/v1/videos" in url
        assert self.api_base in url

    def test_transform_video_create_request(self):
        """Test video creation request transformation."""
        video_params = {
            "seconds": 8,
            "size": "720x1280"
        }
        
        litellm_params = GenericLiteLLMParams(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key
        )
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        api_base = f"{self.api_base}/openai/v1/videos"
        
        data, files, url = self.config.transform_video_create_request(
            model=self.model,
            prompt="A cinematic shot of a city at night",
            api_base=api_base,
            video_create_optional_request_params=video_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert data["prompt"] == "A cinematic shot of a city at night"
        assert data["seconds"] == 8
        assert data["size"] == "720x1280"
        assert data["model"] == self.model
        # URL should be returned as-is for Azure
        assert url == api_base

    def test_transform_video_create_response(self):
        """Test video creation response transformation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "video_azure_123",
            "object": "video",
            "status": "queued",
            "created_at": 1712697600,
            "model": "sora-2"
        }
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_create_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert isinstance(result, VideoObject)
        assert result.id == "video_azure_123"
        assert result.object == "video"
        assert result.status == "queued"
        assert result.model == "sora-2"

    def test_transform_video_remix_response(self):
        """Test video remix response transformation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "video_remix_azure_123",
            "object": "video",
            "status": "queued",
            "created_at": 1712697600,
            "model": "sora-2",
            "remixed_from_video_id": "video_azure_123"
        }
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_remix_response(
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert isinstance(result, VideoObject)
        assert result.id == "video_remix_azure_123"
        assert result.status == "queued"
        assert hasattr(result, 'remixed_from_video_id')
        assert result.remixed_from_video_id == "video_azure_123"

    def test_transform_video_delete_response(self):
        """Test video delete response transformation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "video_azure_123",
            "object": "video",
            "deleted": True,
            "status": "deleted",
            "created_at": 1712697600
        }
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_delete_response(
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert isinstance(result, VideoObject)
        assert result.id == "video_azure_123"
        assert result.object == "video"
        assert result.status == "deleted"

    def test_transform_video_content_response(self):
        """Test video content response transformation."""
        mock_response = MagicMock()
        mock_response.content = b"fake video content"
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_content_response(
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert isinstance(result, bytes)
        assert result == b"fake video content"

    def test_url_construction_with_trailing_slash(self):
        """Test URL construction with API base that has trailing slash."""
        api_base_with_slash = "https://your-resource.openai.azure.com/"
        litellm_params = {"api_base": api_base_with_slash}
        
        url = self.config.get_complete_url(
            model=self.model,
            api_base=api_base_with_slash,
            litellm_params=litellm_params
        )
        
        # Should not have double slashes
        assert "//openai/v1/videos" not in url
        assert "/openai/v1/videos" in url

    def test_url_construction_without_trailing_slash(self):
        """Test URL construction with API base that doesn't have trailing slash."""
        api_base_without_slash = "https://your-resource.openai.azure.com"
        litellm_params = {"api_base": api_base_without_slash}
        
        url = self.config.get_complete_url(
            model=self.model,
            api_base=api_base_without_slash,
            litellm_params=litellm_params
        )
        
        # Should have proper slash separation
        assert "/openai/v1/videos" in url
        assert url.startswith(api_base_without_slash)

    def test_video_create_with_file_upload(self):
        """Test video creation with file upload (input_reference)."""
        video_params = {
            "seconds": 10,
            "input_reference": "test_image.png"
        }
        
        litellm_params = GenericLiteLLMParams(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key
        )
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        api_base = f"{self.api_base}/openai/v1/videos"
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=b"fake image data")):
                data, files, url = self.config.transform_video_create_request(
                    model=self.model,
                    prompt="A video with reference image",
                    api_base=api_base,
                    video_create_optional_request_params=video_params,
                    litellm_params=litellm_params,
                    headers=headers
                )
        
        assert data["prompt"] == "A video with reference image"
        assert data["seconds"] == 10
        assert len(files) == 1
        assert files[0][0] == "input_reference"
        assert url == api_base

    def test_error_handling_in_response_transformation(self):
        """Test error handling in response transformation methods."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }
        mock_response.status_code = 401
        
        logging_obj = MagicMock()
        
        # Test that error responses raise exceptions
        with pytest.raises(Exception):
            self.config.transform_video_create_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=logging_obj
            )

    def test_azure_specific_environment_validation(self):
        """Test Azure-specific environment validation with different key sources."""
        headers = {"Content-Type": "application/json"}
        
        # Test with azure_key
        with patch('litellm.llms.azure.videos.transformation.litellm') as mock_litellm:
            mock_litellm.api_key = None
            mock_litellm.azure_key = "azure-test-key"
            mock_litellm.openai_key = None
            
            result_headers = self.config.validate_environment(
                headers=headers,
                model=self.model,
                api_key=None
            )
            
            assert result_headers["Authorization"] == "Bearer azure-test-key"

    def test_usage_data_creation_in_video_create(self):
        """Test that usage data is created correctly in video create response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "video_azure_123",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "model": "sora-2",
            "seconds": "10"
        }
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_create_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert hasattr(result, 'usage')
        assert result.usage is not None
        assert "duration_seconds" in result.usage
        assert result.usage["duration_seconds"] == 10.0

    def test_usage_data_creation_in_video_remix(self):
        """Test that usage data is created correctly in video remix response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "video_remix_azure_123",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "model": "sora-2",
            "seconds": "15"
        }
        
        logging_obj = MagicMock()
        
        result = self.config.transform_video_remix_response(
            raw_response=mock_response,
            logging_obj=logging_obj
        )
        
        assert hasattr(result, 'usage')
        assert result.usage is not None
        assert "duration_seconds" in result.usage
        assert result.usage["duration_seconds"] == 15.0


if __name__ == "__main__":
    pytest.main([__file__])
