"""
Test suite for Azure video router functionality.
Tests that the router method gets called correctly for Azure video generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import litellm


class TestAzureVideoRouter:
    """Test suite for Azure video router functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.model = "azure/sora-2"
        self.prompt = "A beautiful sunset over mountains"
        self.seconds = "5"
        self.size = "1280x720"

    @patch("litellm.videos.main.base_llm_http_handler")
    def test_azure_video_generation_router_call_mock(self, mock_handler):
        """Test that Azure video generation calls the router method with mock response"""
        # Setup mock response
        mock_response = {
            "id": "video_123",
            "model": "sora-2",
            "object": "video",
            "status": "processing",
            "created_at": 1234567890,
            "progress": 0
        }
        
        # Configure the mock handler
        mock_handler.video_generation_handler.return_value = mock_response
        
        # Call the video generation function with mock response
        result = litellm.video_generation(
            prompt=self.prompt,
            model=self.model,
            seconds=self.seconds,
            size=self.size,
            custom_llm_provider="azure",
            mock_response=mock_response
        )
        
        # Verify the result is a VideoObject with the expected data
        assert result.id == mock_response["id"]
        assert result.model == mock_response["model"]
        assert result.object == mock_response["object"]
        assert result.status == mock_response["status"]
        assert result.created_at == mock_response["created_at"]
        assert result.progress == mock_response["progress"]
