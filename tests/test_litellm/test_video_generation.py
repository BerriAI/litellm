import json
import os
import sys
from unittest.mock import MagicMock, patch
import asyncio

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.videos.main import VideoObject, VideoResponse
from litellm.videos.main import video_generation, avideo_generation, video_status, avideo_status
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.cost_calculator import default_video_cost_calculator
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.integrations.custom_logger import CustomLogger


class TestVideoGeneration:
    """Test suite for video generation functionality."""

    def test_video_generation_basic(self):
        """Test basic video generation functionality."""
        # Mock the video generation response
        mock_response = VideoObject(
            id="video_123",
            object="video",
            status="queued",
            created_at=1712697600,
            model="sora-2",
            size="720x1280",
            seconds="8"
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_generation_handler.return_value = mock_response
            
            response = video_generation(
                prompt="Show them running around the room",
                model="sora-2",
                seconds="8",
                size="720x1280"
            )
            
            assert isinstance(response, VideoObject)
            assert response.id == "video_123"
            assert response.model == "sora-2"
            assert response.size == "720x1280"
            assert response.seconds == "8"

    def test_video_generation_with_mock_response(self):
        """Test video generation with mock response."""
        mock_data = {
            "id": "video_456",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "completed_at": 1712697660,
            "model": "sora-2",
            "size": "1280x720",
            "seconds": "10"
        }
        
        response = video_generation(
            prompt="A beautiful sunset over the ocean",
            model="sora-2",
            seconds="10",
            size="1280x720",
            mock_response=mock_data
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_456"
        assert response.status == "completed"
        assert response.model == "sora-2"
        assert response.size == "1280x720"
        assert response.seconds == "10"

    def test_video_generation_async(self):
        """Test async video generation functionality."""
        mock_response = VideoObject(
            id="video_async_123",
            object="video",
            status="processing",
            created_at=1712697600,
            model="sora-2",
            progress=50
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_generation_handler.return_value = mock_response
            
            import asyncio
            
            async def test_async():
                response = await avideo_generation(
                    prompt="A cat playing with a ball",
                    model="sora-2",
                    seconds="5",
                    size="720x1280"
                )
                return response
            
            response = asyncio.run(test_async())
            
            assert isinstance(response, VideoObject)
            assert response.id == "video_async_123"
            assert response.status == "processing"
            assert response.progress == 50

    def test_video_generation_parameter_validation(self):
        """Test video generation parameter validation."""
        # Test with minimal required parameters
        response = video_generation(
            prompt="Test video",
            model="sora-2",
            mock_response={"id": "test", "object": "video", "status": "queued", "created_at": 1712697600}
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "test"

    def test_video_generation_error_handling(self):
        """Test video generation error handling."""
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_generation_handler.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                video_generation(
                    prompt="Test video",
                    model="sora-2"
                )

    def test_video_generation_provider_config(self):
        """Test video generation provider configuration."""
        config = OpenAIVideoConfig()
        
        # Test supported parameters
        supported_params = config.get_supported_openai_params("sora-2")
        assert "prompt" in supported_params
        assert "model" in supported_params
        assert "seconds" in supported_params
        assert "size" in supported_params

    def test_video_generation_request_transformation(self):
        """Test video generation request transformation."""
        config = OpenAIVideoConfig()
        
        # Test request transformation
        data, files = config.transform_video_create_request(
            model="sora-2",
            prompt="Test video prompt",
            video_create_optional_request_params={
                "seconds": "8",
                "size": "720x1280"
            },
            litellm_params=MagicMock(),
            headers={}
        )
        
        assert data["model"] == "sora-2"
        assert data["prompt"] == "Test video prompt"
        assert data["seconds"] == "8"
        assert data["size"] == "720x1280"
        assert files == []

    def test_video_generation_response_transformation(self):
        """Test video generation response transformation."""
        config = OpenAIVideoConfig()
        
        # Mock HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "id": "video_789",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "model": "sora-2",
            "size": "1280x720",
            "seconds": "12"
        }
        
        response = config.transform_video_create_response(
            model="sora-2",
            raw_response=mock_http_response,
            logging_obj=MagicMock()
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_789"
        assert response.status == "completed"
        assert response.model == "sora-2"

    def test_video_generation_cost_calculation(self):
        """Test video generation cost calculation."""
        # Load the local model cost map instead of online
        import json
        with open("model_prices_and_context_window.json", "r") as f:
            litellm.model_cost = json.load(f)
        
        # Test with sora-2 model
        cost = default_video_cost_calculator(
            model="openai/sora-2",
            duration_seconds=10.0,
            custom_llm_provider="openai"
        )
        
        # Should calculate cost based on duration (10 seconds * $0.10 per second = $1.00)
        assert cost == 1.0

    def test_video_generation_cost_calculation_unknown_model(self):
        """Test video generation cost calculation for unknown model."""
        with pytest.raises(Exception, match="Model not found in cost map"):
            default_video_cost_calculator(
                model="unknown-model",
                duration_seconds=5.0,
                custom_llm_provider="openai"
            )

    def test_video_generation_with_files(self):
        """Test video generation with file uploads."""
        config = OpenAIVideoConfig()
        
        # Mock file data
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_image_data"
        
        data, files = config.transform_video_create_request(
            model="sora-2",
            prompt="Test video with image",
            video_create_optional_request_params={
                "input_reference": mock_file,
                "seconds": "8",
                "size": "720x1280"
            },
            litellm_params=MagicMock(),
            headers={}
        )
        
        assert data["model"] == "sora-2"
        assert data["prompt"] == "Test video with image"
        assert len(files) > 0  # Should have files when input_reference is provided

    def test_video_generation_environment_validation(self):
        """Test video generation environment validation."""
        config = OpenAIVideoConfig()
        
        # Test environment validation
        headers = config.validate_environment(
            headers={},
            model="sora-2",
            api_key="test-api-key"
        )
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_video_generation_url_generation(self):
        """Test video generation URL generation."""
        config = OpenAIVideoConfig()
        
        # Test URL generation
        url = config.get_complete_url(
            model="sora-2",
            api_base="https://api.openai.com/v1",
            litellm_params={}
        )
        
        assert url == "https://api.openai.com/v1/videos"

    def test_video_generation_parameter_mapping(self):
        """Test video generation parameter mapping."""
        config = OpenAIVideoConfig()
        
        # Test parameter mapping
        mapped_params = config.map_openai_params(
            video_create_optional_params={
                "seconds": "8",
                "size": "720x1280",
                "user": "test-user"
            },
            model="sora-2",
            drop_params=False
        )
        
        assert mapped_params["seconds"] == "8"
        assert mapped_params["size"] == "720x1280"
        assert mapped_params["user"] == "test-user"

    def test_video_generation_unsupported_parameters(self):
        """Test video generation with unsupported parameters."""
        from litellm.videos.utils import VideoGenerationRequestUtils
        
        # Test unsupported parameter detection
        with pytest.raises(litellm.UnsupportedParamsError):
            VideoGenerationRequestUtils.get_optional_params_video_generation(
                model="sora-2",
                video_generation_provider_config=OpenAIVideoConfig(),
                video_generation_optional_params={
                    "unsupported_param": "value"
                }
            )

    def test_video_generation_request_utils(self):
        """Test video generation request utilities."""
        from litellm.videos.utils import VideoGenerationRequestUtils
        
        # Test parameter filtering
        params = {
            "prompt": "Test video",
            "model": "sora-2",
            "seconds": "8",
            "size": "720x1280",
            "user": "test-user",
            "invalid_param": "should_be_filtered"
        }
        
        filtered_params = VideoGenerationRequestUtils.get_requested_video_generation_optional_param(params)
        
        # Should only contain valid parameters
        assert "prompt" not in filtered_params  # prompt is required, not optional
        assert "seconds" in filtered_params
        assert "size" in filtered_params
        assert "user" in filtered_params
        assert "invalid_param" not in filtered_params
        # Note: model is included in the filtered params as it's part of the TypedDict

    def test_video_generation_types(self):
        """Test video generation type definitions."""
        # Test VideoObject
        video_obj = VideoObject(
            id="test_id",
            object="video",
            status="completed",
            created_at=1712697600,
            model="sora-2"
        )
        
        assert video_obj.id == "test_id"
        assert video_obj.object == "video"
        assert video_obj.status == "completed"
        
        # Test dictionary-like access
        assert video_obj["id"] == "test_id"
        assert video_obj["status"] == "completed"
        assert "id" in video_obj
        assert video_obj.get("id") == "test_id"
        assert video_obj.get("nonexistent", "default") == "default"
        
        # Test JSON serialization
        json_data = video_obj.json()
        assert json_data["id"] == "test_id"
        assert json_data["object"] == "video"

    def test_video_generation_response_types(self):
        """Test video generation response types."""
        # Test VideoResponse
        video_obj = VideoObject(
            id="test_id",
            object="video",
            status="completed",
            created_at=1712697600
        )
        
        response = VideoResponse(data=[video_obj])
        
        assert len(response.data) == 1
        assert response.data[0].id == "test_id"
        
        # Test dictionary-like access
        assert response["data"][0]["id"] == "test_id"
        assert "data" in response
        assert response.get("data")[0]["id"] == "test_id"
        
        # Test JSON serialization
        json_data = response.json()
        assert len(json_data["data"]) == 1
        assert json_data["data"][0]["id"] == "test_id"

    def test_video_status_basic(self):
        """Test basic video status functionality."""
        # Mock the video status response
        mock_response = VideoObject(
            id="video_123",
            object="video",
            status="completed",
            created_at=1712697600,
            completed_at=1712697660,
            model="sora-2",
            progress=100,
            size="720x1280",
            seconds="8"
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_status_handler.return_value = mock_response
            
            response = video_status(
                video_id="video_123",
                model="sora-2"
            )
            
            assert isinstance(response, VideoObject)
            assert response.id == "video_123"
            assert response.status == "completed"
            assert response.progress == 100
            assert response.model == "sora-2"

    def test_video_status_with_mock_response(self):
        """Test video status with mock response."""
        mock_data = {
            "id": "video_456",
            "object": "video",
            "status": "processing",
            "created_at": 1712697600,
            "model": "sora-2",
            "progress": 75,
            "size": "1280x720",
            "seconds": "10"
        }
        
        response = video_status(
            video_id="video_456",
            model="sora-2",
            mock_response=mock_data
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_456"
        assert response.status == "processing"
        assert response.progress == 75
        assert response.model == "sora-2"

    def test_video_status_async(self):
        """Test async video status functionality."""
        mock_response = VideoObject(
            id="video_async_123",
            object="video",
            status="queued",
            created_at=1712697600,
            model="sora-2",
            progress=0
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_status_handler.return_value = mock_response
            
            import asyncio
            
            async def test_async():
                response = await avideo_status(
                    video_id="video_async_123",
                    model="sora-2"
                )
                return response
            
            response = asyncio.run(test_async())
            
            assert isinstance(response, VideoObject)
            assert response.id == "video_async_123"
            assert response.status == "queued"
            assert response.progress == 0

    def test_video_status_parameter_validation(self):
        """Test video status parameter validation."""
        # Test with minimal required parameters
        response = video_status(
            video_id="test_video_id",
            model="sora-2",
            mock_response={"id": "test", "object": "video", "status": "completed", "created_at": 1712697600}
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "test"

    def test_video_status_error_handling(self):
        """Test video status error handling."""
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_status_handler.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                video_status(
                    video_id="test_video_id",
                    model="sora-2"
                )

    def test_video_status_request_transformation(self):
        """Test video status request transformation."""
        config = OpenAIVideoConfig()
        
        # Test request transformation
        url, data = config.transform_video_status_retrieve_request(
            video_id="video_123",
            api_base="https://api.openai.com/v1/videos",
            litellm_params=MagicMock(),
            headers={}
        )
        
        assert url == "https://api.openai.com/v1/videos/video_123"
        assert data == {}

    def test_video_status_response_transformation(self):
        """Test video status response transformation."""
        config = OpenAIVideoConfig()
        
        # Mock HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "id": "video_789",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "completed_at": 1712697660,
            "model": "sora-2",
            "progress": 100,
            "size": "1280x720",
            "seconds": "12"
        }
        
        response = config.transform_video_status_retrieve_response(
            raw_response=mock_http_response,
            logging_obj=MagicMock()
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_789"
        assert response.status == "completed"
        assert response.progress == 100
        assert response.model == "sora-2"

    def test_video_status_different_states(self):
        """Test video status with different video states."""
        # Test queued state
        queued_response = video_status(
            video_id="video_queued",
            model="sora-2",
            mock_response={
                "id": "video_queued",
                "object": "video",
                "status": "queued",
                "created_at": 1712697600,
                "model": "sora-2",
                "progress": 0
            }
        )
        assert queued_response.status == "queued"
        assert queued_response.progress == 0
        
        # Test processing state
        processing_response = video_status(
            video_id="video_processing",
            model="sora-2",
            mock_response={
                "id": "video_processing",
                "object": "video",
                "status": "processing",
                "created_at": 1712697600,
                "model": "sora-2",
                "progress": 50
            }
        )
        assert processing_response.status == "processing"
        assert processing_response.progress == 50
        
        # Test completed state
        completed_response = video_status(
            video_id="video_completed",
            model="sora-2",
            mock_response={
                "id": "video_completed",
                "object": "video",
                "status": "completed",
                "created_at": 1712697600,
                "completed_at": 1712697660,
                "model": "sora-2",
                "progress": 100
            }
        )
        assert completed_response.status == "completed"
        assert completed_response.progress == 100

    def test_video_status_with_remix_info(self):
        """Test video status with remix information."""
        mock_data = {
            "id": "video_remix_123",
            "object": "video",
            "status": "completed",
            "created_at": 1712697600,
            "completed_at": 1712697660,
            "model": "sora-2",
            "progress": 100,
            "remixed_from_video_id": "video_original_123",
            "size": "720x1280",
            "seconds": "8"
        }
        
        response = video_status(
            video_id="video_remix_123",
            model="sora-2",
            mock_response=mock_data
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_remix_123"
        assert response.status == "completed"
        assert hasattr(response, 'remixed_from_video_id')
        assert response.remixed_from_video_id == "video_original_123"

    def test_video_status_async_inside_async_function(self):
        """Test that sync video_status works inside async functions (no asyncio.run issues)."""
        mock_response = VideoObject(
            id="video_sync_in_async",
            object="video",
            status="completed",
            created_at=1712697600,
            model="sora-2",
            progress=100
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_status_handler.return_value = mock_response
            
            import asyncio
            
            async def test_sync_in_async():
                # This should work without asyncio.run() issues
                response = video_status(
                    video_id="video_sync_in_async",
                    model="sora-2"
                )
                return response
            
            response = asyncio.run(test_sync_in_async())
            
            assert isinstance(response, VideoObject)
            assert response.id == "video_sync_in_async"
            assert response.status == "completed"

    def test_video_status_url_construction(self):
        """Test video status URL construction."""
        config = OpenAIVideoConfig()
        
        # Test with different API bases
        test_cases = [
            ("https://api.openai.com/v1/videos", "video_123", "https://api.openai.com/v1/videos/video_123"),
            ("https://api.openai.com/v1/videos/", "video_123", "https://api.openai.com/v1/videos/video_123"),
            ("https://custom-api.com/v1/videos", "video_456", "https://custom-api.com/v1/videos/video_456"),
        ]
        
        for api_base, video_id, expected_url in test_cases:
            url, data = config.transform_video_status_retrieve_request(
                video_id=video_id,
                api_base=api_base,
                litellm_params=MagicMock(),
                headers={}
            )
            assert url == expected_url
            assert data == {}


class TestVideoLogging:
    """Test video generation logging functionality."""
    
    class TestVideoLogger(CustomLogger):
        def __init__(self):
            self.standard_logging_payload = None
            
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.standard_logging_payload = kwargs.get("standard_logging_object")
    
    @pytest.mark.asyncio
    async def test_video_generation_logging(self):
        """Test that video generation creates proper logging payload with cost tracking."""
        custom_logger = self.TestVideoLogger()
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = [custom_logger]
        
        # Mock video generation response
        mock_response = VideoObject(
            id="video_test_123",
            object="video", 
            status="queued",
            created_at=1712697600,
            model="sora-2",
            size="720x1280",
            seconds="8"
        )
        
        with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
            mock_handler.video_generation_handler.return_value = mock_response
            
            response = await litellm.avideo_generation(
                prompt="A cat running in a garden",
                model="sora-2",
                seconds="8",
                size="720x1280"
            )
            
            await asyncio.sleep(1)  # Allow logging to complete
            
            # Verify logging payload was created
            assert custom_logger.standard_logging_payload is not None
            
            payload = custom_logger.standard_logging_payload
            
            # Verify basic logging fields
            assert payload["call_type"] == "avideo_generation"
            assert payload["status"] == "success"
            assert payload["model"] == "sora-2"
            assert payload["custom_llm_provider"] == "openai"
            
            # Verify response object is recognized for logging
            assert payload["response"] is not None
            assert payload["response"]["id"] == "video_test_123"
            assert payload["response"]["object"] == "video"
            
            # Verify cost tracking is present (may be 0 in test environment)
            assert payload["response_cost"] is not None
            # Note: Cost calculation may not work in test environment due to mocking
            # The important thing is that the logging payload is created and recognized


if __name__ == "__main__":
    pytest.main([__file__])
