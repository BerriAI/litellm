import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.cost_calculator import default_video_cost_calculator
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.videos.main import VideoObject, VideoResponse
from litellm.videos import main as videos_main
from litellm.videos.main import (
    avideo_generation,
    avideo_status,
    video_generation,
    video_status,
)


class TestVideoGeneration:
    """Test suite for video generation functionality."""

    def test_video_generation_basic(self):
        """Test basic video generation functionality."""
        # Use mock_response parameter for reliable testing
        response = video_generation(
            prompt="Show them running around the room",
            model="sora-2",
            seconds="8",
            size="720x1280",
            mock_response={
                "id": "video_123",
                "object": "video",
                "status": "queued",
                "created_at": 1712697600,
                "model": "sora-2",
                "size": "720x1280",
                "seconds": "8"
            }
        )
        
        assert isinstance(response, VideoObject)
        assert response.id == "video_123"
        assert response.status == "queued"
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
        
        # Mock the async_video_generation_handler to return the mock_response
        async_mock = AsyncMock(return_value=mock_response)
        with patch.object(videos_main.base_llm_http_handler, 'async_video_generation_handler', async_mock):
            with patch.object(videos_main.base_llm_http_handler, 'video_generation_handler', side_effect=lambda **kwargs: async_mock(**kwargs)):
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
        with patch.object(videos_main.base_llm_http_handler, 'video_generation_handler', side_effect=Exception("API Error")):
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
        data, files, returned_api_base = config.transform_video_create_request(
            model="sora-2",
            prompt="Test video prompt",
            api_base="https://api.openai.com/v1/videos",
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
        assert returned_api_base == "https://api.openai.com/v1/videos"

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
        import json
        import os

        # Try to load the local model cost map, skip if not found
        cost_map_path = "model_prices_and_context_window.json"
        if not os.path.exists(cost_map_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", cost_map_path),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", cost_map_path),
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    cost_map_path = path
                    break
            else:
                pytest.skip("model_prices_and_context_window.json not found")
        
        with open(cost_map_path, "r") as f:
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
        
        data, files, returned_api_base = config.transform_video_create_request(
            model="sora-2",
            prompt="Test video with image",
            api_base="https://api.openai.com/v1/videos",
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

    def test_video_generation_uses_api_key_from_litellm_params(self):
        """Test that video generation handler uses api_key from litellm_params when function parameter is None."""
        handler = BaseLLMHTTPHandler()
        config = OpenAIVideoConfig()
        
        # Mock the validate_environment method to capture the api_key passed to it
        with patch.object(config, 'validate_environment') as mock_validate:
            mock_validate.return_value = {"Authorization": "Bearer deployment-api-key"}
            
            # Mock the transform and HTTP client
            with patch.object(config, 'transform_video_create_request') as mock_transform:
                mock_transform.return_value = ({"model": "sora-2", "prompt": "test"}, [], "https://api.openai.com/v1/videos")
                
                # Mock the transform_video_create_response to avoid needing a real response
                with patch.object(config, 'transform_video_create_response') as mock_transform_response:
                    mock_video_object = MagicMock()
                    mock_video_object.id = "video_123"
                    mock_video_object.object = "video"
                    mock_video_object.status = "queued"
                    mock_transform_response.return_value = mock_video_object
                    
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "id": "video_123",
                        "object": "video",
                        "status": "queued",
                        "created_at": 1712697600,
                        "model": "sora-2"
                    }
                    mock_response.status_code = 200
                    
                    mock_client = MagicMock()
                    mock_client.post.return_value = mock_response
                    
                    with patch(
                        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
                        return_value=mock_client,
                    ):
                        result = handler.video_generation_handler(
                            model="sora-2",
                            prompt="test prompt",
                            video_generation_provider_config=config,
                            video_generation_optional_request_params={},
                            custom_llm_provider="openai",
                            litellm_params={"api_key": "deployment-api-key", "api_base": "https://api.openai.com/v1"},
                            logging_obj=MagicMock(),
                            timeout=5.0,
                            api_key=None,  # Function parameter is None
                            _is_async=False,
                        )
                    
                    # Verify validate_environment was called with api_key from litellm_params
                    mock_validate.assert_called_once()
                    call_args = mock_validate.call_args
                assert call_args.kwargs["api_key"] == "deployment-api-key"

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
        """Test video generation with provider-specific parameters via extra_body."""
        from litellm.videos.utils import VideoGenerationRequestUtils

        # Test that provider-specific parameters can be passed via extra_body
        # This allows support for Vertex AI and Gemini specific parameters
        result = VideoGenerationRequestUtils.get_optional_params_video_generation(
            model="sora-2",
            video_generation_provider_config=OpenAIVideoConfig(),
            video_generation_optional_params={
                "seconds": "8",
                "extra_body": {
                    "vertex_ai_param": "value",
                    "gemini_param": "value2"
                }
            }
        )
        
        # extra_body params should be merged into the result
        assert result["seconds"] == "8"
        assert result["vertex_ai_param"] == "value"
        assert result["gemini_param"] == "value2"
        # extra_body itself should be removed from the result
        assert "extra_body" not in result

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
        # Use mock_response parameter for reliable testing
        response = video_status(
            video_id="video_123",
            model="sora-2",
            mock_response={
                "id": "video_123",
                "object": "video",
                "status": "completed",
                "created_at": 1712697600,
                "completed_at": 1712697660,
                "model": "sora-2",
                "progress": 100,
                "size": "720x1280",
                "seconds": "8"
            }
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
        
        # Mock the async_video_status_handler to return the mock_response
        async_mock = AsyncMock(return_value=mock_response)
        with patch.object(videos_main.base_llm_http_handler, 'async_video_status_handler', async_mock):
            with patch.object(videos_main.base_llm_http_handler, 'video_status_handler', side_effect=lambda **kwargs: async_mock(**kwargs)):
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
        with patch.object(videos_main.base_llm_http_handler, 'video_status_handler', side_effect=Exception("API Error")):
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
        import asyncio
        
        async def test_sync_in_async():
            # This should work without asyncio.run() issues
            # Use mock_response parameter for reliable testing
            response = video_status(
                video_id="video_sync_in_async",
                model="sora-2",
                mock_response={
                    "id": "video_sync_in_async",
                    "object": "video",
                    "status": "completed",
                    "created_at": 1712697600,
                    "model": "sora-2",
                    "progress": 100
                }
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
        """Test that video generation creates proper logging payload with cost tracking.

        Note: Uses AsyncMock with side_effect pattern for reliable parallel execution.
        """
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

        # Create async mock function to return the mock_response
        async def mock_async_handler(*args, **kwargs):
            return mock_response

        # Patch the async_video_generation_handler method on base_llm_http_handler
        with patch.object(videos_main.base_llm_http_handler, 'async_video_generation_handler', side_effect=mock_async_handler):
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


def test_openai_transform_video_content_request_empty_params():
    """OpenAI content transform should return empty params to ensure GET is used."""
    config = OpenAIVideoConfig()
    url, params = config.transform_video_content_request(
        video_id="video_123",
        api_base="https://api.openai.com/v1/videos",
        litellm_params={},
        headers={},
    )

    assert url == "https://api.openai.com/v1/videos/video_123/content"
    assert params == {}


def test_video_content_handler_uses_get_for_openai():
    """HTTP handler must use GET (not POST) for OpenAI content download."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    # Clear the HTTP client cache to prevent test isolation issues
    # In CI, a cached real HTTPHandler from a previous test might bypass the mock
    if hasattr(litellm, 'in_memory_llm_clients_cache'):
        litellm.in_memory_llm_clients_cache.flush_cache()

    handler = BaseLLMHTTPHandler()
    config = OpenAIVideoConfig()

    # Use spec=HTTPHandler so isinstance(mock_client, HTTPHandler) returns True,
    # ensuring the handler uses our mock directly instead of creating a new client.
    mock_client = MagicMock(spec=HTTPHandler)
    mock_response = MagicMock()
    mock_response.content = b"mp4-bytes"
    mock_client.get.return_value = mock_response

    # Patch _get_httpx_client to ensure no real HTTP client is created
    # This prevents test isolation issues where isinstance check might fail
    with patch('litellm.llms.custom_httpx.llm_http_handler._get_httpx_client') as mock_get_client:
        mock_get_client.return_value = mock_client

        result = handler.video_content_handler(
            video_id="video_abc",
            video_content_provider_config=config,
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(api_base="https://api.openai.com/v1"),
            logging_obj=MagicMock(),
            timeout=5.0,
            api_key="sk-test",
            client=mock_client,
            _is_async=False,
        )

    assert result == b"mp4-bytes"
    mock_client.get.assert_called_once()
    assert not mock_client.post.called
    called_url = mock_client.get.call_args.kwargs["url"]
    assert called_url == "https://api.openai.com/v1/videos/video_abc/content"


def test_video_content_respects_api_base_and_api_key_from_kwargs():
    """Test that video_content respects api_base and api_key from kwargs (simulating database entry)."""
    from litellm.videos.main import video_content

    # Mock the handler to capture litellm_params
    captured_litellm_params = None
    
    def capture_litellm_params(*args, **kwargs):
        nonlocal captured_litellm_params
        captured_litellm_params = kwargs.get("litellm_params")
        return b"mp4-bytes"
    
    with patch('litellm.videos.main.base_llm_http_handler') as mock_handler:
        mock_handler.video_content_handler = capture_litellm_params
        
        # Call video_content with api_base and api_key in kwargs (simulating database entry)
        # This simulates how the router passes model config from database via **kwargs
        result = video_content(
            video_id="video_test_123",
            custom_llm_provider="azure",
            api_base="https://test-resource.openai.azure.com/",  # Passed via kwargs by router
            api_key="test-api-key-from-db",  # Passed via kwargs by router
        )
    
    # Verify that api_base and api_key from kwargs were included in litellm_params
    assert captured_litellm_params is not None
    assert captured_litellm_params.get("api_base") == "https://test-resource.openai.azure.com/"
    assert captured_litellm_params.get("api_key") == "test-api-key-from-db"
    assert result == b"mp4-bytes"


def test_openai_video_config_has_async_transform():
    """Ensure OpenAIVideoConfig exposes async_transform_video_content_response at runtime."""
    cfg = OpenAIVideoConfig()
    assert callable(getattr(cfg, "async_transform_video_content_response", None))


def test_gemini_video_config_has_async_transform():
    """Ensure GeminiVideoConfig exposes async_transform_video_content_response at runtime."""
    cfg = GeminiVideoConfig()
    assert callable(getattr(cfg, "async_transform_video_content_response", None))


def test_encode_video_id_with_provider_handles_azure_video_prefix():
    """
    Test that encode_video_id_with_provider correctly encodes Azure/OpenAI video IDs
    that start with 'video_' prefix.
    
    This test verifies the fix for the issue where Azure returns video IDs like
    'video_69323201cf6081909263f751f89991e6', which were previously skipped
    from encoding, causing video status retrieval to default to 'openai' provider.
    """
    from litellm.types.videos.utils import (
        decode_video_id_with_provider,
        encode_video_id_with_provider,
    )

    # Test case: Azure returns a video ID starting with 'video_'
    raw_azure_video_id = "video_69323201cf6081909263f751f89991e6"
    provider = "azure"
    model_id = "azure/sora-2"
    
    # Encode the video ID with provider information
    encoded_id = encode_video_id_with_provider(
        video_id=raw_azure_video_id,
        provider=provider,
        model_id=model_id
    )
    
    # Verify the ID was encoded (should be different from the original)
    assert encoded_id != raw_azure_video_id
    assert encoded_id.startswith("video_")
    
    # Decode the encoded ID to verify provider information is preserved
    decoded = decode_video_id_with_provider(encoded_id)
    assert decoded.get("custom_llm_provider") == provider
    assert decoded.get("model_id") == model_id
    assert decoded.get("video_id") == raw_azure_video_id
    
    # Verify that encoding an already-encoded ID doesn't double-encode it
    encoded_twice = encode_video_id_with_provider(
        video_id=encoded_id,
        provider=provider,
        model_id=model_id
    )
    assert encoded_twice == encoded_id  # Should return the same encoded ID
    
class TestVideoListTransformation:
    """Tests for video list request/response transformation with provider ID encoding."""

    def test_transform_video_list_response_encodes_first_id_and_last_id(self):
        """Verify that first_id and last_id are encoded with provider metadata."""
        config = OpenAIVideoConfig()

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "video_aaa",
                    "object": "video",
                    "model": "sora-2",
                    "status": "completed",
                },
                {
                    "id": "video_bbb",
                    "object": "video",
                    "model": "sora-2",
                    "status": "completed",
                },
            ],
            "first_id": "video_aaa",
            "last_id": "video_bbb",
            "has_more": False,
        }

        result = config.transform_video_list_response(
            raw_response=mock_http_response,
            logging_obj=MagicMock(),
            custom_llm_provider="azure",
        )

        from litellm.types.videos.utils import decode_video_id_with_provider

        # data[].id should be encoded
        for item in result["data"]:
            decoded = decode_video_id_with_provider(item["id"])
            assert decoded["custom_llm_provider"] == "azure"

        # first_id and last_id should also be encoded
        first_decoded = decode_video_id_with_provider(result["first_id"])
        assert first_decoded["custom_llm_provider"] == "azure"
        assert first_decoded["video_id"] == "video_aaa"
        assert first_decoded["model_id"] == "sora-2"

        last_decoded = decode_video_id_with_provider(result["last_id"])
        assert last_decoded["custom_llm_provider"] == "azure"
        assert last_decoded["video_id"] == "video_bbb"
        assert last_decoded["model_id"] == "sora-2"

    def test_transform_video_list_response_no_provider_leaves_ids_unchanged(self):
        """When custom_llm_provider is None, all IDs should remain unchanged."""
        config = OpenAIVideoConfig()

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "object": "list",
            "data": [
                {"id": "video_aaa", "object": "video", "model": "sora-2", "status": "completed"},
            ],
            "first_id": "video_aaa",
            "last_id": "video_aaa",
            "has_more": False,
        }

        result = config.transform_video_list_response(
            raw_response=mock_http_response,
            logging_obj=MagicMock(),
            custom_llm_provider=None,
        )

        assert result["data"][0]["id"] == "video_aaa"
        assert result["first_id"] == "video_aaa"
        assert result["last_id"] == "video_aaa"

    def test_transform_video_list_response_missing_pagination_fields(self):
        """first_id / last_id may be absent or null; should not raise."""
        config = OpenAIVideoConfig()

        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "object": "list",
            "data": [
                {"id": "video_aaa", "object": "video", "model": "sora-2", "status": "completed"},
            ],
            "has_more": False,
        }

        result = config.transform_video_list_response(
            raw_response=mock_http_response,
            logging_obj=MagicMock(),
            custom_llm_provider="azure",
        )

        # data[].id should still be encoded
        from litellm.types.videos.utils import decode_video_id_with_provider

        decoded = decode_video_id_with_provider(result["data"][0]["id"])
        assert decoded["custom_llm_provider"] == "azure"

        # first_id / last_id should not be present
        assert "first_id" not in result
        assert "last_id" not in result

    def test_transform_video_list_request_decodes_after_parameter(self):
        """Encoded 'after' cursor should be decoded back to the raw provider ID."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        config = OpenAIVideoConfig()

        raw_id = "video_69888baee890819086dd3366bfc372fe"
        encoded_id = encode_video_id_with_provider(raw_id, "azure", "sora-2")

        url, params = config.transform_video_list_request(
            api_base="https://my-resource.openai.azure.com/openai/v1/videos",
            litellm_params=MagicMock(),
            headers={},
            after=encoded_id,
            limit=10,
        )

        assert params["after"] == raw_id
        assert params["limit"] == "10"

    def test_transform_video_list_request_passes_through_plain_after(self):
        """A plain (non-encoded) 'after' value should pass through unchanged."""
        config = OpenAIVideoConfig()

        url, params = config.transform_video_list_request(
            api_base="https://api.openai.com/v1/videos",
            litellm_params=MagicMock(),
            headers={},
            after="video_plain_id",
        )

        assert params["after"] == "video_plain_id"

    def test_transform_video_list_roundtrip(self):
        """first_id from list response should decode correctly when used as after parameter."""
        config = OpenAIVideoConfig()

        # Simulate a list response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "object": "list",
            "data": [
                {"id": "video_aaa", "object": "video", "model": "sora-2", "status": "completed"},
                {"id": "video_bbb", "object": "video", "model": "sora-2", "status": "completed"},
            ],
            "first_id": "video_aaa",
            "last_id": "video_bbb",
            "has_more": True,
        }

        list_result = config.transform_video_list_response(
            raw_response=mock_http_response,
            logging_obj=MagicMock(),
            custom_llm_provider="azure",
        )

        # Use the encoded last_id as the 'after' cursor for the next page
        _, params = config.transform_video_list_request(
            api_base="https://my-resource.openai.azure.com/openai/v1/videos",
            litellm_params=MagicMock(),
            headers={},
            after=list_result["last_id"],
        )

        # The after param sent to the upstream API should be the raw video ID
        assert params["after"] == "video_bbb"


class TestVideoEndpointsProxyLitellmParams:
    """Test that video proxy endpoints (status, content, remix) respect litellm_params from proxy config."""

    @pytest.fixture
    def client_with_vertex_config(self, monkeypatch):
        """Create a test client with a proxy config that includes Vertex AI model with litellm_params."""
        import asyncio
        import tempfile

        import yaml
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.proxy_server import (
            cleanup_router_config_variables,
            initialize,
            router,
        )
        from litellm.proxy.video_endpoints.endpoints import router as video_router

        # Clean up any existing router config
        cleanup_router_config_variables()

        # Create inline config
        config = {
            "model_list": [
                {
                    "model_name": "vertex-ai-sora-2",
                    "litellm_params": {
                        "model": "vertex_ai/veo-2.0-generate-001",
                        "vertex_project": "test-project-123",
                        "vertex_location": "global",
                        "vertex_credentials": "/path/to/test-credentials.json",
                    }
                }
            ]
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_fp = f.name
        
        try:
            # Initialize the proxy with the test config
            app = FastAPI()
            asyncio.run(initialize(config=config_fp, debug=True))
            app.include_router(router)
            app.include_router(video_router)

            return TestClient(app)
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(config_fp):
                os.unlink(config_fp)

    @pytest.fixture
    def mock_video_generation_response(self):
        """Mock video generation response with encoded video_id."""
        from litellm.types.videos.utils import encode_video_id_with_provider

        # Create an encoded video_id that includes provider and model_id
        original_video_id = "projects/test-project-123/locations/global/publishers/google/models/veo-2.0-generate-001/operations/test-operation-123"
        encoded_video_id = encode_video_id_with_provider(
            video_id=original_video_id,
            provider="vertex_ai",
            model_id="veo-2.0-generate-001",
        )

        return VideoObject(
            id=encoded_video_id,
            object="video",
            status="processing",
            created_at=1712697600,
            model="vertex_ai/veo-2.0-generate-001",
        )

    @pytest.fixture
    def mock_video_status_response(self):
        """Mock video status response."""
        return VideoObject(
            id="video_test_123",
            object="video",
            status="completed",
            created_at=1712697600,
            completed_at=1712697660,
            model="vertex_ai/veo-2.0-generate-001",
            progress=100,
        )

    @pytest.fixture
    def mock_video_content_response(self):
        """Mock video content response (raw bytes)."""
        return b"fake_video_content_bytes"

    @pytest.mark.asyncio
    async def test_video_status_respects_litellm_params(
        self, client_with_vertex_config, mock_video_generation_response, mock_video_status_response
    ):
        """Test that video_status endpoint uses litellm_params from proxy config."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create an encoded video_id
        encoded_video_id = mock_video_generation_response.id

        # Mock the router instance
        mock_router_instance = MagicMock()
        mock_router_instance.resolve_model_name_from_model_id.return_value = "vertex-ai-sora-2"
        mock_router_instance.model_names = {"vertex-ai-sora-2"}
        mock_router_instance.has_model_id.return_value = False

        # Mock route_request to capture the data being passed
        # route_request should return a coroutine (not await it), so we return a coroutine
        async def mock_route_request_func(*args, **kwargs):
            return mock_video_status_response
        
        # Create a coroutine that will be added to tasks
        def create_mock_coroutine(*args, **kwargs):
            return mock_route_request_func(*args, **kwargs)

        with patch("litellm.proxy.proxy_server.llm_router", mock_router_instance):
            with patch("litellm.proxy.common_request_processing.route_request", side_effect=create_mock_coroutine) as mock_route_request:
                # Make request to video_status endpoint
                response = client_with_vertex_config.get(
                    f"/v1/videos/{encoded_video_id}",
                    headers={"Authorization": "Bearer sk-1234"},
                )

                # Verify the endpoint was called
                assert response.status_code == 200, f"Response: {response.text}"

                # Verify that route_request was called
                assert mock_route_request.called
                call_args = mock_route_request.call_args
                # route_request is called with data as a keyword argument
                data_passed = call_args.kwargs.get("data", {}) if call_args.kwargs else (call_args.args[0] if call_args.args and len(call_args.args) > 0 else {})

                # Verify that model was resolved and added to data
                assert data_passed.get("model") == "vertex-ai-sora-2", (
                    f"Expected model to be 'vertex-ai-sora-2', got '{data_passed.get('model')}'. "
                    f"Full data: {data_passed}, call_args: {call_args}"
                )
                # Verify that custom_llm_provider is set from decoded video_id
                assert data_passed.get("custom_llm_provider") == "vertex_ai", (
                    f"Expected custom_llm_provider to be 'vertex_ai', got '{data_passed.get('custom_llm_provider')}'. "
                    f"Full data: {data_passed}"
                )

    @pytest.mark.asyncio
    async def test_video_content_respects_litellm_params(
        self, client_with_vertex_config, mock_video_generation_response, mock_video_content_response
    ):
        """Test that video_content endpoint uses litellm_params from proxy config."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create an encoded video_id
        encoded_video_id = mock_video_generation_response.id

        # Mock the router instance
        mock_router_instance = MagicMock()
        mock_router_instance.resolve_model_name_from_model_id.return_value = "vertex-ai-sora-2"
        mock_router_instance.model_names = {"vertex-ai-sora-2"}
        mock_router_instance.has_model_id.return_value = False

        # Mock route_request to capture the data being passed
        # route_request should return a coroutine (not await it), so we return a coroutine
        async def mock_route_request_func(*args, **kwargs):
            return mock_video_content_response
        
        # Create a coroutine that will be added to tasks
        def create_mock_coroutine(*args, **kwargs):
            return mock_route_request_func(*args, **kwargs)

        with patch("litellm.proxy.proxy_server.llm_router", mock_router_instance):
            with patch("litellm.proxy.common_request_processing.route_request", side_effect=create_mock_coroutine) as mock_route_request:
                # Make request to video_content endpoint
                response = client_with_vertex_config.get(
                    f"/v1/videos/{encoded_video_id}/content",
                    headers={"Authorization": "Bearer sk-1234"},
                )

                # Verify the endpoint was called
                assert response.status_code == 200, f"Response: {response.text}"

                # Verify that route_request was called
                assert mock_route_request.called
                call_args = mock_route_request.call_args
                # route_request is called with data as a keyword argument
                data_passed = call_args.kwargs.get("data", {}) if call_args.kwargs else (call_args.args[0] if call_args.args and len(call_args.args) > 0 else {})

                # Verify that model was resolved and added to data
                assert data_passed.get("model") == "vertex-ai-sora-2", (
                    f"Expected model to be 'vertex-ai-sora-2', got '{data_passed.get('model')}'. "
                    f"Full data: {data_passed}, call_args: {call_args}"
                )
                # Verify that custom_llm_provider is correctly set from decoded video_id (not "openai")
                assert data_passed.get("custom_llm_provider") == "vertex_ai", (
                    f"Expected custom_llm_provider to be 'vertex_ai', got '{data_passed.get('custom_llm_provider')}'. "
                    f"Full data: {data_passed}"
                )

    @pytest.mark.asyncio
    async def test_video_content_preserves_custom_llm_provider_from_decoded_id(
        self, client_with_vertex_config, mock_video_generation_response, mock_video_content_response
    ):
        """Test that video_content preserves custom_llm_provider from decoded video_id."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create an encoded video_id
        encoded_video_id = mock_video_generation_response.id

        # Mock the router instance
        mock_router_instance = MagicMock()
        mock_router_instance.resolve_model_name_from_model_id.return_value = "vertex-ai-sora-2"
        mock_router_instance.model_names = {"vertex-ai-sora-2"}
        mock_router_instance.has_model_id.return_value = False

        # Mock route_request to capture the data being passed
        # route_request should return a coroutine (not await it), so we return a coroutine
        async def mock_route_request_func(*args, **kwargs):
            return mock_video_content_response
        
        # Create a coroutine that will be added to tasks
        def create_mock_coroutine(*args, **kwargs):
            return mock_route_request_func(*args, **kwargs)

        with patch("litellm.proxy.proxy_server.llm_router", mock_router_instance):
            with patch("litellm.proxy.common_request_processing.route_request", side_effect=create_mock_coroutine) as mock_route_request:
                # Make request to video_content endpoint
                response = client_with_vertex_config.get(
                    f"/v1/videos/{encoded_video_id}/content",
                    headers={"Authorization": "Bearer sk-1234"},
                )

                # Verify the endpoint was called
                assert response.status_code == 200, f"Response: {response.text}"

                # Verify that route_request was called
                assert mock_route_request.called
                call_args = mock_route_request.call_args
                # route_request is called with data as a keyword argument
                data_passed = call_args.kwargs.get("data", {}) if call_args.kwargs else (call_args.args[0] if call_args.args and len(call_args.args) > 0 else {})

                # Most importantly: verify that custom_llm_provider is "vertex_ai" not "openai"
                # This was the bug we fixed - it was defaulting to "openai" before
                assert data_passed.get("custom_llm_provider") == "vertex_ai", (
                    f"Expected custom_llm_provider to be 'vertex_ai', "
                    f"but got '{data_passed.get('custom_llm_provider')}'. "
                    f"Full data: {data_passed}, call_args: {call_args}"
                )


if __name__ == "__main__":
    pytest.main([__file__])
