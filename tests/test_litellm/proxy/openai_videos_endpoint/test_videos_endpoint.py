import json
import os
import sys
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.proxy._types import LiteLLM_UserTableFiltered, UserAPIKeyAuth
from litellm.proxy.hooks import get_proxy_hook
from litellm.proxy.management_endpoints.internal_user_endpoints import ui_view_users
from litellm.proxy.proxy_server import app

client = TestClient(app)
from litellm.caching.caching import DualCache
from litellm.proxy.proxy_server import hash_token
from litellm.proxy.utils import ProxyLogging


@pytest.fixture
def llm_router() -> Router:
    llm_router = Router(
        model_list=[
            {
                "model_name": "sora-2",
                "litellm_params": {
                    "model": "openai/sora-2",
                    "api_key": "openai_api_key",
                },
                "model_info": {
                    "id": "sora-2-id",
                },
            },
        ]
    )
    return llm_router


def test_video_endpoints_exist():
    """
    Test that video endpoints are properly registered
    """
    # Test that the endpoints exist by checking the app routes
    routes = [route.path for route in app.routes]
    
    # Check for video endpoints (they might be under different paths)
    video_routes = [route for route in routes if "videos" in route]
    assert len(video_routes) > 0, f"No video routes found. Available routes: {routes}"
    
    # Check for specific video endpoint patterns
    assert any("videos" in route for route in routes), f"No video routes found in: {routes}"


def test_video_create_endpoint_structure(monkeypatch, llm_router: Router):
    """
    Test video creation endpoint structure and basic functionality
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock acreate_video as an async function
    with patch("litellm.acreate_video", new=AsyncMock()) as mock_create_video:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video creation response
        from litellm.types.llms.openai import OpenAIVideoObject
        mock_video_response = OpenAIVideoObject(
            id="test-video-id",
            object="video",
            status="processing",
            created_at=1234567890,
            model="sora-2",
            seconds="4",
            size="720x1280",
            usage={
                "video_duration_seconds": 4,
                "model": "sora-2",
                "size": "720x1280"
            }
        )
        mock_create_video.return_value = mock_video_response

        # Test video creation
        response = client.post(
            "/v1/videos",
            data={
                "prompt": "A cat playing with yarn",
                "model": "sora-2",
                "seconds": "4",
                "size": "720x1280",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["id"] == "test-video-id"
        assert response_data["object"] == "video"
        assert response_data["status"] == "processing"

        # Verify acreate_video was called with correct parameters
        mock_create_video.assert_called_once()
        call_args = mock_create_video.call_args
        assert call_args.kwargs["prompt"] == "A cat playing with yarn"
        assert call_args.kwargs["model"] == "sora-2"
        assert call_args.kwargs["seconds"] == "4"
        assert call_args.kwargs["size"] == "720x1280"


def test_video_retrieve_endpoint_structure(monkeypatch, llm_router: Router):
    """
    Test video retrieval endpoint structure
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock avideo_retrieve as an async function
    with patch("litellm.avideo_retrieve", new=AsyncMock()) as mock_retrieve_video:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video retrieval response
        from litellm.types.llms.openai import OpenAIVideoObject
        mock_video_response = OpenAIVideoObject(
            id="video_test-video-id",
            object="video",
            status="completed",
            created_at=1234567890,
            model="sora-2",
            seconds="4",
            size="720x1280",
            usage={
                "video_duration_seconds": 4,
                "model": "sora-2",
                "size": "720x1280"
            }
        )
        mock_retrieve_video.return_value = mock_video_response

        # Test video retrieval
        response = client.get(
            "/v1/videos/video_test-video-id",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["id"] == "video_test-video-id"
        assert response_data["object"] == "video"
        assert response_data["status"] == "completed"

        # Verify avideo_retrieve was called with correct parameters
        mock_retrieve_video.assert_called_once()
        call_args = mock_retrieve_video.call_args
        assert call_args.kwargs["video_id"] == "video_test-video-id"


def test_video_list_endpoint_structure(monkeypatch, llm_router: Router):
    """
    Test video listing endpoint structure
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock avideo_list as an async function
    with patch("litellm.avideo_list", new=AsyncMock()) as mock_list_videos:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video list response
        from litellm.types.llms.openai import OpenAIVideoObject
        mock_video1 = OpenAIVideoObject(
            id="video-1",
            object="video",
            status="completed",
            created_at=1234567890,
            model="sora-2",
            seconds="4",
            size="720x1280"
        )
        mock_video2 = OpenAIVideoObject(
            id="video-2",
            object="video",
            status="processing",
            created_at=1234567891,
            model="sora-2-pro",
            seconds="8",
            size="1280x720"
        )
        
        # Mock the video list response - make it JSON serializable
        class MockListResponse:
            def __init__(self, data, object_type):
                self.data = data
                self.object = object_type
            
            def dict(self):
                return {
                    'data': [item.dict() if hasattr(item, 'dict') else item for item in self.data],
                    'object': self.object
                }
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def keys(self):
                return ['data', 'object']
        
        mock_list_response = MockListResponse([mock_video1, mock_video2], 'list')
        mock_list_videos.return_value = mock_list_response

        # Test video listing
        response = client.get(
            "/v1/videos",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        response_data = response.json()
        
        # The response should contain the mock data
        assert "object" in response_data or "data" in response_data
        if "object" in response_data:
            assert response_data["object"] == "list"
        if "data" in response_data:
            assert len(response_data["data"]) == 2
            assert response_data["data"][0]["id"] == "video-1"
            assert response_data["data"][1]["id"] == "video-2"

        # Verify avideo_list was called
        mock_list_videos.assert_called_once()


def test_video_delete_endpoint_structure(monkeypatch, llm_router: Router):
    """
    Test video deletion endpoint structure
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock avideo_delete as an async function
    with patch("litellm.avideo_delete", new=AsyncMock()) as mock_delete_video:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video deletion response - make it JSON serializable
        class MockDeleteResponse:
            def __init__(self):
                self.id = 'video_test-video-id'
                self.object = 'video'
                self.deleted = True
            
            def dict(self):
                return {
                    'id': self.id,
                    'object': self.object,
                    'deleted': self.deleted
                }
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def keys(self):
                return ['id', 'object', 'deleted']
        
        mock_delete_response = MockDeleteResponse()
        mock_delete_video.return_value = mock_delete_response

        # Test video deletion
        response = client.delete(
            "/v1/videos/video_test-video-id",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        response_data = response.json()
        
        # The response should contain the mock data
        assert "id" in response_data or "object" in response_data
        if "id" in response_data:
            assert response_data["id"] == "video_test-video-id"
        if "deleted" in response_data:
            assert response_data["deleted"] is True

        # Verify avideo_delete was called with correct parameters
        mock_delete_video.assert_called_once()
        call_args = mock_delete_video.call_args
        assert call_args.kwargs["video_id"] == "video_test-video-id"


def test_video_content_endpoint_structure(monkeypatch, llm_router: Router):
    """
    Test video content retrieval endpoint structure
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock avideo_content as an async function
    with patch("litellm.avideo_content", new=AsyncMock()) as mock_content_video:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video content response (needs to have a response attribute)
        mock_content_response = type('MockContentResponse', (), {
            'response': type('MockHttpxResponse', (), {
                'content': b"fake video content data",
                'status_code': 200,
                'headers': {'content-type': 'video/mp4'}
            })()
        })()
        mock_content_video.return_value = mock_content_response

        # Test video content retrieval
        response = client.get(
            "/v1/videos/video_test-video-id/content",
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        assert response.content == b"fake video content data"

        # Verify avideo_content was called with correct parameters
        mock_content_video.assert_called_once()
        call_args = mock_content_video.call_args
        assert call_args.kwargs["video_id"] == "video_test-video-id"


def test_video_cost_calculation_integration(monkeypatch, llm_router: Router):
    """
    Test that video cost calculation is integrated properly
    """
    # Mock the videos configuration
    mock_videos_config = [{"custom_llm_provider": "openai", "api_key": "test-key"}]
    monkeypatch.setattr("litellm.proxy.openai_videos_endpoints.video_endpoints.videos_config", mock_videos_config)

    # Mock acreate_video as an async function
    with patch("litellm.acreate_video", new=AsyncMock()) as mock_create_video:
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=DualCache(default_in_memory_ttl=1)
        )
        proxy_logging_obj._add_proxy_hooks(llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj)

        # Mock the video creation response with usage information
        from litellm.types.llms.openai import OpenAIVideoObject
        mock_video_response = OpenAIVideoObject(
            id="test-video-id",
            object="video",
            status="completed",
            created_at=1234567890,
            model="sora-2",
            seconds="4",
            size="720x1280",
            usage={
                "video_duration_seconds": 4,
                "model": "sora-2",
                "size": "720x1280"
            }
        )
        mock_create_video.return_value = mock_video_response

        # Test video creation
        response = client.post(
            "/v1/videos",
            data={
                "prompt": "A cat playing with yarn",
                "model": "sora-2",
                "seconds": "4",
                "size": "720x1280",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200
        response_data = response.json()
        
        # Verify usage information is present for cost calculation
        assert "usage" in response_data
        usage = response_data["usage"]
        assert "video_duration_seconds" in usage
        assert usage["video_duration_seconds"] == 4
        assert usage["model"] == "sora-2"
        assert usage["size"] == "720x1280"

        # Verify acreate_video was called
        mock_create_video.assert_called_once()
