#!/usr/bin/env python3
"""
Test to verify the Google GenAI proxy API endpoints
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm


def test_google_generate_content_endpoint():
    """Test that the google_generate_content endpoint correctly routes requests"""
    # Skip this test if we can't import the required modules due to missing dependencies
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a FastAPI app and include the router (required for FastAPI 0.120+)
    app = FastAPI()
    app.include_router(google_router)
    
    # Create a test client
    client = TestClient(app)
    
    # Mock the router's agenerate_content method
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.agenerate_content = AsyncMock(return_value={"test": "response"})
        
        # Send a request to the endpoint
        response = client.post(
            "/v1beta/models/test-model:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        assert response.json() == {"test": "response"}
        
        # Verify that agenerate_content was called
        mock_router.agenerate_content.assert_called_once()


def test_google_stream_generate_content_endpoint():
    """Test that the google_stream_generate_content endpoint correctly routes streaming requests"""
    # Skip this test if we can't import the required modules due to missing dependencies
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a FastAPI app and include the router (required for FastAPI 0.120+)
    app = FastAPI()
    app.include_router(google_router)
    
    # Create a test client
    client = TestClient(app)
    
    # Mock the router's agenerate_content_stream method to return a stream
    async def mock_stream_generator():
        yield 'data: {"test": "stream_chunk_1"}\n\n'
        yield 'data: {"test": "stream_chunk_2"}\n\n'
        yield "data: [DONE]\n\n"
    
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.agenerate_content_stream = AsyncMock(return_value=mock_stream_generator())
        
        # Send a request to the endpoint
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify that agenerate_content_stream was called with correct parameters
        mock_router.agenerate_content_stream.assert_called_once()
        call_args = mock_router.agenerate_content_stream.call_args
        assert call_args[1]["stream"] is True
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_google_generate_content_with_cost_tracking_metadata():
    """Test that the google_generate_content endpoint includes user metadata for cost tracking"""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a FastAPI app and include the router (required for FastAPI 0.120+)
    app = FastAPI()
    app.include_router(google_router)
    
    # Create a test client
    client = TestClient(app)
    
    # Mock all required proxy server dependencies
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router, \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.proxy_config") as mock_proxy_config, \
         patch("litellm.proxy.proxy_server.version", "1.0.0"), \
         patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request") as mock_add_data:
        
        mock_router.agenerate_content = AsyncMock(return_value={"test": "response"})
        
        # Mock add_litellm_data_to_request to return data with metadata
        async def mock_add_litellm_data(data, request, user_api_key_dict, proxy_config, general_settings, version):
            # Simulate adding user metadata
            data["litellm_metadata"] = {
                "user_api_key_user_id": "test-user-id",
                "user_api_key_team_id": "test-team-id",
                "user_api_key": "hashed-key",
            }
            return data
        
        mock_add_data.side_effect = mock_add_litellm_data
        
        # Send a request to the endpoint
        response = client.post(
            "/v1beta/models/test-model:generateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify that add_litellm_data_to_request was called
        mock_add_data.assert_called_once()
        
        # Verify that agenerate_content was called with metadata
        mock_router.agenerate_content.assert_called_once()
        call_args = mock_router.agenerate_content.call_args
        called_data = call_args[1]
        
        # Verify that litellm_metadata exists and contains user information
        assert "litellm_metadata" in called_data
        assert called_data["litellm_metadata"]["user_api_key_user_id"] == "test-user-id"
        assert called_data["litellm_metadata"]["user_api_key_team_id"] == "test-team-id"


def test_google_stream_generate_content_with_cost_tracking_metadata():
    """Test that the google_stream_generate_content endpoint includes user metadata for cost tracking"""
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a FastAPI app and include the router (required for FastAPI 0.120+)
    app = FastAPI()
    app.include_router(google_router)
    
    # Create a test client
    client = TestClient(app)
    
    # Mock the router's agenerate_content_stream method to return a stream
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = lambda self: mock_stream
    mock_stream.__anext__.side_effect = StopAsyncIteration
    
    # Mock all required proxy server dependencies
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router, \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.proxy_config") as mock_proxy_config, \
         patch("litellm.proxy.proxy_server.version", "1.0.0"), \
         patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request") as mock_add_data:
        
        mock_router.agenerate_content_stream = AsyncMock(return_value=mock_stream)
        
        # Mock add_litellm_data_to_request to return data with metadata
        async def mock_add_litellm_data(data, request, user_api_key_dict, proxy_config, general_settings, version):
            # Simulate adding user metadata
            data["litellm_metadata"] = {
                "user_api_key_user_id": "test-user-id",
                "user_api_key_team_id": "test-team-id",
                "user_api_key": "hashed-key",
            }
            return data
        
        mock_add_data.side_effect = mock_add_litellm_data
        
        # Send a request to the endpoint
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify that add_litellm_data_to_request was called
        mock_add_data.assert_called_once()
        
        # Verify that agenerate_content_stream was called with metadata
        mock_router.agenerate_content_stream.assert_called_once()
        call_args = mock_router.agenerate_content_stream.call_args
        called_data = call_args[1]
        
        # Verify that litellm_metadata exists and contains user information
        assert "litellm_metadata" in called_data
        assert called_data["litellm_metadata"]["user_api_key_user_id"] == "test-user-id"
        assert called_data["litellm_metadata"]["user_api_key_team_id"] == "test-team-id"
        # Verify stream is set to True
        assert called_data["stream"] is True


def test_google_generate_content_with_system_instruction():
    """
    Test that systemInstruction is correctly passed through from the endpoint to the router.
    
    This test verifies the fix for systemInstruction being dropped when forwarding
    requests to Vertex AI through the Google GenAI endpoint.
    """
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a FastAPI app and include the router
    app = FastAPI()
    app.include_router(google_router)
    
    # Create a test client
    client = TestClient(app)
    
    # Mock all required proxy server dependencies
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router, \
         patch("litellm.proxy.proxy_server.general_settings", {}), \
         patch("litellm.proxy.proxy_server.proxy_config") as mock_proxy_config, \
         patch("litellm.proxy.proxy_server.version", "1.0.0"), \
         patch("litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request") as mock_add_data:
        
        mock_router.agenerate_content = AsyncMock(return_value={"test": "response"})
        
        # Mock add_litellm_data_to_request to pass through data unchanged
        async def mock_add_litellm_data(data, request, user_api_key_dict, proxy_config, general_settings, version):
            return data
        
        mock_add_data.side_effect = mock_add_litellm_data
        
        # Define the systemInstruction to test
        system_instruction = {
            "parts": [{"text": "Your name is Doodle."}]
        }
        
        # Send a request with systemInstruction
        response = client.post(
            "/v1beta/models/gemini-2.5-pro:generateContent",
            json={
                "systemInstruction": system_instruction,
                "contents": [
                    {
                        "parts": [{"text": "What is your name?"}],
                        "role": "user"
                    }
                ]
            },
            headers={"Authorization": "Bearer sk-test-key"}
        )
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify that agenerate_content was called
        mock_router.agenerate_content.assert_called_once()
        call_args = mock_router.agenerate_content.call_args
        called_data = call_args[1]
        
        # Verify that systemInstruction is present in the call arguments
        assert "systemInstruction" in called_data
        assert called_data["systemInstruction"] == system_instruction
        assert called_data["systemInstruction"]["parts"][0]["text"] == "Your name is Doodle."
        
        # Verify contents are also present
        assert "contents" in called_data
        assert len(called_data["contents"]) == 1
        assert called_data["contents"][0]["role"] == "user"