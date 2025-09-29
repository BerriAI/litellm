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
        from fastapi.testclient import TestClient
        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a test client
    client = TestClient(google_router)
    
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
        from fastapi.testclient import TestClient
        from litellm.proxy.google_endpoints.endpoints import router as google_router
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    
    # Create a test client
    client = TestClient(google_router)
    
    # Mock the router's agenerate_content method to return a stream
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = lambda self: mock_stream
    mock_stream.__anext__.side_effect = StopAsyncIteration
    
    with patch("litellm.proxy.proxy_server.llm_router") as mock_router:
        mock_router.agenerate_content = AsyncMock(return_value=mock_stream)
        
        # Send a request to the endpoint
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify that agenerate_content was called with correct parameters
        mock_router.agenerate_content.assert_called_once()
        call_args = mock_router.agenerate_content.call_args
        assert call_args[1]["stream"] is True
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]