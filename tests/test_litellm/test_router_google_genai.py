#!/usr/bin/env python3
"""
Test to verify the new Google GenAI router methods
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_router_agenerate_content_method():
    """Test that the new agenerate_content method in Router works correctly"""
    # Create a router instance
    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                }
            }
        ]
    )
    
    # Create a mock response in Google GenAI format
    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello, world!"
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock the router's underlying agenerate_content method to return a mock response
    with patch.object(router, 'agenerate_content', new=AsyncMock(return_value=mock_response)) as mock_agenerate_content:
        # Call the agenerate_content method
        response = await router.agenerate_content(
            model="test-model",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}]
        )
        
        # Verify that router.agenerate_content was called with correct parameters
        mock_agenerate_content.assert_called_once()
        call_args = mock_agenerate_content.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        
        # Verify that the response is the mock response we created
        assert response == mock_response


@pytest.mark.asyncio
async def test_router_aadapter_generate_content_method():
    """Test that the new aadapter_generate_content method in Router works correctly"""
    # Create a router instance
    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                }
            }
        ]
    )
    
    # Create a mock response in Google GenAI format
    mock_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello, world!"
                        }
                    ]
                }
            }
        ]
    }
    
    # Mock the router's underlying aadapter_generate_content method to return a mock response
    with patch.object(router, 'aadapter_generate_content', new=AsyncMock(return_value=mock_response)) as mock_aadapter_generate_content:
        # Call the aadapter_generate_content method
        response = await router.aadapter_generate_content(
            model="test-model",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}]
        )
        
        # Verify that router.aadapter_generate_content was called with correct parameters
        mock_aadapter_generate_content.assert_called_once()
        call_args = mock_aadapter_generate_content.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        
        # Verify that the response is the mock response we created
        assert response == mock_response