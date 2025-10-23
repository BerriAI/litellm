import logging
import os
import sys
import traceback
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.utils import ImageObject


@pytest.mark.asyncio
async def test_xinference_image_generation():
    """Test basic xinference image generation with mocked OpenAI client."""
    
    # Mock OpenAI response
    mock_openai_response = {
        "created": 1699623600,
        "data": [
            {
                "url": "https://example.com/image.png"
            }
        ]
    }
    
    # Create a proper mock response object
    class MockResponse:
        def model_dump(self):
            return mock_openai_response
    
    # Create a mock client with the images.generate method
    mock_client = AsyncMock()
    mock_client.images.generate = AsyncMock(return_value=MockResponse())
    
    # Capture the actual arguments sent to OpenAI client
    captured_args = None
    captured_kwargs = None
    
    async def capture_generate_call(*args, **kwargs):
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = kwargs
        return MockResponse()
    
    mock_client.images.generate.side_effect = capture_generate_call
    
    # Mock the _get_openai_client method to return our mock client
    with patch.object(litellm.main.openai_chat_completions, '_get_openai_client', return_value=mock_client):
        response = await litellm.aimage_generation(
            model="xinference/stabilityai/stable-diffusion-3.5-large",
            prompt="A beautiful sunset over a calm ocean",
            api_base="http://mock.image.generation.api",
        )
        
        # Print the captured arguments for debugging
        print("Arguments sent to openai_aclient.images.generate:")
        print("args:", json.dumps(captured_args, indent=4, default=str))
        print("kwargs:", json.dumps(captured_kwargs, indent=4, default=str))
        
        # Validate the response
        assert response is not None
        assert response.created == 1699623600
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url == "https://example.com/image.png"
        
        # Validate that the OpenAI client was called with correct parameters
        mock_client.images.generate.assert_called_once()
        assert captured_kwargs is not None
        assert captured_kwargs["model"] == "stabilityai/stable-diffusion-3.5-large"  # xinference/ prefix removed
        assert captured_kwargs["prompt"] == "A beautiful sunset over a calm ocean"


@pytest.mark.asyncio
async def test_xinference_image_generation_with_response_format():
    """
    Test xinference image generation with additional parameters.
    Ensure all documented params are passed in.

    https://inference.readthedocs.io/en/v1.1.1/reference/generated/xinference.client.handlers.ImageModelHandle.text_to_image.html#xinference.client.handlers.ImageModelHandle.text_to_image
    """
    
    # Mock OpenAI response
    mock_openai_response = {
        "created": 1699623600,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg=="
            }
        ]
    }
    
    # Create a proper mock response object
    class MockResponse:
        def model_dump(self):
            return mock_openai_response
    
    # Create a mock client with the images.generate method
    mock_client = AsyncMock()
    mock_client.images.generate = AsyncMock(return_value=MockResponse())
    
    # Capture the actual arguments sent to OpenAI client
    captured_args = None
    captured_kwargs = None
    
    async def capture_generate_call(*args, **kwargs):
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = kwargs
        return MockResponse()
    
    mock_client.images.generate.side_effect = capture_generate_call
    
    # Mock the _get_openai_client method to return our mock client
    with patch.object(litellm.main.openai_chat_completions, '_get_openai_client', return_value=mock_client):
        response = await litellm.aimage_generation(
            model="xinference/stabilityai/stable-diffusion-3.5-large",
            api_base="http://mock.image.generation.api",
            prompt="A beautiful sunset over a calm ocean",
            response_format="b64_json",
            n=1,
            size="1024x1024",
        )
        
        # Print the captured arguments for debugging
        print("Arguments sent to openai_aclient.images.generate:")
        print("args:", json.dumps(captured_args, indent=4, default=str))
        print("kwargs:", json.dumps(captured_kwargs, indent=4, default=str))
        
        # Validate the response
        assert response is not None
        assert response.created == 1699623600
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        
        # Validate that the OpenAI client was called with correct parameters
        mock_client.images.generate.assert_called_once()
        assert captured_kwargs is not None
        assert captured_kwargs["model"] == "stabilityai/stable-diffusion-3.5-large"  # xinference/ prefix removed
        assert captured_kwargs["prompt"] == "A beautiful sunset over a calm ocean"
        assert captured_kwargs["response_format"] == "b64_json"
        assert captured_kwargs["n"] == 1
        assert captured_kwargs["size"] == "1024x1024"
        expected_args = ["model", "prompt", "response_format", "n", "size"]
        # only expected args should be present
        assert all(arg in captured_kwargs for arg in expected_args)

