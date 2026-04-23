import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import aimage_generation


@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("fal_ai/fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1-ultra"),
        ("fal_ai/fal-ai/stable-diffusion-v35-medium", "fal-ai/stable-diffusion-v35-medium"),
    ],
)
@pytest.mark.asyncio
async def test_fal_ai_image_generation_basic(model, expected_endpoint):
    """
    Test that fal_ai image generation constructs correct request body and URL.
    
    Validates:
    - Correct API endpoint URL construction
    - Proper request body format with prompt
    - Correct Authorization header format
    """
    captured_url = None
    captured_json_data = None
    captured_headers = None
    
    def capture_post_call(*args, **kwargs):
        nonlocal captured_url, captured_json_data, captured_headers
        
        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")
        captured_headers = kwargs.get("headers")
        
        # Mock response with fal.ai format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://example.com/generated-image.png",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg"
                }
            ],
            "seed": 42
        }
        
        return mock_response
    
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_post.side_effect = capture_post_call
        
        test_api_key = "test-fal-ai-key-12345"
        test_prompt = "A cute baby sea otter"
        
        response = await aimage_generation(
            model=model,
            prompt=test_prompt,
            api_key=test_api_key,
        )
        
        # Validate response
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0
        
        # Validate URL
        assert captured_url is not None
        assert "fal.run" in captured_url
        assert expected_endpoint in captured_url
        print(f"Validated URL: {captured_url}")
        
        # Validate headers
        assert captured_headers is not None
        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == f"Key {test_api_key}"
        print(f"Validated headers: {captured_headers}")
        
        # Validate request body
        assert captured_json_data is not None
        assert captured_json_data["prompt"] == test_prompt
        print(f"Validated request body: {captured_json_data}")

