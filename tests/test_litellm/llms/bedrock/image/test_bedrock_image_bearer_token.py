import json
import os
import sys
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

# Mock response for Bedrock image generation
mock_image_response = {
    "images": ["base64_encoded_image_data"],
    "error": None
}

class TestBedrockImageGeneration:
    def test_image_generation_with_api_key_bearer_token(self):
        """Test image generation with bearer token authentication"""
        litellm.set_verbose = True
        test_api_key = "test-bearer-token-12345"
        model = "bedrock/stability.sd3-large-v1:0"
        prompt = "A cute baby sea otter"

        with patch("litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.image_generation") as mock_bedrock_image_gen:
            # Setup mock response
            mock_image_response_obj = litellm.ImageResponse()
            mock_image_response_obj.data = [{"url": "https://example.com/image.jpg"}]
            mock_bedrock_image_gen.return_value = mock_image_response_obj

            response = litellm.image_generation(
                model=model,
                prompt=prompt,
                aws_region_name="us-west-2",
                api_key=test_api_key
            )

            assert response is not None
            assert len(response.data) > 0
            
            mock_bedrock_image_gen.assert_called_once()
            for call in mock_bedrock_image_gen.call_args_list:
                if "headers" in call.kwargs:
                    headers = call.kwargs["headers"]
                    if "Authorization" in headers and headers["Authorization"] == f"Bearer {test_api_key}":
                        break

    def test_image_generation_with_env_variable_bearer_token(self, monkeypatch):
        """Test image generation with bearer token from environment variable"""
        litellm.set_verbose = True
        test_api_key = "env-bearer-token-12345"
        model = "bedrock/stability.sd3-large-v1:0"
        prompt = "A cute baby sea otter"
        
        # Mock the environment variable
        with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": test_api_key}), \
             patch("litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.image_generation") as mock_bedrock_image_gen:
            
            mock_image_response_obj = litellm.ImageResponse()
            mock_image_response_obj.data = [{"url": "https://example.com/image.jpg"}]
            mock_bedrock_image_gen.return_value = mock_image_response_obj

            response = litellm.image_generation(
                model=model,
                prompt=prompt,
                aws_region_name="us-west-2"
            )

            assert response is not None
            assert len(response.data) > 0
            
            mock_bedrock_image_gen.assert_called_once()
            for call in mock_bedrock_image_gen.call_args_list:
                if "headers" in call.kwargs:
                    headers = call.kwargs["headers"]
                    if "Authorization" in headers and headers["Authorization"] == f"Bearer {test_api_key}":
                        break

    @pytest.mark.asyncio
    async def test_async_image_generation_with_bearer_token(self):
        """Test async image generation with bearer token authentication"""
        litellm.set_verbose = True
        test_api_key = "async-bearer-token-12345"
        model = "bedrock/stability.sd3-large-v1:0"
        prompt = "A cute baby sea otter"

        with patch("litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.async_image_generation") as mock_async_bedrock_image_gen:
            mock_image_response_obj = litellm.ImageResponse()
            mock_image_response_obj.data = [{"url": "https://example.com/image.jpg"}]
            mock_async_bedrock_image_gen.return_value = mock_image_response_obj

            # Call async image generation with api_key parameter
            response = await litellm.aimage_generation(
                model=model,
                prompt=prompt,
                aws_region_name="us-west-2",
                api_key=test_api_key
            )

            assert response is not None
            assert len(response.data) > 0
            
            mock_async_bedrock_image_gen.assert_called_once()
            for call in mock_async_bedrock_image_gen.call_args_list:
                if "headers" in call.kwargs:
                    headers = call.kwargs["headers"]
                    if "Authorization" in headers and headers["Authorization"] == f"Bearer {test_api_key}":
                        break

    def test_image_generation_with_sigv4(self):
        """Test image generation falls back to SigV4 auth when no bearer token is provided"""
        litellm.set_verbose = True
        model = "bedrock/stability.sd3-large-v1:0"
        prompt = "A cute baby sea otter"

        with patch("litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.image_generation") as mock_bedrock_image_gen:
            mock_image_response_obj = litellm.ImageResponse()
            mock_image_response_obj.data = [{"url": "https://example.com/image.jpg"}]
            mock_bedrock_image_gen.return_value = mock_image_response_obj

            response = litellm.image_generation(
                model=model,
                prompt=prompt,
                aws_region_name="us-west-2"
            )
            
            assert response is not None
            assert len(response.data) > 0
            mock_bedrock_image_gen.assert_called_once()