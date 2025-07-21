"""
Test file for Recraft AI image generation
"""

import os
import pytest
from unittest.mock import AsyncMock, Mock, patch

import litellm
from litellm.types.utils import ImageResponse


class TestRecraftImageGeneration:
    """Test Recraft image generation functionality."""

    def test_recraft_provider_detection(self):
        """Test that Recraft provider is correctly detected."""
        model, provider, _, _ = litellm.get_llm_provider(model="recraft/recraftv3")
        assert model == "recraftv3"
        assert provider == "recraft"

    def test_recraft_image_generation_config(self):
        """Test RecraftImageGenerationConfig basic functionality."""
        from litellm.llms.recraft.image_generation import RecraftImageGenerationConfig
        
        config = RecraftImageGenerationConfig()
        
        # Test supported params
        supported_params = config.get_supported_openai_params("recraftv3")
        assert "n" in supported_params
        assert "response_format" in supported_params
        assert "size" in supported_params
        assert "user" in supported_params
        
        # Test parameter mapping
        optional_params = {}
        mapped_params = config.map_openai_params(
            non_default_params={"style": "digital_illustration", "substyle": "hand_drawn"},
            optional_params=optional_params,
            model="recraftv3",
            drop_params=False,
        )
        
        assert mapped_params["style"] == "digital_illustration"
        assert mapped_params["substyle"] == "hand_drawn"

    def test_recraft_url_generation(self):
        """Test URL generation for Recraft API."""
        from litellm.llms.recraft.image_generation import RecraftImageGenerationConfig
        
        config = RecraftImageGenerationConfig()
        
        # Test default URL
        url = config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="recraftv3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://external.api.recraft.ai/v1/images/generations"
        
        # Test custom API base
        url = config.get_complete_url(
            api_base="https://custom.api.example.com/v1",
            api_key="test-key",
            model="recraftv3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.example.com/v1/images/generations"

    def test_recraft_request_transformation(self):
        """Test request data transformation for Recraft API."""
        from litellm.llms.recraft.image_generation import RecraftImageGenerationConfig
        
        config = RecraftImageGenerationConfig()
        
        # Test basic request transformation
        data = config.transform_request(
            model="recraftv3",
            prompt="a beautiful landscape",
            optional_params={
                "n": 2,
                "size": "1024x1024",
                "style": "digital_illustration",
                "negative_prompt": "blurry, low quality"
            },
            headers={},
        )
        
        assert data["prompt"] == "a beautiful landscape"
        assert data["model"] == "recraftv3"
        assert data["n"] == 2
        assert data["size"] == "1024x1024"
        assert data["style"] == "digital_illustration"
        assert data["negative_prompt"] == "blurry, low quality"

    @pytest.mark.asyncio
    async def test_recraft_async_image_generation_mock(self):
        """Test async image generation with mocked response."""
        # Mock response data
        mock_response_data = {
            "data": [
                {
                    "url": "https://img.recraft.ai/test-image-url",
                    "b64_json": None
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        # Mock the HTTP client
        with patch("litellm.llms.recraft.image_generation.AsyncHTTPHandler") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Mock logging object
            mock_logging = Mock()
            mock_logging.pre_call = Mock()
            mock_logging.post_call = Mock()
            
            # Test the async image generation
            response = await litellm.aimage_generation(
                model="recraft/recraftv3",
                prompt="a beautiful sunset",
                api_key="test-key",
                litellm_logging_obj=mock_logging,
            )
            
            # Verify the response
            assert isinstance(response, ImageResponse)
            mock_logging.pre_call.assert_called_once()
            mock_logging.post_call.assert_called_once()
            mock_client.post.assert_called_once()

    def test_recraft_sync_image_generation_mock(self):
        """Test sync image generation with mocked response."""
        # Mock response data
        mock_response_data = {
            "data": [
                {
                    "url": "https://img.recraft.ai/test-image-url",
                    "b64_json": None
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        # Mock the HTTP client
        with patch("litellm.llms.recraft.image_generation.HTTPHandler") as mock_client_class:
            mock_client = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Mock logging object
            mock_logging = Mock()
            mock_logging.pre_call = Mock()
            mock_logging.post_call = Mock()
            
            # Test the sync image generation
            response = litellm.image_generation(
                model="recraft/recraftv3",
                prompt="a beautiful sunset",
                api_key="test-key",
                litellm_logging_obj=mock_logging,
            )
            
            # Verify the response
            assert isinstance(response, ImageResponse)
            mock_logging.pre_call.assert_called_once()
            mock_logging.post_call.assert_called_once()
            mock_client.post.assert_called_once()

    def test_recraft_environment_validation(self):
        """Test environment validation for API key."""
        from litellm.llms.recraft.image_generation import RecraftImageGenerationConfig
        
        config = RecraftImageGenerationConfig()
        
        # Test with valid API key
        headers = {}
        validated_headers = config.validate_environment(
            headers=headers,
            model="recraftv3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        
        assert "Authorization" in validated_headers
        assert validated_headers["Authorization"] == "Bearer test-key"
        assert validated_headers["Content-Type"] == "application/json"
        
        # Test with missing API key
        with pytest.raises(ValueError, match="Recraft API key is required"):
            config.validate_environment(
                headers={},
                model="recraftv3",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_recraft_provider_in_utils(self):
        """Test that Recraft is properly registered in ProviderConfigManager."""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager
        
        # Test that RECRAFT provider exists
        assert hasattr(LlmProviders, "RECRAFT")
        assert LlmProviders.RECRAFT == "recraft"
        
        # Test that provider config can be retrieved
        config = ProviderConfigManager.get_provider_image_generation_config(
            model="recraftv3",
            provider=LlmProviders.RECRAFT,
        )
        
        assert config is not None
        from litellm.llms.recraft.image_generation import RecraftImageGenerationConfig
        assert isinstance(config, RecraftImageGenerationConfig)


if __name__ == "__main__":
    pytest.main([__file__])