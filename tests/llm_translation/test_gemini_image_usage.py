"""
Test for Gemini image generation usage metadata extraction.

This test verifies the fix for issue #18323 where image_generation() 
was returning usage=0 while completion() returned proper token usage.
"""
import pytest
from unittest.mock import patch, MagicMock
import litellm
from litellm.types.utils import ImageResponse, ImageObject, ImageUsage


@pytest.mark.parametrize(
    "model_name",
    [
        "gemini/gemini-2.5-flash-image",
        "gemini/gemini-2.0-flash-preview-image-generation",
        "gemini/gemini-3-pro-image-preview",
    ],
)
def test_gemini_image_generation_usage_metadata(model_name: str):
    """
    Test that image_generation() properly extracts and returns usage metadata
    from Gemini API responses.
    
    This test verifies the fix for issue #18323.
    """
    
    # Mock response data that includes usageMetadata (like real Gemini API)
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "test_base64_image_data"
                            }
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 35,
            "candidatesTokenCount": 1716,
            "totalTokenCount": 1751,
            "promptTokensDetails": [
                {
                    "modality": "TEXT",
                    "tokenCount": 35
                }
            ],
            "candidatesTokensDetails": [
                {
                    "modality": "TEXT",
                    "tokenCount": 213
                },
                {
                    "modality": "IMAGE",
                    "tokenCount": 1120
                }
            ]
        }
    }
    
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        # Mock successful HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_post.return_value = mock_http_response
        
        # Call image_generation
        response = litellm.image_generation(
            model=model_name,
            prompt="A cute baby sea otter eating a cute baby spinach with cute starry cereals dressing",
            api_key="test_api_key",
        )
        
        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0
        
        # IMPORTANT: Validate usage metadata is properly extracted
        assert response.usage is not None, "Usage should not be None"
        
        # Note: The usage object might be converted to Usage type by Pydantic/OpenAI SDK
        # but it should still have the ImageUsage fields (input_tokens, output_tokens, etc.)
        
        # Validate token counts match the mock response
        assert hasattr(response.usage, 'input_tokens'), "Usage should have input_tokens attribute"
        assert hasattr(response.usage, 'output_tokens'), "Usage should have output_tokens attribute"
        assert hasattr(response.usage, 'total_tokens'), "Usage should have total_tokens attribute"
        
        assert response.usage.input_tokens == 35, f"Expected input_tokens=35, got {response.usage.input_tokens}"
        assert response.usage.output_tokens == 1716, f"Expected output_tokens=1716, got {response.usage.output_tokens}"
        assert response.usage.total_tokens == 1751, f"Expected total_tokens=1751, got {response.usage.total_tokens}"
        
        # Validate input tokens details
        assert hasattr(response.usage, 'input_tokens_details'), "Usage should have input_tokens_details attribute"
        assert response.usage.input_tokens_details is not None, "Input tokens details should not be None"
        
        # input_tokens_details might be a dict or an object
        if isinstance(response.usage.input_tokens_details, dict):
            assert response.usage.input_tokens_details['text_tokens'] == 35, f"Expected text_tokens=35, got {response.usage.input_tokens_details['text_tokens']}"
            assert response.usage.input_tokens_details['image_tokens'] == 0, f"Expected image_tokens=0, got {response.usage.input_tokens_details['image_tokens']}"
        else:
            assert response.usage.input_tokens_details.text_tokens == 35, f"Expected text_tokens=35, got {response.usage.input_tokens_details.text_tokens}"
            assert response.usage.input_tokens_details.image_tokens == 0, f"Expected image_tokens=0, got {response.usage.input_tokens_details.image_tokens}"
        
        # Verify the usage is not all zeros (the bug we're fixing)
        assert response.usage.total_tokens > 0, "Total tokens should be greater than 0"
        assert response.usage.input_tokens > 0, "Input tokens should be greater than 0"
        assert response.usage.output_tokens > 0, "Output tokens should be greater than 0"


def test_gemini_image_generation_without_usage_metadata():
    """
    Test that image_generation() handles responses without usageMetadata gracefully.
    """
    
    # Mock response data without usageMetadata
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "test_base64_image_data"
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        # Mock successful HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_post.return_value = mock_http_response
        
        # Call image_generation
        response = litellm.image_generation(
            model="gemini/gemini-3-pro-image-preview",
            prompt="Test prompt",
            api_key="test_api_key",
        )
        
        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0
        
        # Usage should be None if not present in response
        # (or have default values depending on implementation)
        # This ensures we don't crash when usageMetadata is missing


def test_gemini_imagen_models_no_usage_extraction():
    """
    Test that non-Gemini Imagen models don't attempt to extract usage metadata
    from the different response format.
    """
    
    # Mock response data for Imagen models (different format)
    mock_response_data = {
        "predictions": [
            {
                "bytesBase64Encoded": "test_base64_image_data"
            }
        ]
    }
    
    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        # Mock successful HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_post.return_value = mock_http_response
        
        # Call image_generation with an Imagen model
        response = litellm.image_generation(
            model="gemini/imagen-3.0-generate-001",
            prompt="Test prompt",
            api_key="test_api_key",
        )
        
        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        
        # For Imagen models, we don't extract usage from the predictions format
        # This test just ensures we don't crash
