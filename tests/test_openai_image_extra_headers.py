import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.utils import ImageResponse


@pytest.mark.asyncio
async def test_openai_image_generation_extra_headers_not_in_body():
    openai_handler = OpenAIChatCompletion()

    mock_client = MagicMock()
    mock_images = MagicMock()
    mock_client.images = mock_images
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "created": 123456789,
        "data": [{"url": "https://example.com/image.png"}],
    }
    mock_images.generate = AsyncMock(return_value=mock_response)

    with patch.object(openai_handler, "_get_openai_client", return_value=mock_client):
        data = {
            "model": "gpt-image-2",
            "prompt": "a test image",
            "extra_headers": {"cf-aig-authorization": "Bearer token123"},
        }
        await openai_handler.aimage_generation(
            prompt="a test image",
            data=data,
            model_response=ImageResponse(),
            timeout=10,
            logging_obj=MagicMock(),
            api_key="test-key",
        )

        assert mock_images.generate.called
        call_kwargs = mock_images.generate.call_args[1]

        # extra_headers should be passed as a top-level kwarg to OpenAI SDK, not merged in data body
        assert "extra_headers" in call_kwargs
        assert call_kwargs["extra_headers"] == {"cf-aig-authorization": "Bearer token123"}
        assert "model" in call_kwargs
        assert "prompt" in call_kwargs
