from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from litellm import aimage_generation
from litellm.llms.fal_ai.image_generation import get_fal_ai_image_generation_config
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("fal_ai/fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1-ultra"),
        (
            "fal_ai/fal-ai/stable-diffusion-v35-medium",
            "fal-ai/stable-diffusion-v35-medium",
        ),
        ("fal_ai/fal-ai/nano-banana", "fal-ai/nano-banana"),
        (
            "fal_ai/fal-ai/gemini-25-flash-image",
            "fal-ai/gemini-25-flash-image",
        ),
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
                    "content_type": "image/jpeg",
                }
            ],
            "seed": 42,
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


@pytest.mark.parametrize(
    "model,expected_url",
    [
        ("bytedance/seedream/v5.5/text-to-image", "https://fal.run/bytedance/seedream/v5.5/text-to-image"),
        ("bytedance/seedream/v5/lite/text-to-image", "https://fal.run/bytedance/seedream/v5/lite/text-to-image"),
        (
            "fal-ai/bytedance/seedream/v4.5/text-to-image",
            "https://fal.run/fal-ai/bytedance/seedream/v4.5/text-to-image",
        ),
        ("fal-ai/bytedance/seedream/v3/text-to-image", "https://fal.run/fal-ai/bytedance/seedream/v3/text-to-image"),
        ("bytedance/seedream/v3/text-to-image", "https://fal.run/fal-ai/bytedance/seedream/v3/text-to-image"),
        ("bytedance/seedream", "https://fal.run/fal-ai/bytedance/seedream/v3/text-to-image"),
        (
            "fal-ai/bytedance/dreamina/v3.1/text-to-image",
            "https://fal.run/fal-ai/bytedance/dreamina/v3.1/text-to-image",
        ),
        ("fal-ai/flux-pro/v1/vto", "https://fal.run/fal-ai/flux-pro/v1/vto"),
        ("fal-ai/flux-pro/kontext/text-to-image", "https://fal.run/fal-ai/flux-pro/kontext/text-to-image"),
        ("fal-ai/flux-pro/v1.1-ultra-finetuned", "https://fal.run/fal-ai/flux-pro/v1.1-ultra-finetuned"),
        ("fal-ai/flux-pro/v1.1-ultra", "https://fal.run/fal-ai/flux-pro/v1.1-ultra"),
        ("flux-pro/v1.1-ultra", "https://fal.run/fal-ai/flux-pro/v1.1-ultra"),
        ("fal-ai/flux-pro/v1.1", "https://fal.run/fal-ai/flux-pro/v1.1"),
        ("flux-pro/v1.1", "https://fal.run/fal-ai/flux-pro/v1.1"),
        ("fal-ai/flux-pro", "https://fal.run/fal-ai/flux-pro/v1.1"),
        ("flux-pro", "https://fal.run/fal-ai/flux-pro/v1.1"),
        ("fal-ai/flux/schnell", "https://fal.run/fal-ai/flux/schnell"),
        ("fal-ai/flux/dev", "https://fal.run/fal-ai/flux/dev"),
    ],
)
def test_fal_ai_seedream_and_flux_pro_models_post_to_the_endpoint_they_name(model: str, expected_url: str) -> None:
    config: Final = get_fal_ai_image_generation_config(model)

    assert (
        config.get_complete_url(
            api_base="https://fal.run", api_key=None, model=model, optional_params={}, litellm_params={}
        )
        == expected_url
    )


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.mark.parametrize(
    "model",
    [
        "fal-ai/flux-pro/../../bytedance/seedream/v5.5/text-to-image",
        "fal-ai/flux-pro/./v1/vto",
        "fal-ai/bytedance/seedream/..",
    ],
)
def test_fal_ai_model_path_with_dot_segments_is_rejected_before_any_request(model: str) -> None:
    config: Final = get_fal_ai_image_generation_config(model)

    with pytest.raises(ValueError, match="dot path segment"):
        config.get_complete_url(
            api_base="https://fal.run", api_key=None, model=model, optional_params={}, litellm_params={}
        )
