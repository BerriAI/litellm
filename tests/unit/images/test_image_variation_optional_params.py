"""`image_variation()` must forward its OpenAI-style arguments to the provider.

`n`, `response_format`, `size` and `user` are declared on the public function
but were dropped before the request was built.
"""

from unittest.mock import patch

import litellm
from litellm.images.main import image_variation


def _image_response() -> "litellm.utils.ImageResponse":
    return litellm.utils.ImageResponse(
        created=1234567890,
        data=[{"url": "https://example.com/image.png"}],
    )


def _optional_params(mock_call) -> dict:
    return mock_call.call_args.kwargs["optional_params"]


class TestImageVariationOptionalParams:
    @patch("litellm.images.main.openai_image_variations")
    def test_openai_receives_the_requested_params(self, mock_openai) -> None:
        mock_openai.image_variations.return_value = _image_response()

        image_variation(
            image=b"fake-image-bytes",
            model="openai/dall-e-2",
            n=4,
            size="512x512",
            response_format="b64_json",
            user="user-123",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )

        optional_params = _optional_params(mock_openai.image_variations)
        assert optional_params["n"] == 4
        assert optional_params["size"] == "512x512"
        assert optional_params["response_format"] == "b64_json"
        assert optional_params["user"] == "user-123"

    @patch("litellm.images.main.openai_image_variations")
    def test_omitted_params_are_not_invented(self, mock_openai) -> None:
        """Nothing the caller left out may be sent, so provider defaults stand."""
        mock_openai.image_variations.return_value = _image_response()

        image_variation(
            image=b"fake-image-bytes",
            model="openai/dall-e-2",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )

        assert _optional_params(mock_openai.image_variations) == {}

    @patch("litellm.images.main.base_llm_aiohttp_handler")
    def test_topaz_maps_size_and_response_format(self, mock_handler) -> None:
        """Topaz renames these, which only happens if map_openai_params runs."""
        mock_handler.image_variations.return_value = _image_response()

        image_variation(
            image=b"fake-image-bytes",
            model="topaz/Standard V2",
            size="1024x768",
            response_format="b64_json",
            api_key="topaz-key",
            api_base="https://api.topazlabs.com",
        )

        optional_params = _optional_params(mock_handler.image_variations)
        assert optional_params["output_width"] == "1024"
        assert optional_params["output_height"] == "768"
        assert optional_params["output_format"] == "b64_json"
