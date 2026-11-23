"""deAPI `/v1/images/edits`: OpenAI's multipart edit request on the deAPI gateway, which reports the
charge in `usage.cost`. deAPI has no inpainting model, so `mask` is not offered."""

import io

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.deapi.image_edit.transformation import DeapiImageEditConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DEAPI_EDITS_URL = "https://oai.deapi.ai/v1/images/edits"
MODEL = "deapi/Flux_2_Klein_4B_BF16"
SELLER_MODEL = "Flux_2_Klein_4B_BF16"
REPORTED_COST = 0.033156
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PNG_B64 = "iVBORw0KGgo="


def _edit(cost: float | None = REPORTED_COST) -> httpx.Response:
    usage = {
        "input_tokens": 0,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
        "output_tokens": 0,
        "total_tokens": 0,
        "output_tokens_details": None,
    }
    body = {
        "created": 1795858900,
        "data": [{"b64_json": PNG_B64}],
        "usage": usage if cost is None else {**usage, "cost": cost},
    }
    return httpx.Response(200, json=body, headers={} if cost is None else {"X-DeAPI-Cost": f"{cost:.8f}"})


def _image() -> io.BytesIO:
    image = io.BytesIO(PNG_BYTES)
    image.name = "dot.png"
    return image


class TestRegistration:
    def test_deapi_is_a_native_image_edit_provider(self):
        config = ProviderConfigManager.get_provider_image_edit_config(model=SELLER_MODEL, provider=LlmProviders.DEAPI)

        assert isinstance(config, DeapiImageEditConfig)

    def test_mask_is_not_offered(self):
        assert "mask" not in DeapiImageEditConfig().get_supported_openai_params(SELLER_MODEL)


class TestImageEdit:
    def test_posts_multipart_with_the_bearer_key_and_the_image(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EDITS_URL).mock(return_value=_edit())

        response = litellm.image_edit(model=MODEL, image=_image(), prompt="make it blue", size="512x512")

        assert response.data[0].b64_json == PNG_B64
        request = respx_mock.calls.last.request
        assert request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert request.headers["Content-Type"].startswith("multipart/form-data")
        body = request.content
        assert b"make it blue" in body
        assert PNG_BYTES in body
        assert SELLER_MODEL.encode() in body

    def test_the_reported_cost_is_the_response_cost(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EDITS_URL).mock(return_value=_edit())

        response = litellm.image_edit(model=MODEL, image=_image(), prompt="make it blue")

        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST

    def test_an_answer_without_cost_leaves_pricing_to_the_price_map(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_EDITS_URL).mock(return_value=_edit(cost=None))

        response = litellm.image_edit(model=MODEL, image=_image(), prompt="make it blue")

        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.image_edit(model=MODEL, image=_image(), prompt="make it blue")
        assert not respx_mock.calls
