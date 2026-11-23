"""deAPI `/v1/images/generations`: the answer carries a gpt-image style `usage` whose `cost` is the
charge for all `n` images."""

import json

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.deapi.image_generation.transformation import DeapiImageGenerationConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

DEAPI_IMAGES_URL = "https://oai.deapi.ai/v1/images/generations"
MODEL = "deapi/Flux_2_Klein_4B_BF16"
SELLER_MODEL = "Flux_2_Klein_4B_BF16"
REPORTED_COST = 0.001244
PNG_B64 = "iVBORw0KGgo="


def _image(cost: float | None = REPORTED_COST) -> httpx.Response:
    usage = {
        "input_tokens": 0,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
        "output_tokens": 0,
        "total_tokens": 0,
        "output_tokens_details": None,
    }
    body = {
        "created": 1795858800,
        "data": [{"b64_json": PNG_B64}],
        "usage": usage if cost is None else {**usage, "cost": cost},
    }
    return httpx.Response(200, json=body, headers={} if cost is None else {"X-DeAPI-Cost": f"{cost:.8f}"})


class TestRegistration:
    def test_deapi_is_a_native_image_generation_provider(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=SELLER_MODEL, provider=LlmProviders.DEAPI
        )

        assert isinstance(config, DeapiImageGenerationConfig)

    def test_missing_key_is_an_authentication_error_before_any_request(self, no_deapi_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            litellm.image_generation(model=MODEL, prompt="a red dot")
        assert not respx_mock.calls


class TestImageGeneration:
    def test_posts_the_openai_request_with_the_bearer_key(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_IMAGES_URL).mock(return_value=_image())

        response = litellm.image_generation(
            model=MODEL, prompt="a red dot", size="512x512", quality="low", n=1, response_format="b64_json"
        )

        assert response.data[0].b64_json == PNG_B64
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {deapi_key}"
        assert json.loads(respx_mock.calls.last.request.content) == {
            "model": SELLER_MODEL,
            "prompt": "a red dot",
            "size": "512x512",
            "quality": "low",
            "n": 1,
            "response_format": "b64_json",
        }

    def test_the_reported_cost_is_the_response_cost(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_IMAGES_URL).mock(return_value=_image())

        response = litellm.image_generation(model=MODEL, prompt="a red dot")

        assert get_response_cost_from_hidden_params(response._hidden_params) == REPORTED_COST
        assert response._hidden_params["response_cost"] == REPORTED_COST

    def test_an_answer_without_cost_leaves_pricing_to_the_price_map(self, deapi_key, respx_mock):
        respx_mock.post(DEAPI_IMAGES_URL).mock(return_value=_image(cost=None))

        response = litellm.image_generation(model=MODEL, prompt="a red dot")

        assert get_response_cost_from_hidden_params(response._hidden_params) is None

    def test_a_value_the_gateway_cannot_render_reaches_it_and_its_error_surfaces(self, deapi_key, respx_mock):
        envelope = {
            "error": {
                "message": "background=transparent is not supported, images are PNG without alpha.",
                "type": "invalid_request_error",
                "param": "background",
                "code": "param_not_supported",
            }
        }
        respx_mock.post(DEAPI_IMAGES_URL).mock(return_value=httpx.Response(400, json=envelope))

        with pytest.raises(litellm.BadRequestError, match="without alpha"):
            litellm.image_generation(model=MODEL, prompt="a red dot", background="transparent")
        assert json.loads(respx_mock.calls.last.request.content)["background"] == "transparent"

    @pytest.mark.asyncio
    async def test_async_call_tracks_the_same_cost(self, deapi_key, httpx_transport, respx_mock):
        respx_mock.post(DEAPI_IMAGES_URL).mock(return_value=_image())

        response = await litellm.aimage_generation(model=MODEL, prompt="a red dot")

        assert response._hidden_params["response_cost"] == REPORTED_COST
