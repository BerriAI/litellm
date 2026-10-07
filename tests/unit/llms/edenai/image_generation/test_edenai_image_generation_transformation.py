"""Eden AI `/v3/images/generations`: OpenAI's image generation API served by Eden's gateway, which
reports the real per-request cost at the top level of the body."""

import json

import httpx
import pytest

import litellm
from litellm.cost_calculator import get_response_cost_from_hidden_params
from litellm.llms.edenai.image_generation.transformation import EdenAIImageGenerationConfig
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_IMAGES_URL = f"{EDEN_BASE}/images/generations"
EDEN_REPORTED_COST = 0.0042
MODEL = "edenai/openai/gpt-image-1-mini"
SELLER_MODEL = "openai/gpt-image-1-mini"
PNG_B64 = "iVBORw0KGgoAAAANSUhE"


def _eden_image(cost: float | None = EDEN_REPORTED_COST) -> dict:
    """Live `/v3/images/generations` body: OpenAI shape plus Eden's top-level `cost` and `provider`."""
    body = {
        "created": 1788818607,
        "background": None,
        "data": [{"b64_json": PNG_B64, "revised_prompt": None, "url": None}],
        "output_format": "png",
        "quality": "low",
        "size": "1024x1024",
        "usage": {
            "total_tokens": 281,
            "input_tokens": 9,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 9},
            "output_tokens": 272,
            "output_tokens_details": {"image_tokens": 272, "text_tokens": 0},
        },
        "provider": "openai",
    }
    return body if cost is None else {**body, "cost": cost}


def _request_body(respx_mock) -> dict:
    return json.loads(respx_mock.calls.last.request.content)


class TestRegistration:
    def test_eden_is_a_native_image_generation_provider(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=SELLER_MODEL, provider=LlmProviders.EDENAI
        )

        assert isinstance(config, EdenAIImageGenerationConfig)


class TestAuthentication:
    def test_missing_key_is_an_authentication_error_before_any_request(self, no_eden_key, respx_mock):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            litellm.image_generation(model=MODEL, prompt="a red square")
        assert not respx_mock.calls


class TestImageGeneration:
    def test_a_param_outside_the_openai_image_set_is_rejected_unless_dropped(self, eden_key, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(200, json=_eden_image()))

        with pytest.raises(litellm.UnsupportedParamsError, match="imageConfig"):
            litellm.image_generation(model=MODEL, prompt="a red square", imageConfig={"aspectRatio": "16:9"})
        litellm.image_generation(
            model=MODEL, prompt="a red square", imageConfig={"aspectRatio": "16:9"}, drop_params=True
        )

        assert "imageConfig" not in _request_body(respx_mock)

    def test_posts_to_eden_with_the_bearer_key_and_the_seller_model_id(self, eden_key, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(200, json=_eden_image()))

        response = litellm.image_generation(model=MODEL, prompt="a red square", size="1024x1024", quality="low", n=1)

        assert isinstance(response, ImageResponse)
        assert response.data[0].b64_json == PNG_B64
        assert respx_mock.calls.last.request.headers["Authorization"] == f"Bearer {eden_key}"
        assert _request_body(respx_mock) == {
            "model": SELLER_MODEL,
            "prompt": "a red square",
            "size": "1024x1024",
            "quality": "low",
            "n": 1,
        }

    def test_eden_reported_cost_beats_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(200, json=_eden_image()))

        response = litellm.image_generation(model=MODEL, prompt="a red square")

        assert get_response_cost_from_hidden_params(response._hidden_params) == EDEN_REPORTED_COST
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST

    def test_a_body_without_cost_leaves_pricing_to_the_price_map(self, eden_key, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(200, json=_eden_image(cost=None)))

        response = litellm.image_generation(model=MODEL, prompt="a red square")

        assert get_response_cost_from_hidden_params(response._hidden_params) is None
        assert response.usage is not None
        assert response.usage.output_tokens == 272

    @pytest.mark.asyncio
    async def test_async_call_tracks_the_same_cost(self, eden_key, httpx_transport, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(200, json=_eden_image()))

        response = await litellm.aimage_generation(model=MODEL, prompt="a red square")

        assert response.data[0].b64_json == PNG_B64
        assert response._hidden_params["response_cost"] == EDEN_REPORTED_COST


class TestErrors:
    def test_middleware_401_maps_to_authentication_error(self, eden_key, respx_mock):
        respx_mock.post(EDEN_IMAGES_URL).mock(return_value=httpx.Response(401, json={"detail": "Invalid token."}))

        with pytest.raises(litellm.AuthenticationError, match="Invalid token"):
            litellm.image_generation(model=MODEL, prompt="a red square")
