from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails

SEEDREAM_MODELS: Final = (
    "tencent/seedream-image-v5.0-pro",
    "tencent/seedream-image-v5.0-flash",
    "tencent/seedream-image-v5.0-lite",
)


def _image_response(total_tokens: int, images: int = 1) -> ImageResponse:
    return ImageResponse(
        created=1792551825,
        data=[ImageObject(url=f"https://cos.example/{index}.jpg") for index in range(images)],
        usage=ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
            output_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


@pytest.mark.parametrize("model", SEEDREAM_MODELS)
@pytest.mark.parametrize("call_type", ["image_generation", "image_edit"])
def test_seedream_bills_tokenhub_usage_tokens_for_a_tier_size(
    local_model_cost_map: None, model: str, call_type: str
) -> None:
    token_rate: Final = litellm.model_cost[model]["output_cost_per_image_token"]

    cost: Final = litellm.completion_cost(
        completion_response=_image_response(total_tokens=56250),
        model=model,
        call_type=call_type,
        size="2K",
    )

    assert cost == pytest.approx(56250 * token_rate)


@pytest.mark.parametrize("model", SEEDREAM_MODELS)
def test_seedream_falls_back_to_the_per_image_price_without_usage(local_model_cost_map: None, model: str) -> None:
    per_image_price: Final = litellm.model_cost[model]["output_cost_per_image"]

    cost: Final = litellm.completion_cost(
        completion_response=_image_response(total_tokens=0, images=2),
        model=model,
        call_type="image_generation",
        size="2K",
    )

    assert cost == pytest.approx(2 * per_image_price)


async def test_a_deployment_priced_tencent_image_model_outside_seedream_bills_its_deployment_price() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"created": 1792551825, "data": [{"url": "https://cos.example/hy.png"}]})

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "hy-image",
                "litellm_params": {"model": "tencent/hy-image-v3.5-preview", "api_key": "sk-tokenhub"},
                "model_info": {"input_cost_per_image": 0.04},
            }
        ]
    )

    response: Final = await router.aimage_generation(
        model="hy-image",
        prompt="a red paper lantern",
        client=openai.AsyncOpenAI(
            api_key="sk-tokenhub",
            base_url="https://tokenhub-intl.tencentcloudmaas.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ),
    )

    assert response._hidden_params["response_cost"] == pytest.approx(0.04)


def test_a_seedream_deployment_priced_per_token_in_the_ui_bills_its_own_rate() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "created": 1792551825,
                "data": [{"output_format": "jpeg", "size": "1024x1024", "url": "https://cos.example/0.jpg"}],
                "tokenhub_usage": {"total_tokens": 11250},
            },
        )

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "seedream-flash",
                "litellm_params": {
                    "model": "tencent/seedream-image-v5.0-flash",
                    "api_key": "sk-tokenhub",
                    "output_cost_per_token": 0.000001,
                },
            }
        ]
    )

    response: Final = router.image_generation(
        model="seedream-flash",
        prompt="a red paper lantern",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
    )

    assert response._hidden_params["response_cost"] == pytest.approx(11250 * 0.000001)
