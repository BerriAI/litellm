import json
from typing import Final

import pytest
import respx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.minimax.image_generation.transformation import (
    MinimaxImageGenerationConfig,
)

GENERATIONS_URL: Final = "https://api.minimax.io/v1/images/generations"
PROMPT_TEXT_TOKENS: Final = 3174
OUTPUT_IMAGE_TOKENS: Final = 4080


def _generation_response(model: str) -> dict[str, object]:
    return {
        "created": 1794341959,
        "data": [{"url": "https://example.com/h3-image/output-0.jpg", "size": "1024x1024", "revised_prompt": "a cube"}],
        "model": model,
        "output_format": "jpeg",
        "usage": {
            "generated_images": 1,
            "total_tokens": PROMPT_TEXT_TOKENS + OUTPUT_IMAGE_TOKENS,
            "input_tokens": PROMPT_TEXT_TOKENS,
            "output_tokens": OUTPUT_IMAGE_TOKENS,
            "input_tokens_details": {"text_tokens": PROMPT_TEXT_TOKENS, "image_tokens": 0},
            "output_tokens_details": {"image_tokens": OUTPUT_IMAGE_TOKENS},
        },
    }


def test_minimax_image_generation_url_and_body():
    config = MinimaxImageGenerationConfig()
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="MiniMax-H3-Image",
            optional_params={},
            litellm_params={},
        )
        == "https://api.minimax.io/v1/images/generations"
    )
    assert "aspect_ratio" in config.get_supported_openai_params("MiniMax-H3-Image")
    assert "area" in config.get_supported_openai_params("MiniMax-H3-Image")
    assert config.transform_image_generation_request(
        model="MiniMax-H3-Image",
        prompt="a red cube",
        optional_params={"aspect_ratio": "1:1", "area": "1K"},
        litellm_params={},
        headers={},
    ) == {
        "model": "MiniMax-H3-Image",
        "prompt": "a red cube",
        "aspect_ratio": "1:1",
        "area": "1K",
    }


@pytest.mark.parametrize("model", ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"])
def test_minimax_image_generation_is_billed_at_its_cost_map_token_rates(model: str):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(GENERATIONS_URL).respond(200, json=_generation_response(model))
        response: Final = litellm.image_generation(
            model=f"minimax/{model}",
            prompt="a red cube",
            api_key="sk-fake",
            aspect_ratio="1:1",
            area="1K",
            output_format="png",
        )

    sent: Final = json.loads(route.calls.last.request.content)
    row: Final = litellm.model_cost[f"minimax/{model}"]
    expected_cost: Final = (
        PROMPT_TEXT_TOKENS * row["input_cost_per_token"] + OUTPUT_IMAGE_TOKENS * row["output_cost_per_image_token"]
    )
    assert sent == {"model": model, "prompt": "a red cube", "aspect_ratio": "1:1", "area": "1K", "output_format": "png"}
    assert response.data[0].url == "https://example.com/h3-image/output-0.jpg"
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("params", "expected_body_params"),
    [
        ({"extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"area": "2K", "extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"extra_body": {"model": "MiniMax-H3-Image-Highspeed", "seed": 7}}, {"seed": 7}),
    ],
)
def test_minimax_image_generation_sends_extra_body_fields_at_the_top_level(
    params: dict[str, object], expected_body_params: dict[str, object]
):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(GENERATIONS_URL).respond(200, json=_generation_response("MiniMax-H3-Image"))
        litellm.image_generation(model="minimax/MiniMax-H3-Image", prompt="a red cube", api_key="sk-fake", **params)

    assert json.loads(route.calls.last.request.content) == {
        "model": "MiniMax-H3-Image",
        "prompt": "a red cube",
        **expected_body_params,
    }


def test_minimax_image_generation_refuses_stream_before_calling_minimax():
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(GENERATIONS_URL).respond(200, json=_generation_response("MiniMax-H3-Image"))
        with pytest.raises(UnsupportedParamsError) as error:
            litellm.image_generation(model="minimax/MiniMax-H3-Image", prompt="a red cube", api_key="sk-fake", stream=True)

    assert error.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("params", [{"stream": True}, {"extra_body": {"stream": True}}])
def test_minimax_image_generation_drops_stream_under_drop_params(params: dict[str, object]):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(GENERATIONS_URL).respond(200, json=_generation_response("MiniMax-H3-Image"))
        litellm.image_generation(
            model="minimax/MiniMax-H3-Image", prompt="a red cube", api_key="sk-fake", drop_params=True, **params
        )

    assert json.loads(route.calls.last.request.content) == {"model": "MiniMax-H3-Image", "prompt": "a red cube"}
