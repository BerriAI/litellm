import base64
import json
from typing import Final

import pytest
import respx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.minimax.image_edit.transformation import MinimaxImageEditConfig
from litellm.types.router import GenericLiteLLMParams

EDITS_URL: Final = "https://api.minimax.io/v1/images/edits"
PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PROMPT_TEXT_TOKENS: Final = 2950
INPUT_IMAGE_TOKENS: Final = 1024
OUTPUT_IMAGE_TOKENS: Final = 4096


def _edit_response(model: str) -> dict[str, object]:
    return {
        "created": 1794341959,
        "data": [{"url": "https://example.com/h3-image/edit-0.jpg", "size": "1024x1024", "revised_prompt": "night"}],
        "model": model,
        "output_format": "png",
        "usage": {
            "generated_images": 1,
            "total_tokens": PROMPT_TEXT_TOKENS + INPUT_IMAGE_TOKENS + OUTPUT_IMAGE_TOKENS,
            "input_tokens": PROMPT_TEXT_TOKENS + INPUT_IMAGE_TOKENS,
            "output_tokens": OUTPUT_IMAGE_TOKENS,
            "input_tokens_details": {"text_tokens": PROMPT_TEXT_TOKENS, "image_tokens": INPUT_IMAGE_TOKENS},
            "output_tokens_details": {"image_tokens": OUTPUT_IMAGE_TOKENS},
        },
    }


def test_minimax_image_edit_sends_json_image_url():
    config = MinimaxImageEditConfig()
    assert config.use_multipart_form_data() is False
    assert (
        config.get_complete_url(model="MiniMax-H3-Image", api_base=None, litellm_params={})
        == "https://api.minimax.io/v1/images/edits"
    )
    body, files = config.transform_image_edit_request(
        model="MiniMax-H3-Image",
        prompt="make it night",
        image="https://example.com/ref.png",
        image_edit_optional_request_params={"aspect_ratio": "1:1", "area": "1K"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    assert body == {
        "model": "MiniMax-H3-Image",
        "images": [{"image_url": "https://example.com/ref.png"}],
        "aspect_ratio": "1:1",
        "area": "1K",
        "prompt": "make it night",
    }


@pytest.mark.parametrize("model", ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"])
def test_minimax_image_edit_uploads_file_as_data_url_and_bills_its_token_rates(model: str):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response(model))
        response: Final = litellm.image_edit(
            model=f"minimax/{model}",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            aspect_ratio="1:1",
            area="1K",
            output_format="png",
        )

    sent: Final = json.loads(route.calls.last.request.content)
    row: Final = litellm.model_cost[f"minimax/{model}"]
    expected_cost: Final = (
        PROMPT_TEXT_TOKENS * row["input_cost_per_token"]
        + INPUT_IMAGE_TOKENS * row["input_cost_per_image_token"]
        + OUTPUT_IMAGE_TOKENS * row["output_cost_per_image_token"]
    )
    assert sent == {
        "model": model,
        "images": [{"image_url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"}],
        "aspect_ratio": "1:1",
        "area": "1K",
        "output_format": "png",
        "prompt": "make it night",
    }
    assert response.data[0].url == "https://example.com/h3-image/edit-0.jpg"
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("params", "expected_body_params"),
    [
        ({"extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"area": "2K", "extra_body": {"area": "1K"}}, {"area": "1K"}),
        ({"extra_body": {"model": "MiniMax-H3-Image-Highspeed", "seed": 7}}, {"seed": 7}),
    ],
)
def test_minimax_image_edit_sends_extra_body_fields_at_the_top_level(
    params: dict[str, object], expected_body_params: dict[str, object]
):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            **params,
        )

    assert json.loads(route.calls.last.request.content) == {
        "model": "MiniMax-H3-Image",
        "images": [{"image_url": f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"}],
        "prompt": "make it night",
        **expected_body_params,
    }


def test_minimax_image_edit_sends_form_output_compression_as_a_number():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            output_format="jpeg",
            output_compression="80",
        )

    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["output_compression"] == 80


@pytest.mark.parametrize("stream", [True, "true"])
def test_minimax_image_edit_refuses_stream_before_calling_minimax(stream: bool | str):
    with respx.mock(assert_all_called=False) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        with pytest.raises(UnsupportedParamsError) as error:
            litellm.image_edit(
                model="minimax/MiniMax-H3-Image",
                image=PNG_BYTES,
                prompt="make it night",
                api_key="sk-fake",
                stream=stream,
            )

    assert error.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("stream", [False, "false"])
def test_minimax_image_edit_accepts_stream_off(stream: bool | str):
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            stream=stream,
        )

    assert route.call_count == 1
    assert "stream" not in json.loads(route.calls.last.request.content)


def test_minimax_image_edit_drops_stream_under_drop_params():
    with respx.mock(assert_all_called=True) as router:
        route: Final = router.post(EDITS_URL).respond(200, json=_edit_response("MiniMax-H3-Image"))
        litellm.image_edit(
            model="minimax/MiniMax-H3-Image",
            image=PNG_BYTES,
            prompt="make it night",
            api_key="sk-fake",
            stream=True,
            drop_params=True,
        )

    assert route.call_count == 1
    assert "stream" not in json.loads(route.calls.last.request.content)
