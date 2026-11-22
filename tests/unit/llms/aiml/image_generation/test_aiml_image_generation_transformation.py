import os
from datetime import datetime

import httpx
import pytest
from pydantic import ValidationError

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.aiml.image_generation.transformation import (
    AimlImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


def test_openai_style_model_supports_full_openai_param_surface():
    params = AimlImageGenerationConfig().get_supported_openai_params(
        "openai/gpt-image-2"
    )
    assert {
        "n",
        "size",
        "quality",
        "response_format",
        "output_format",
        "background",
        "moderation",
        "output_compression",
    } == set(params)


def test_flux_style_model_keeps_legacy_param_surface():
    assert AimlImageGenerationConfig().get_supported_openai_params("flux-pro/v1.1") == [
        "n",
        "response_format",
        "size",
    ]


def test_openai_style_request_passes_params_through_unchanged():
    """gpt-image-2 must receive OpenAI-shaped fields (size string, n, response_format) verbatim;
    the flux-style remapping to ``num_images``/``image_size``/``output_format`` would break the upstream call.
    """
    config = AimlImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={
            "n": 1,
            "size": "1024x1536",
            "quality": "high",
            "response_format": "b64_json",
            "output_format": "png",
        },
        optional_params={},
        model="openai/gpt-image-2",
        drop_params=False,
    )

    body = config.transform_image_generation_request(
        model="openai/gpt-image-2",
        prompt="A cute baby sea otter",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert body == {
        "model": "openai/gpt-image-2",
        "prompt": "A cute baby sea otter",
        "n": 1,
        "size": "1024x1536",
        "quality": "high",
        "response_format": "b64_json",
        "output_format": "png",
    }


def test_flux_style_request_still_remaps_to_legacy_fields():
    config = AimlImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={
            "n": 2,
            "size": "1024x1024",
            "response_format": "png",
        },
        optional_params={},
        model="flux-pro/v1.1",
        drop_params=False,
    )

    body = config.transform_image_generation_request(
        model="flux-pro/v1.1",
        prompt="hello",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert body["model"] == "flux-pro/v1.1"
    assert body["prompt"] == "hello"
    assert body["num_images"] == 2
    assert body["image_size"] == {"width": 1024, "height": 1024}
    assert body["output_format"] == "png"
    assert "n" not in body
    assert "size" not in body
    assert "response_format" not in body


def test_openai_style_unsupported_param_raises_without_drop_params():
    with pytest.raises(ValueError, match='Supported parameters are'):
        AimlImageGenerationConfig().map_openai_params(
            non_default_params={"image_size": {"width": 1024, "height": 1024}},
            optional_params={},
            model="openai/gpt-image-2",
            drop_params=False,
        )


def test_openai_style_unsupported_param_dropped_with_drop_params():
    mapped = AimlImageGenerationConfig().map_openai_params(
        non_default_params={"image_size": {"width": 1024, "height": 1024}},
        optional_params={},
        model="openai/gpt-image-2",
        drop_params=True,
    )
    assert mapped == {}


def _transform_response(payload: object) -> ImageResponse:
    return AimlImageGenerationConfig().transform_image_generation_response(
        model="flux-pro",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Logging(
            model="flux-pro",
            messages=[],
            stream=False,
            call_type="image_generation",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="aiml-image-generation-test",
            function_id="aiml-image-generation-test",
        ),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "data": [
                    {"url": "https://cdn.aiml.example/a.png", "revised_prompt": "a red fox", "width": 1024},
                    "not-an-image",
                    {"b64_json": "QUJD", "revised_prompt": "a blue fox"},
                    [],
                    {"b64_json": "", "image_base64": "REVG"},
                    {"content_type": "image/png"},
                ]
            },
            [
                (None, "https://cdn.aiml.example/a.png", "a red fox"),
                ("QUJD", None, "a blue fox"),
                ("REVG", None, None),
            ],
        ),
        (
            {
                "output": {
                    "choices": [
                        {"image_base64": "QUJD", "url": "https://cdn.aiml.example/ignored.png"},
                        "skipped",
                        {"url": "https://cdn.aiml.example/b.png"},
                        {"finish_reason": "stop"},
                    ]
                }
            },
            [("QUJD", None, None), (None, "https://cdn.aiml.example/b.png", None)],
        ),
        (
            {
                "images": [
                    {"url": "https://cdn.aiml.example/c.png", "image_base64": "aWdub3JlZA=="},
                    [1, 2],
                    {"image_base64": "REVG"},
                    {"seed": 7},
                ]
            },
            [(None, "https://cdn.aiml.example/c.png", None), ("REVG", None, None)],
        ),
        ({"output": {"choices": {"finish_reason": "stop"}}}, []),
        ({"output": {"choices": "pending"}}, []),
        ({"images": {"count": 0}}, []),
        ({"images": ""}, []),
        ({"data": "pending"}, []),
        ({"id": "gen-1"}, []),
        ([], []),
    ],
)
def test_transform_image_generation_response_reads_images_from_every_response_shape(
    payload: object, expected: list[tuple[str | None, str | None, str | None]]
):
    result = _transform_response(payload)

    assert [(image.b64_json, image.url, image.revised_prompt) for image in result.data] == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [7]},
        {"data": [None]},
        {"data": ["aiml-secret-url-token"]},
        {"data": [["b64_json", "aiml-secret-url-token"]]},
        {"output": {"choices": [False]}},
        {"output": {"choices": ["aiml-secret image_base64"]}},
        {"output": {"choices": None}},
        {"images": [1.5]},
        {"images": ["aiml-secret-url-token"]},
        {"images": {"url": "aiml-secret-url-token"}},
        {"images": 3},
    ],
)
def test_transform_image_generation_response_rejects_malformed_image_entries_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform_response(payload)

    assert "aiml-secret" not in str(exc_info.value)


def _rejection_text(payload: object) -> str:
    with pytest.raises(ValidationError) as exc_info:
        _transform_response(payload)

    return str(exc_info.value)


@pytest.mark.parametrize("position", [403, 429])
@pytest.mark.parametrize("malformed_entry", [None, 7, "url"])
def test_transform_image_generation_response_rejection_text_is_the_same_wherever_the_malformed_entry_sits(
    position: int, malformed_entry: object
):
    first = [malformed_entry]
    late = [*[{"seed": 1}] * position, malformed_entry]

    assert _rejection_text({"data": late}) == _rejection_text({"data": first})
    assert _rejection_text({"output": {"choices": late}}) == _rejection_text({"output": {"choices": first}})
    assert _rejection_text({"images": late}) == _rejection_text({"images": first})
