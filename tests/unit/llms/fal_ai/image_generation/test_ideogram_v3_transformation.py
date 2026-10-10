from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.ideogram_v3_transformation import FalAIIdeogramV3Config
from litellm.types.utils import ImageResponse


def _transform(payload: object) -> ImageResponse:
    return FalAIIdeogramV3Config().transform_image_generation_response(
        model="fal-ai/ideogram/v3",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_files_and_seed():
    response = _transform(
        {"images": [{"url": "https://fal.media/a.png", "file_name": "a.png"}, "https://fal.media/b.png"], "seed": 42}
    )

    assert [image.url for image in response.data] == ["https://fal.media/a.png", "https://fal.media/b.png"]
    assert [image.b64_json for image in response.data] == [None, None]
    assert response._hidden_params["seed"] == 42


@pytest.mark.parametrize("payload", [{}, {"images": None}, {"images": "https://fal.media/a.png"}])
def test_transform_image_generation_response_without_an_image_list_is_empty(payload: dict[str, object]):
    response = _transform(payload)

    assert response.data == []
    assert "seed" not in response._hidden_params


@pytest.mark.parametrize("payload", [7, "https://fal.media/a.png", [{"url": "https://fal.media/a.png"}]])
def test_transform_image_generation_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "fal.media" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", "square_hd"),
        (" 1536x1024 ", "landscape_16_9"),
        ("640x480", {"width": 640, "height": 480}),
        ("wide", "square_hd"),
        ("axb", "square_hd"),
        ({"width": 640, "height": 480, "unit": "px"}, {"width": 640, "height": 480}),
        ({"width": "640"}, {"width": "640"}),
        (512, 512),
    ],
)
def test_map_openai_params_translates_size_to_image_size(size: object, expected: object):
    optional_params = FalAIIdeogramV3Config().map_openai_params(
        non_default_params={"size": size, "n": 2, "response_format": "url"},
        optional_params={},
        model="fal-ai/ideogram/v3",
        drop_params=False,
    )

    assert optional_params == {"image_size": expected, "num_images": 2}
