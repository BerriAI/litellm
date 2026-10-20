from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.recraft_v3_transformation import FalAIRecraftV3Config
from litellm.types.utils import ImageResponse


def _transform(payload: object) -> ImageResponse:
    return FalAIRecraftV3Config().transform_image_generation_response(
        model="fal-ai/recraft/v3/text-to-image",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_file_objects_and_bare_urls():
    response = _transform(
        {"images": [{"url": "https://fal.media/a.png", "content_type": "image/png"}, "https://fal.media/b.png", 7]}
    )

    assert [image.url for image in response.data] == ["https://fal.media/a.png", "https://fal.media/b.png"]
    assert [image.b64_json for image in response.data] == [None, None]


@pytest.mark.parametrize("payload", [{}, {"images": None}, {"images": "https://fal.media/a.png"}])
def test_transform_image_generation_response_without_an_image_list_is_empty(payload: dict[str, object]):
    assert _transform(payload).data == []


@pytest.mark.parametrize("payload", [7, "https://fal.media/a.png", [{"url": "https://fal.media/a.png"}]])
def test_transform_image_generation_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "fal.media" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", "square_hd"),
        ("576x1024", "portrait_16_9"),
        ("640x480", {"width": 640, "height": 480}),
        ("wide", "square_hd"),
        ("axb", "square_hd"),
    ],
)
def test_map_openai_params_translates_size_to_image_size(size: str, expected: object):
    optional_params = FalAIRecraftV3Config().map_openai_params(
        non_default_params={"size": size, "n": 2, "response_format": "url"},
        optional_params={},
        model="fal-ai/recraft/v3/text-to-image",
        drop_params=False,
    )

    assert optional_params == {"image_size": expected}
