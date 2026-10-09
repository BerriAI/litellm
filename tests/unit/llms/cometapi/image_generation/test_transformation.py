from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.cometapi.image_generation.transformation import CometAPIImageGenerationConfig
from litellm.types.utils import ImageResponse


def _transform(payload: object) -> ImageResponse:
    return CometAPIImageGenerationConfig().transform_image_generation_response(
        model="dall-e-3",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_each_image_object():
    response = _transform({"created": 1, "data": [{"url": "https://img.cometapi.com/a.png"}, {"b64_json": "QUJD"}, {}]})

    assert [(image.url, image.b64_json) for image in response.data] == [
        ("https://img.cometapi.com/a.png", None),
        (None, "QUJD"),
        (None, None),
    ]


@pytest.mark.parametrize("payload", [{}, {"created": 1}, {"data": []}, {"data": ""}, {"data": {}}, []])
def test_transform_image_generation_response_without_images_has_no_data(payload: object):
    assert _transform(payload).data == []


@pytest.mark.parametrize(
    "payload",
    [
        ["data", "https://img.cometapi.com/a.png"],
        {"data": None},
        {"data": "https://img.cometapi.com/a.png"},
        {"data": {"url": "https://img.cometapi.com/a.png"}},
        {"data": ["https://img.cometapi.com/a.png"]},
        {"data": [{"url": "https://img.cometapi.com/a.png"}, 7]},
    ],
)
def test_transform_image_generation_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "img.cometapi.com" not in str(exc_info.value)
