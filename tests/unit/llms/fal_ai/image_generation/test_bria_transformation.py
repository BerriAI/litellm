from datetime import datetime
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.fal_ai.image_generation.bria_transformation import FalAIBriaConfig
from litellm.types.utils import ImageResponse

BRIA_MODEL: Final = "bria/text-to-image/3.2"
IMAGE_URL: Final = "https://fal.media/a.png"


def _transform(payload: object) -> ImageResponse:
    return FalAIBriaConfig().transform_image_generation_response(
        model=BRIA_MODEL,
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Logging(
            model=BRIA_MODEL,
            messages=[{"role": "user", "content": "a red fox"}],
            stream=False,
            call_type="image_generation",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="bria-call",
            function_id="bria-function",
        ),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_the_single_image():
    response: Final = _transform(
        {"image": {"url": IMAGE_URL, "content_type": "image/png", "width": 1024, "height": 1024}, "seed": 7}
    )

    assert [(image.url, image.b64_json) for image in response.data] == [(IMAGE_URL, None)]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"image": None},
        {"image": {}},
        {"image": IMAGE_URL},
        {"image": [{"url": IMAGE_URL}]},
        {"images": [IMAGE_URL]},
    ],
)
def test_transform_image_generation_response_without_an_image_object_is_empty(payload: dict[str, object]):
    response: Final = _transform(payload)

    assert response.data == []


@pytest.mark.parametrize("payload", [IMAGE_URL, [IMAGE_URL], [{"image": {"url": IMAGE_URL}}]])
def test_transform_image_generation_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert IMAGE_URL not in str(exc_info.value)
