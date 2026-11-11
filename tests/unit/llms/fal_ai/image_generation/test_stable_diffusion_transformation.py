from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.stable_diffusion_transformation import FalAIStableDiffusionConfig
from litellm.types.utils import ImageResponse


def _transform(payload: object) -> ImageResponse:
    return FalAIStableDiffusionConfig().transform_image_generation_response(
        model="fal-ai/stable-diffusion-v35-medium",
        raw_response=httpx.Response(200, json=payload),
        model_response=ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def test_transform_image_generation_response_maps_images_and_metadata():
    response = _transform(
        {
            "images": [{"url": "https://fal.media/a.png", "width": 1024}, "https://fal.media/b.png", 7],
            "seed": 42,
            "timings": {"inference": 2.5},
            "has_nsfw_concepts": [False, False],
        }
    )

    assert [image.url for image in response.data] == ["https://fal.media/a.png", "https://fal.media/b.png"]
    assert response._hidden_params["seed"] == 42
    assert response._hidden_params["timings"] == {"inference": 2.5}
    assert response._hidden_params["has_nsfw_concepts"] == [False, False]


@pytest.mark.parametrize("payload", [{}, {"images": None}, {"images": "https://fal.media/a.png"}])
def test_transform_image_generation_response_without_an_image_list_is_empty(payload: dict[str, object]):
    response = _transform(payload)

    assert response.data == []
    assert "seed" not in response._hidden_params
    assert "timings" not in response._hidden_params
    assert "has_nsfw_concepts" not in response._hidden_params


@pytest.mark.parametrize("payload", [7, "https://fal.media/a.png", [{"url": "https://fal.media/a.png"}]])
def test_transform_image_generation_response_rejects_non_object_bodies(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert "fal.media" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", "square_hd"),
        ("1024x576", "landscape_16_9"),
        ("640x480", {"width": 640, "height": 480}),
        ("wide", "landscape_4_3"),
        ("axb", "landscape_4_3"),
    ],
)
def test_map_openai_params_translates_size_to_image_size(size: str, expected: object):
    optional_params = FalAIStableDiffusionConfig().map_openai_params(
        non_default_params={"size": size, "n": 2, "response_format": "b64_json"},
        optional_params={},
        model="fal-ai/stable-diffusion-v35-medium",
        drop_params=False,
    )

    assert optional_params == {"image_size": expected, "num_images": 2, "output_format": "jpeg"}
