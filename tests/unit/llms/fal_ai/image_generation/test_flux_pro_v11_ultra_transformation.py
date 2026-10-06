from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.fal_ai.image_generation.flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig
from litellm.types.utils import ImageResponse


def _transform(payload: object) -> ImageResponse:
    return FalAIFluxProV11UltraConfig().transform_image_generation_response(
        model="fal-ai/flux-pro/v1.1-ultra",
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
            "images": [{"url": "https://fal.media/a.png", "width": 2752, "height": 1536}, "https://fal.media/b.png"],
            "seed": 42,
            "timings": {"inference": 2.5},
            "has_nsfw_concepts": [False, False],
        }
    )

    assert [image.url for image in response.data] == ["https://fal.media/a.png", "https://fal.media/b.png"]
    assert response.data[0].provider_specific_fields == {"width": 2752, "height": 1536}
    assert response._hidden_params["seed"] == 42
    assert response._hidden_params["timings"] == {"inference": 2.5}
    assert response._hidden_params["has_nsfw_concepts"] == [False, False]


@pytest.mark.parametrize("payload", [{}, {"images": None}, {"images": []}])
def test_transform_image_generation_response_without_images_is_empty(payload: dict[str, object]):
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
