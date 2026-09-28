import httpx
import pytest

from litellm.llms.fal_ai.image_generation import (
    FalAIFluxDevConfig,
    FalAIFluxSchnellConfig,
    FalAIImageGenerationConfig,
    get_fal_ai_image_generation_config,
)
from litellm.types.utils import ImageResponse


@pytest.mark.parametrize("model", ["fal-ai/flux/dev", "flux/dev", "flux-dev"])
def test_flux_dev_config_selected(model):
    config = get_fal_ai_image_generation_config(model)
    assert isinstance(config, FalAIFluxDevConfig)
    assert not isinstance(config, FalAIImageGenerationConfig)


def test_flux_schnell_still_routes_to_schnell():
    config = get_fal_ai_image_generation_config("fal-ai/flux/schnell")
    assert isinstance(config, FalAIFluxSchnellConfig)
    assert not isinstance(config, FalAIFluxDevConfig)


def test_flux_dev_url_targets_dev_endpoint():
    url = FalAIFluxDevConfig().get_complete_url(
        api_base=None, api_key="k", model="fal-ai/flux/dev", optional_params={}, litellm_params={}
    )
    assert url == "https://fal.run/fal-ai/flux/dev"


def test_flux_dev_maps_openai_params_and_builds_request():
    config = FalAIFluxDevConfig()
    optional_params = config.map_openai_params(
        non_default_params={"n": 2, "size": "1024x1024", "response_format": "b64_json"},
        optional_params={},
        model="fal-ai/flux/dev",
        drop_params=False,
    )
    body = config.transform_image_generation_request(
        model="fal-ai/flux/dev", prompt="a cat", optional_params=optional_params, litellm_params={}, headers={}
    )
    assert body["prompt"] == "a cat"
    assert body["num_images"] == 2
    assert body["image_size"] == "square_hd"


def test_flux_dev_response_yields_one_image_object_per_fal_image():
    raw = httpx.Response(
        200,
        json={
            "images": [
                {"url": "https://fal.media/a.png", "width": 1024, "height": 768, "content_type": "image/png"},
                {"url": "https://fal.media/b.png", "width": 512, "height": 512, "content_type": "image/webp"},
            ]
        },
    )
    response = FalAIFluxDevConfig().transform_image_generation_response(
        model="fal-ai/flux/dev",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert [image.url for image in response.data] == ["https://fal.media/a.png", "https://fal.media/b.png"]
    assert [image.provider_specific_fields for image in response.data] == [
        {"width": 1024, "height": 768, "content_type": "image/png"},
        {"width": 512, "height": 512, "content_type": "image/webp"},
    ]


def test_flux_dev_response_omits_provider_specific_fields_when_fal_omits_metadata():
    raw = httpx.Response(200, json={"images": [{"url": "https://fal.media/a.png"}]})
    response = FalAIFluxDevConfig().transform_image_generation_response(
        model="fal-ai/flux/dev",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data[0].provider_specific_fields is None


@pytest.mark.parametrize(
    "invalid_field, invalid_value, expected_fields",
    (
        ("width", True, {"height": 768, "content_type": "image/png"}),
        ("width", 0, {"height": 768, "content_type": "image/png"}),
        ("width", -1, {"height": 768, "content_type": "image/png"}),
        ("height", True, {"width": 1024, "content_type": "image/png"}),
        ("height", 0, {"width": 1024, "content_type": "image/png"}),
        ("height", -1, {"width": 1024, "content_type": "image/png"}),
    ),
)
def test_flux_dev_response_drops_invalid_dimension_metadata(invalid_field, invalid_value, expected_fields):
    metadata = {"width": 1024, "height": 768, "content_type": "image/png"}
    metadata[invalid_field] = invalid_value
    raw = httpx.Response(
        200,
        json={
            "images": [
                {
                    "url": "https://fal.media/a.png",
                    **metadata,
                }
            ]
        },
    )
    response = FalAIFluxDevConfig().transform_image_generation_response(
        model="fal-ai/flux/dev",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=None,
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data[0].provider_specific_fields == expected_fields
