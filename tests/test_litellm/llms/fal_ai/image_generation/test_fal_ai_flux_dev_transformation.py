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
    raw = httpx.Response(200, json={"images": [{"url": "https://fal.media/a.png"}, {"url": "https://fal.media/b.png"}]})
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
