import pytest

import litellm
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.image_generation import (
    FalAIGPTImage2Config,
    FalAINanoBananaConfig,
    get_fal_ai_image_generation_config,
)
from litellm.llms.fal_ai.image_generation.gpt_image_2_transformation import (
    map_gpt_image_quality,
    supported_gpt_image_qualities,
)
from litellm.types.utils import ImageObject, ImageResponse


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-image-2",
        "gpt-image-2",
        "openai/gpt-image-2/edit",
    ],
)
def test_gpt_image_2_config_selected(model):
    assert isinstance(get_fal_ai_image_generation_config(model), FalAIGPTImage2Config)


def test_nano_banana_still_routes_to_nano_banana_config():
    assert isinstance(
        get_fal_ai_image_generation_config("fal-ai/nano-banana"),
        FalAINanoBananaConfig,
    )


@pytest.mark.parametrize(
    "model,expected_url",
    [
        ("openai/gpt-image-2", "https://fal.run/openai/gpt-image-2"),
        ("gpt-image-2", "https://fal.run/openai/gpt-image-2"),
        ("openai/gpt-image-2/edit", "https://fal.run/openai/gpt-image-2/edit"),
    ],
)
def test_get_complete_url_derives_endpoint_from_model(model, expected_url):
    url = FalAIGPTImage2Config().get_complete_url(
        api_base=None,
        api_key="test-key",
        model=model,
        optional_params={},
        litellm_params={},
    )
    assert url == expected_url


def test_get_complete_url_respects_api_base_override():
    url = FalAIGPTImage2Config().get_complete_url(
        api_base="https://proxy.internal/",
        api_key="test-key",
        model="openai/gpt-image-2",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://proxy.internal/openai/gpt-image-2"


@pytest.mark.parametrize(
    "non_default_params,expected",
    [
        ({"n": 3}, {"num_images": 3}),
        ({"size": "1024x1536"}, {"image_size": {"width": 1024, "height": 1536}}),
        ({"size": "auto"}, {"image_size": "auto"}),
        ({"quality": "medium"}, {"quality": "medium"}),
        ({"quality": "hd"}, {"quality": "high"}),
        ({"quality": "standard"}, {"quality": "medium"}),
        ({"quality": "nonsense"}, {"quality": "auto"}),
        ({"output_format": "webp"}, {"output_format": "webp"}),
        ({"response_format": "url"}, {}),
    ],
)
def test_map_openai_params(non_default_params, expected):
    assert (
        FalAIGPTImage2Config().map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="openai/gpt-image-2",
            drop_params=False,
        )
        == expected
    )


def test_map_openai_params_keeps_explicit_provider_params():
    mapped = FalAIGPTImage2Config().map_openai_params(
        non_default_params={"n": 4, "size": "1024x1024"},
        optional_params={"num_images": 1, "image_size": "square_hd"},
        model="openai/gpt-image-2",
        drop_params=False,
    )
    assert mapped == {"num_images": 1, "image_size": "square_hd"}


def test_map_openai_params_raises_on_unsupported_param():
    with pytest.raises(ValueError, match="style"):
        FalAIGPTImage2Config().map_openai_params(
            non_default_params={"style": "vivid"},
            optional_params={},
            model="openai/gpt-image-2",
            drop_params=False,
        )


def test_map_openai_params_drops_unsupported_param():
    assert (
        FalAIGPTImage2Config().map_openai_params(
            non_default_params={"style": "vivid"},
            optional_params={},
            model="openai/gpt-image-2",
            drop_params=True,
        )
        == {}
    )


def test_transform_image_generation_request():
    assert FalAIGPTImage2Config().transform_image_generation_request(
        model="openai/gpt-image-2",
        prompt="a red bicycle",
        optional_params={"quality": "high", "num_images": 2},
        litellm_params={},
        headers={},
    ) == {"prompt": "a red bicycle", "quality": "high", "num_images": 2}


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-image-2.5/flare/text-to-image",
        "openai/gpt-image-2.5/sunburst/text-to-image",
    ],
)
def test_gpt_image_25_routes_to_its_own_fal_endpoint(model):
    config = get_fal_ai_image_generation_config(model)
    assert isinstance(config, FalAIGPTImage2Config)
    assert (
        config.get_complete_url(api_base=None, api_key="k", model=model, optional_params={}, litellm_params={})
        == f"https://fal.run/{model}"
    )


@pytest.mark.parametrize(
    "model,quality,expected",
    [
        ("openai/gpt-image-2.5/flare/text-to-image", "xhigh", "xhigh"),
        ("openai/gpt-image-2.5/sunburst/text-to-image", "max", "max"),
        ("openai/gpt-image-2.5/flare/text-to-image", "hd", "high"),
        ("openai/gpt-image-2", "xhigh", "auto"),
        ("openai/gpt-image-2", "max", "auto"),
    ],
)
def test_map_openai_params_quality_tiers_follow_model(model, quality, expected):
    assert FalAIGPTImage2Config().map_openai_params(
        non_default_params={"quality": quality},
        optional_params={},
        model=model,
        drop_params=False,
    ) == {"quality": expected}


@pytest.mark.parametrize(
    "model",
    [
        "some-new-model",
        "openai/some-new-model",
        "fal_ai/openai/some-new-model",
    ],
)
def test_supported_qualities_derived_from_pricing_rows(model):
    model_cost = {
        "fal_ai/xhigh/1024-x-1024/openai/some-new-model": {},
        "fal_ai/low/1024-x-1024/openai/some-new-model": {},
        "fal_ai/max/1024-x-1024/openai/other-model": {},
    }
    assert supported_gpt_image_qualities(model, model_cost) == {"xhigh", "low", "auto"}


def test_map_gpt_image_quality_passes_through_when_no_pricing_rows():
    assert map_gpt_image_quality("xhigh", "some-new-model", {}) == "xhigh"
