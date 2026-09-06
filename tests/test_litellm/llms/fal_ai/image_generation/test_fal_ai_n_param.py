"""Bria and Recraft v3 generate one image per call.

Both used to list "n" as supported and then skip over it, so n=3 returned a
single image and the caller was never told. Not listing it lets the normal
unsupported-param path do its job.

Contrast: nano_banana and ideogram_v3 do map n to num_images, see
test_fal_ai_nano_banana_transformation.py.
"""

import pytest

import litellm
from litellm.llms.fal_ai.image_generation.bria_transformation import FalAIBriaConfig
from litellm.llms.fal_ai.image_generation.recraft_v3_transformation import (
    FalAIRecraftV3Config,
)

CONFIGS = [
    (FalAIBriaConfig, "bria/text-to-image/3.2"),
    (FalAIRecraftV3Config, "fal-ai/recraft/v3/text-to-image"),
]


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_n_not_claimed(config_cls, model):
    assert "n" not in config_cls().get_supported_openai_params(model)


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_n_raises_instead_of_being_skipped(config_cls, model):
    with pytest.raises(ValueError, match="not supported"):
        config_cls().map_openai_params(
            non_default_params={"n": 3},
            optional_params={},
            model=model,
            drop_params=False,
        )


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_n_still_droppable(config_cls, model):
    params = config_cls().map_openai_params(
        non_default_params={"n": 3},
        optional_params={},
        model=model,
        drop_params=True,
    )
    assert "n" not in params
    assert "num_images" not in params


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_size_still_maps(config_cls, model):
    """The params these models do serve must keep working."""
    params = config_cls().map_openai_params(
        non_default_params={"size": "1024x1024"},
        optional_params={},
        model=model,
        drop_params=False,
    )
    # bria calls it aspect_ratio, recraft calls it image_size
    assert params in ({"aspect_ratio": "1:1"}, {"image_size": "square_hd"})


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_n_raises_through_get_optional_params_image_gen(config_cls, model):
    """The path a caller actually takes.

    map_openai_params is not reached with an unsupported param: the check in
    get_optional_params_image_gen runs first and raises UnsupportedParamsError.
    Testing only the config in isolation would miss which error users see.
    """
    from litellm.utils import get_optional_params_image_gen

    with pytest.raises(litellm.UnsupportedParamsError, match="`n` is not supported"):
        get_optional_params_image_gen(
            model=model,
            custom_llm_provider="fal_ai",
            provider_config=config_cls(),
            n=3,
        )


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_n_dropped_end_to_end_with_drop_params(config_cls, model):
    from litellm.utils import get_optional_params_image_gen

    params = get_optional_params_image_gen(
        model=model,
        custom_llm_provider="fal_ai",
        provider_config=config_cls(),
        n=3,
        drop_params=True,
    )
    assert params == {}


@pytest.mark.parametrize("config_cls,model", CONFIGS)
def test_size_still_maps_end_to_end(config_cls, model):
    from litellm.utils import get_optional_params_image_gen

    params = get_optional_params_image_gen(
        model=model,
        custom_llm_provider="fal_ai",
        provider_config=config_cls(),
        size="1024x1024",
    )
    assert params in ({"aspect_ratio": "1:1"}, {"image_size": "square_hd"})
