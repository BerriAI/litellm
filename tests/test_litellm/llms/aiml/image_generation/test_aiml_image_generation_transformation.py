import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")

from litellm.llms.aiml.image_generation.cost_calculator import (
    cost_calculator as aiml_cost_calculator,
)
from litellm.llms.aiml.image_generation.transformation import (
    AimlImageGenerationConfig,
)
from litellm.types.utils import ImageObject, ImageResponse


def test_openai_style_model_supports_full_openai_param_surface():
    params = AimlImageGenerationConfig().get_supported_openai_params(
        "openai/gpt-image-2"
    )
    assert {
        "n",
        "size",
        "quality",
        "response_format",
        "output_format",
        "background",
        "moderation",
        "output_compression",
    } == set(params)


def test_flux_style_model_keeps_legacy_param_surface():
    assert AimlImageGenerationConfig().get_supported_openai_params("flux-pro/v1.1") == [
        "n",
        "response_format",
        "size",
    ]


def test_openai_style_request_passes_params_through_unchanged():
    """gpt-image-2 must receive OpenAI-shaped fields (size string, n, response_format) verbatim;
    the flux-style remapping to ``num_images``/``image_size``/``output_format`` would break the upstream call.
    """
    config = AimlImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={
            "n": 1,
            "size": "1024x1536",
            "quality": "high",
            "response_format": "b64_json",
            "output_format": "png",
        },
        optional_params={},
        model="openai/gpt-image-2",
        drop_params=False,
    )

    body = config.transform_image_generation_request(
        model="openai/gpt-image-2",
        prompt="A cute baby sea otter",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert body == {
        "model": "openai/gpt-image-2",
        "prompt": "A cute baby sea otter",
        "n": 1,
        "size": "1024x1536",
        "quality": "high",
        "response_format": "b64_json",
        "output_format": "png",
    }


def test_flux_style_request_still_remaps_to_legacy_fields():
    config = AimlImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={
            "n": 2,
            "size": "1024x1024",
            "response_format": "png",
        },
        optional_params={},
        model="flux-pro/v1.1",
        drop_params=False,
    )

    body = config.transform_image_generation_request(
        model="flux-pro/v1.1",
        prompt="hello",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert body["model"] == "flux-pro/v1.1"
    assert body["prompt"] == "hello"
    assert body["num_images"] == 2
    assert body["image_size"] == {"width": 1024, "height": 1024}
    assert body["output_format"] == "png"
    assert "n" not in body
    assert "size" not in body
    assert "response_format" not in body


def test_openai_style_unsupported_param_raises_without_drop_params():
    with pytest.raises(ValueError):
        AimlImageGenerationConfig().map_openai_params(
            non_default_params={"image_size": {"width": 1024, "height": 1024}},
            optional_params={},
            model="openai/gpt-image-2",
            drop_params=False,
        )


def test_openai_style_unsupported_param_dropped_with_drop_params():
    mapped = AimlImageGenerationConfig().map_openai_params(
        non_default_params={"image_size": {"width": 1024, "height": 1024}},
        optional_params={},
        model="openai/gpt-image-2",
        drop_params=True,
    )
    assert mapped == {}


def test_cost_calculator_uses_aiml_pricing_for_gpt_image_2():
    """Regression: pricing must come from the ``aiml/openai/gpt-image-2`` entry,
    not the upstream OpenAI token-based entry.
    """
    response = ImageResponse(
        data=[
            ImageObject(b64_json=None, url="https://example.com/1.png"),
            ImageObject(b64_json=None, url="https://example.com/2.png"),
        ]
    )
    assert aiml_cost_calculator(
        model="openai/gpt-image-2", image_response=response
    ) == pytest.approx(0.054 * 2)
