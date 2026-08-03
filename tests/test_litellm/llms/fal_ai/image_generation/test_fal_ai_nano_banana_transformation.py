import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.image_generation import (
    FalAIImagen4Config,
    FalAINanoBananaConfig,
    get_fal_ai_image_generation_config,
)
from litellm.types.utils import ImageObject, ImageResponse


@pytest.mark.parametrize(
    "model",
    [
        "fal-ai/nano-banana",
        "nano-banana",
        "fal-ai/gemini-25-flash-image",
    ],
)
def test_nano_banana_config_selected(model):
    assert isinstance(get_fal_ai_image_generation_config(model), FalAINanoBananaConfig)


def test_imagen4_still_routes_to_imagen4_config():
    assert isinstance(
        get_fal_ai_image_generation_config("fal-ai/imagen4/preview"),
        FalAIImagen4Config,
    )


@pytest.mark.parametrize(
    "model,expected_url",
    [
        ("fal-ai/nano-banana", "https://fal.run/fal-ai/nano-banana"),
        (
            "fal-ai/gemini-25-flash-image",
            "https://fal.run/fal-ai/gemini-25-flash-image",
        ),
        ("nano-banana", "https://fal.run/fal-ai/nano-banana"),
    ],
)
def test_get_complete_url_derives_endpoint_from_model(model, expected_url):
    url = FalAINanoBananaConfig().get_complete_url(
        api_base=None,
        api_key="test-key",
        model=model,
        optional_params={},
        litellm_params={},
    )
    assert url == expected_url


def test_get_complete_url_respects_api_base_override():
    url = FalAINanoBananaConfig().get_complete_url(
        api_base="https://proxy.internal/",
        api_key="test-key",
        model="fal-ai/nano-banana",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://proxy.internal/fal-ai/nano-banana"


def test_map_n_to_num_images():
    optional_params = FalAINanoBananaConfig().map_openai_params(
        non_default_params={"n": 3},
        optional_params={},
        model="fal-ai/nano-banana",
        drop_params=False,
    )
    assert optional_params == {"num_images": 3}


@pytest.mark.parametrize(
    "size,expected_aspect_ratio",
    [
        ("1024x1024", "1:1"),
        ("512x512", "1:1"),
        ("1792x1024", "16:9"),
        ("1024x1792", "9:16"),
        ("1024x768", "4:3"),
        ("768x1024", "3:4"),
    ],
)
def test_map_size_to_aspect_ratio(size, expected_aspect_ratio):
    optional_params = FalAINanoBananaConfig().map_openai_params(
        non_default_params={"size": size},
        optional_params={},
        model="fal-ai/nano-banana",
        drop_params=False,
    )
    assert optional_params == {"aspect_ratio": expected_aspect_ratio}


def test_response_format_is_ignored():
    optional_params = FalAINanoBananaConfig().map_openai_params(
        non_default_params={"response_format": "b64_json"},
        optional_params={},
        model="fal-ai/nano-banana",
        drop_params=False,
    )
    assert optional_params == {}


def test_unsupported_param_raises_without_drop_params():
    with pytest.raises(ValueError):
        FalAINanoBananaConfig().map_openai_params(
            non_default_params={"style": "vivid"},
            optional_params={},
            model="fal-ai/nano-banana",
            drop_params=False,
        )


def test_unsupported_param_dropped_with_drop_params():
    optional_params = FalAINanoBananaConfig().map_openai_params(
        non_default_params={"style": "vivid"},
        optional_params={},
        model="fal-ai/nano-banana",
        drop_params=True,
    )
    assert optional_params == {}


def test_transform_request_includes_prompt_and_mapped_params():
    request = FalAINanoBananaConfig().transform_image_generation_request(
        model="fal-ai/nano-banana",
        prompt="a cat",
        optional_params={"num_images": 2, "aspect_ratio": "16:9"},
        litellm_params={},
        headers={},
    )
    assert request == {
        "prompt": "a cat",
        "num_images": 2,
        "aspect_ratio": "16:9",
    }


@pytest.mark.parametrize(
    "model", ["fal-ai/nano-banana", "fal-ai/gemini-25-flash-image"]
)
def test_nano_banana_pricing_registered(model):
    info = litellm.get_model_info(
        model=model, custom_llm_provider=litellm.LlmProviders.FAL_AI.value
    )
    assert info["output_cost_per_image"] == 0.039
    assert info["mode"] == "image_generation"


def test_cost_calculator_scales_with_image_count():
    image_response = ImageResponse(
        data=[ImageObject(url="https://x/1.png"), ImageObject(url="https://x/2.png")]
    )
    cost = cost_calculator(model="fal-ai/nano-banana", image_response=image_response)
    assert cost == pytest.approx(0.078)
