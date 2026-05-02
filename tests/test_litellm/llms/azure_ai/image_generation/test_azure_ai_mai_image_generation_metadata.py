"""
Test Azure AI MAI image generation model metadata and pricing.
"""

import json
from importlib.resources import files
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


@pytest.mark.parametrize(
    "model_name,output_cost_per_image_token",
    [
        ("azure_ai/MAI-Image-2", 3.3e-05),
        ("azure_ai/MAI-Image-2e", 1.95e-05),
    ],
)
def test_azure_ai_mai_image_model_info(
    use_local_model_cost_map, model_name: str, output_cost_per_image_token: float
):
    model_info = use_local_model_cost_map.get_model_info(model=model_name)

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "image_generation"
    assert model_info["max_input_tokens"] == 32000
    assert model_info["max_tokens"] == 32000
    assert model_info["input_cost_per_token"] == pytest.approx(5e-06)
    assert model_info["output_cost_per_image_token"] == pytest.approx(
        output_cost_per_image_token
    )


@pytest.mark.parametrize(
    "model_name", ["azure_ai/MAI-Image-2", "azure_ai/MAI-Image-2e"]
)
def test_azure_ai_mai_image_raw_model_cost_entry(
    use_local_model_cost_map, model_name: str
):
    model_info = use_local_model_cost_map.model_cost[model_name]

    assert model_info["supported_endpoints"] == ["/v1/images/generations"]
    assert model_info["supported_modalities"] == ["text"]
    assert model_info["supported_output_modalities"] == ["image"]


@pytest.mark.parametrize(
    "model_name,expected_total_cost",
    [
        ("MAI-Image-2", 0.038792),
        ("MAI-Image-2e", 0.024968),
    ],
)
def test_azure_ai_mai_image_cost_calculator(
    use_local_model_cost_map, model_name: str, expected_total_cost: float
):
    from litellm.llms.azure_ai.image_generation.cost_calculator import cost_calculator
    from litellm.types.utils import (
        ImageResponse,
        ImageUsage,
        ImageUsageInputTokensDetails,
    )

    image_response = ImageResponse(
        created=123,
        data=[{"b64_json": "abc", "revised_prompt": "A photorealistic mountain lake"}],
        usage=ImageUsage(
            input_tokens=1000,
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=0,
                text_tokens=1000,
            ),
            output_tokens=1024,
            total_tokens=2024,
        ),
    )

    total_cost = cost_calculator(model=model_name, image_response=image_response)

    assert total_cost == pytest.approx(expected_total_cost)


@pytest.mark.parametrize("model_name", ["MAI-Image-2", "MAI-Image-2e"])
def test_azure_ai_mai_image_generation_config(model_name: str):
    from litellm.llms.azure_ai.image_generation import (
        AzureFoundryMAIImageGenerationConfig,
        get_azure_ai_image_generation_config,
    )

    config = get_azure_ai_image_generation_config(model_name)

    assert isinstance(config, AzureFoundryMAIImageGenerationConfig)
    assert (
        config.get_complete_url(
            model=model_name,
            api_base="https://example.services.ai.azure.com",
            api_key=None,
            optional_params={},
            litellm_params={"api_version": "preview"},
        )
        == "https://example.services.ai.azure.com/mai/v1/images/generations?api-version=preview"
    )


def test_azure_ai_mai_image_generation_request_maps_size():
    from litellm.llms.azure_ai.image_generation.mai_transformation import (
        AzureFoundryMAIImageGenerationConfig,
    )

    request = AzureFoundryMAIImageGenerationConfig().transform_image_generation_request(
        model="MAI-Image-2e",
        prompt="A glowing jellyfish in a glass ocean",
        optional_params={"size": "768x1024"},
        litellm_params={},
        headers={},
    )

    assert request == {
        "model": "MAI-Image-2e",
        "prompt": "A glowing jellyfish in a glass ocean",
        "width": 768,
        "height": 1024,
    }


@patch("litellm.images.main.azure_chat_completions.image_generation")
@patch("litellm.images.main.llm_http_handler.image_generation_handler")
def test_azure_ai_mai_image_generation_routes_through_http_handler(
    mock_image_generation_handler, mock_azure_image_generation
):
    import litellm
    from litellm.images.main import image_generation
    from litellm.llms.azure_ai.image_generation.mai_transformation import (
        AzureFoundryMAIImageGenerationConfig,
    )

    mock_response = litellm.ImageResponse(
        created=123,
        data=[{"b64_json": "abc", "revised_prompt": "A bright desert sunrise"}],
    )
    mock_image_generation_handler.return_value = mock_response

    response = image_generation(
        model="azure_ai/MAI-Image-2e",
        prompt="A bright desert sunrise",
        api_base="https://example.services.ai.azure.com",
        api_key="test-key",
        size="1024x1024",
    )

    assert response == mock_response
    mock_azure_image_generation.assert_not_called()
    mock_image_generation_handler.assert_called_once()
    call_kwargs = mock_image_generation_handler.call_args.kwargs
    assert isinstance(
        call_kwargs["image_generation_provider_config"],
        AzureFoundryMAIImageGenerationConfig,
    )
    assert call_kwargs["image_generation_optional_request_params"] == {
        "size": "1024x1024"
    }
    assert call_kwargs["litellm_params"]["api_base"] == (
        "https://example.services.ai.azure.com"
    )
