import base64
import io

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.fal_ai.image_edit import (
    FalAIFluxLoraDepthEditConfig,
    FalAIImageEditConfig,
    get_fal_ai_image_edit_config,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
MODEL = "fal-ai/flux-lora-depth"


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", ["fal-ai/flux-lora-depth", "flux-lora-depth", "fal_ai/fal-ai/flux-lora-depth"])
def test_dispatch_selects_flux_lora_depth_config(model):
    assert isinstance(get_fal_ai_image_edit_config(model), FalAIFluxLoraDepthEditConfig)


def test_dispatch_keeps_gpt_image_config_for_openai_edit_models():
    config = get_fal_ai_image_edit_config("openai/gpt-image-2.5/flare/edit")
    assert type(config) is FalAIImageEditConfig


def test_provider_config_manager_resolves_flux_lora_depth():
    config = ProviderConfigManager.get_provider_image_edit_config(model=MODEL, provider=LlmProviders.FAL_AI)
    assert isinstance(config, FalAIFluxLoraDepthEditConfig)


@pytest.mark.parametrize("model", ["fal-ai/flux-lora-depth", "flux-lora-depth"])
def test_get_complete_url_targets_endpoint_without_edit_suffix(model):
    url = FalAIFluxLoraDepthEditConfig().get_complete_url(model=model, api_base=None, litellm_params={})
    assert url == "https://fal.run/fal-ai/flux-lora-depth"


def test_get_supported_openai_params_excludes_quality_mask_background():
    params = FalAIFluxLoraDepthEditConfig().get_supported_openai_params(model=MODEL)
    assert "quality" not in params
    assert "mask" not in params
    assert "background" not in params


def test_map_openai_params_translates_n_and_size():
    mapped = FalAIFluxLoraDepthEditConfig().map_openai_params(
        image_edit_optional_params=ImageEditOptionalRequestParams(n=2, size="1024x1536", quality="high"),
        model=MODEL,
        drop_params=False,
    )
    assert mapped == {"num_images": 2, "image_size": {"width": 1024, "height": 1536}}


def test_transform_request_sends_single_image_url_as_data_url():
    body, files = FalAIFluxLoraDepthEditConfig().transform_image_edit_request(
        model=MODEL,
        prompt="follow the depth map",
        image=io.BytesIO(PNG_BYTES),
        image_edit_optional_request_params={"num_images": 1},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == ()
    assert body["prompt"] == "follow the depth map"
    assert body["image_url"] == "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    assert "image_urls" not in body
    assert body["num_images"] == 1


def test_transform_request_passes_remote_url_through_untouched():
    body, _ = FalAIFluxLoraDepthEditConfig().transform_image_edit_request(
        model=MODEL,
        prompt="follow the depth map",
        image="https://example.com/depth.png",
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert body["image_url"] == "https://example.com/depth.png"


def test_transform_request_rejects_two_images():
    with pytest.raises(ValueError, match="exactly one control image"):
        FalAIFluxLoraDepthEditConfig().transform_image_edit_request(
            model=MODEL,
            prompt="follow the depth map",
            image=["https://example.com/a.png", "https://example.com/b.png"],
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_image_edit_cost_uses_flat_output_cost_per_image():
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=MODEL,
        completion_response=ImageResponse(data=[ImageObject(url="https://example.com/out.png")]),
        custom_llm_provider="fal_ai",
        optional_params={},
        call_type="aimage_edit",
    )
    assert cost == litellm.model_cost[f"fal_ai/{MODEL}"]["output_cost_per_image"] > 0
