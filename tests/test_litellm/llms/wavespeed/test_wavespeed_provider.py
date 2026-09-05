"""Tests for WaveSpeed AI provider registration across chat, image, and video surfaces."""

import httpx
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.wavespeed.image_generation.transformation import (
    WaveSpeedImageGenerationConfig,
)
from litellm.llms.wavespeed.videos.transformation import WaveSpeedVideoConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_wavespeed_is_a_known_provider():
    assert LlmProviders.WAVESPEED.value == "wavespeed"
    assert "wavespeed" in litellm.provider_list


def test_chat_json_registry_entry():
    from litellm.constants import openai_compatible_providers

    config = JSONProviderRegistry.get("wavespeed")
    assert config is not None
    assert config.base_url == "https://llm.wavespeed.ai/v1"
    assert config.api_key_env == "WAVESPEED_API_KEY"
    assert config.api_base_env == "WAVESPEED_API_BASE"
    assert "wavespeed" in openai_compatible_providers


def test_upstream_model_prefix_is_preserved(monkeypatch):
    """WaveSpeed chat model ids are themselves `{provider}/{model}`, so only the routing prefix is stripped."""
    model, provider, api_key, api_base = get_llm_provider(
        model="wavespeed/anthropic/claude-opus-4.8",
        custom_llm_provider=None,
        api_base=None,
        api_key="sk-test",
    )

    assert model == "anthropic/claude-opus-4.8"
    assert provider == "wavespeed"
    assert api_base == "https://llm.wavespeed.ai/v1"


def test_media_model_routes_to_wavespeed():
    model, provider, _, _ = get_llm_provider(
        model="wavespeed/bytedance/seedance-2.5/text-to-video",
        custom_llm_provider=None,
        api_base=None,
        api_key="sk-test",
    )

    assert provider == "wavespeed"
    assert model == "bytedance/seedance-2.5/text-to-video"


def test_image_and_video_configs_are_resolved():
    image_config = ProviderConfigManager.get_provider_image_generation_config(
        model="bytedance/seedream-v5.0-pro", provider=LlmProviders.WAVESPEED
    )
    video_config = ProviderConfigManager.get_provider_video_config(
        model="bytedance/seedance-2.5/text-to-video", provider=LlmProviders.WAVESPEED
    )

    assert isinstance(image_config, WaveSpeedImageGenerationConfig)
    assert isinstance(video_config, WaveSpeedVideoConfig)


def test_api_base_autodetects_the_provider(monkeypatch):
    """Pointing api_base at the WaveSpeed chat host is enough to route there."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env-key")

    _, provider, api_key, _ = get_llm_provider(
        model="glm-5",
        custom_llm_provider=None,
        api_base="https://llm.wavespeed.ai/v1",
        api_key=None,
    )

    assert provider == "wavespeed"
    assert api_key == "sk-env-key"


def test_api_key_and_base_resolved_from_env(monkeypatch):
    monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env-key")
    monkeypatch.setenv("WAVESPEED_API_BASE", "https://proxy.internal/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="wavespeed/deepseek/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "wavespeed"
    assert api_key == "sk-env-key"
    assert api_base == "https://proxy.internal/v1"


@respx.mock
def test_image_generation_routes_through_the_public_sdk(monkeypatch):
    """litellm.image_generation dispatches wavespeed models to the polling handler."""
    monkeypatch.setenv("WAVESPEED_API_KEY", "sk-test")
    monkeypatch.delenv("WAVESPEED_API_BASE", raising=False)
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.DEFAULT_POLLING_INTERVAL", 0)

    model = "wavespeed-ai/z-image/turbo"
    output_url = "https://cdn.wavespeed.ai/pred-123.png"
    envelope = {"code": 200, "message": "ok", "data": {"id": "pred-123", "status": "created"}}
    completed = {
        "code": 200,
        "message": "ok",
        "data": {"id": "pred-123", "status": "completed", "outputs": [output_url]},
    }

    submit = respx.post(f"https://api.wavespeed.ai/api/v3/{model}").mock(
        return_value=httpx.Response(200, json=envelope)
    )
    respx.get("https://api.wavespeed.ai/api/v3/predictions/pred-123/result").mock(
        return_value=httpx.Response(200, json=completed)
    )

    response = litellm.image_generation(model=f"wavespeed/{model}", prompt="a red panda")

    assert response.data[0].url == output_url
    assert submit.call_count == 1
