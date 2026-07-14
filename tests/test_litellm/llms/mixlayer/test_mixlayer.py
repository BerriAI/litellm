"""Tests for the Mixlayer provider (JSON-configured, OpenAI-compatible)."""

import json
import os

# Resolve model pricing from the in-repo cost map rather than the hosted one,
# so these assertions test the entries added in this change. ``setdefault`` so
# an explicit override still wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

import litellm
from litellm import get_llm_provider

MIXLAYER_BASE_URL = "https://models.mixlayer.ai/v1"
MIXLAYER_MODELS = [
    "mixlayer/qwen/qwen3.5-4b-free",
    "mixlayer/qwen/qwen3.5-9b",
    "mixlayer/qwen/qwen3.5-35b-a3b",
    "mixlayer/qwen/qwen3.5-397b-a17b",
    "mixlayer/qwen/qwen3.6-27b",
    "mixlayer/qwen/qwen3.6-35b-a3b",
    "mixlayer/z-ai/glm-5.2",
    "mixlayer/moonshotai/kimi-k2.7-code",
]


class TestMixlayerProvider:
    def test_mixlayer_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("mixlayer")

        mixlayer = JSONProviderRegistry.get("mixlayer")
        assert mixlayer is not None
        assert mixlayer.base_url == MIXLAYER_BASE_URL
        assert mixlayer.api_key_env == "MIXLAYER_API_KEY"

    def test_mixlayer_provider_resolution(self):
        """Provider is resolved from the model prefix, with the default api_base."""
        _, provider, _, api_base = get_llm_provider(
            model="mixlayer/qwen/qwen3.5-9b",
            api_key="fake-mixlayer-key",
        )
        assert provider == "mixlayer"
        assert api_base == MIXLAYER_BASE_URL

    def test_mixlayer_api_base_override(self):
        _, provider, _, api_base = get_llm_provider(
            model="mixlayer/qwen/qwen3.5-9b",
            api_key="fake-mixlayer-key",
            api_base="https://custom.mixlayer.ai/v1",
        )
        assert provider == "mixlayer"
        assert api_base == "https://custom.mixlayer.ai/v1"

    def test_mixlayer_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MIXLAYER_API_KEY", "env-mixlayer-key")
        _, provider, dynamic_api_key, _ = get_llm_provider(model="mixlayer/qwen/qwen3.5-9b")
        assert provider == "mixlayer"
        assert dynamic_api_key == "env-mixlayer-key"

    @pytest.mark.parametrize("model", MIXLAYER_MODELS)
    def test_mixlayer_models_in_pricing_map(self, model):
        """Every published Mixlayer model has a pricing entry."""
        info = litellm.get_model_info(model=model)
        assert info["litellm_provider"] == "mixlayer"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] >= 0
        assert info["output_cost_per_token"] >= 0
        assert info["max_input_tokens"] > 0

    @pytest.mark.parametrize("model", MIXLAYER_MODELS)
    def test_mixlayer_capabilities(self, model):
        """
        Capabilities verified against the live Mixlayer API: all models do
        tool calling and reasoning; the Qwen family accepts image input while
        GLM and Kimi do not; and `tool_choice` is accepted but ignored by the
        API, so it is declared unsupported.
        """
        info = litellm.get_model_info(model=model)
        assert info["supports_function_calling"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is False
        # Vision is supported on the Qwen family only.
        assert info["supports_vision"] is ("qwen/" in model)

    def test_mixlayer_pricing_files_agree(self):
        """The pricing map and its backup copy must not drift."""
        with open("model_prices_and_context_window.json") as f:
            main = json.load(f)
        with open("litellm/model_prices_and_context_window_backup.json") as f:
            backup = json.load(f)

        for model in MIXLAYER_MODELS:
            key = model.replace("mixlayer/", "mixlayer/", 1)
            assert main[key] == backup[key]

    def test_mixlayer_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "mixlayer-chat",
                    "litellm_params": {
                        "model": "mixlayer/qwen/qwen3.5-9b",
                        "api_key": "fake-mixlayer-key",
                    },
                }
            ]
        )
        assert router.model_list[0]["model_name"] == "mixlayer-chat"
