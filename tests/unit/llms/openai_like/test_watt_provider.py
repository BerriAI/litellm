"""
Tests for the Watt Cloud provider identity and catalog wiring.

Watt Cloud serves an OpenAI-compatible surface (chat completions, responses,
and Anthropic-style messages) and is registered through the JSON provider
mechanism, so the tests below guard the registration records, the request
shape its dynamic config class produces, and the packaged cost catalog.
"""

import json
from pathlib import Path

import pytest

import litellm

WATT_BASE_URL = "https://api.watt-cloud.com/v1"

WATT_CATALOG_MODELS = [
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash-0731",
    "DeepSeek-V4-Pro",
    "DeepSeek-V4.1-Flash",
    "DeepSeek-V4.1-Flash-1017",
    "GLM-5.1",
    "GLM-5.2",
    "GLM-5.3",
    "GLM-5.3-Flash",
    "Hy4-preview",
    "Kimi-K2.5",
    "Kimi-K2.7-Code",
    "Kimi-K3",
    "Kimi-K3.1-Pro",
    "MiniMax-M3",
    "Qwen3.5-122B-A10B",
    "Qwen3.5-27B",
    "Qwen3.5-35B-A3B",
    "Qwen3.5-397B-A17B",
]


class TestWattProviderRegistration:
    def test_watt_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.WATT.value == "watt"
        assert "watt" in litellm.provider_list

    def test_watt_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        watt = JSONProviderRegistry.get("watt")
        assert watt is not None
        assert watt.base_url == WATT_BASE_URL
        assert watt.api_key_env == "WATT_API_KEY"
        assert watt.api_base_env == "WATT_API_BASE"
        assert watt.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert watt.supported_endpoints == [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ]

    def test_watt_in_openai_compatible_lists(self):
        from litellm.constants import (
            openai_compatible_endpoints,
            openai_compatible_providers,
        )

        assert "watt" in openai_compatible_providers
        assert WATT_BASE_URL in openai_compatible_endpoints

    def test_prefixed_model_resolves_to_watt_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="watt/DeepSeek-V4-Pro",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "DeepSeek-V4-Pro"
        assert provider == "watt"
        assert api_base == WATT_BASE_URL

    def test_api_base_autodetects_watt(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("WATT_API_KEY", "sk-watt-env")

        _, provider, api_key, api_base = get_llm_provider(
            model="DeepSeek-V4-Pro",
            custom_llm_provider=None,
            api_base=WATT_BASE_URL,
            api_key=None,
        )

        assert provider == "watt"
        assert api_base == WATT_BASE_URL
        assert api_key == "sk-watt-env"

    def test_env_api_key_is_read_from_watt_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("WATT_API_KEY", "sk-watt-env")

        provider = JSONProviderRegistry.get("watt")
        assert provider is not None

        api_base, api_key = create_config_class(provider)().get_openai_compatible_provider_info(None, None)
        assert api_base == WATT_BASE_URL
        assert api_key == "sk-watt-env"


class TestWattRequestShape:
    def test_chat_url_and_param_mapping(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("watt")
        assert provider is not None

        config = create_config_class(provider)()
        url = config.get_complete_url(
            api_base=WATT_BASE_URL,
            api_key="sk-test",
            model="watt/DeepSeek-V4-Pro",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == f"{WATT_BASE_URL}/chat/completions"

        params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="watt/DeepSeek-V4-Pro",
            drop_params=False,
        )
        assert params.get("max_tokens") == 256
        assert "max_completion_tokens" not in params

    def test_responses_config_url_and_auth(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("watt") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="watt",
            model="watt/DeepSeek-V4-Pro",
        )
        assert config is not None
        assert config.custom_llm_provider == "watt"
        assert config.get_complete_url(api_base=None, litellm_params={}) == f"{WATT_BASE_URL}/responses"

        monkeypatch.setenv("WATT_API_KEY", "sk-env-key")
        headers = config.validate_environment(
            headers={}, model="DeepSeek-V4-Pro", litellm_params=None
        )
        assert headers["Authorization"] == "Bearer sk-env-key"

    def test_messages_config_url_and_auth(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="DeepSeek-V4-Pro",
            provider=litellm.LlmProviders.WATT,
        )
        assert isinstance(cfg, JSONProviderAnthropicMessagesConfig)
        assert (
            cfg.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="DeepSeek-V4-Pro",
                optional_params={},
                litellm_params={},
            )
            == f"{WATT_BASE_URL}/messages"
        )

        monkeypatch.setenv("WATT_API_KEY", "sk-env-key")
        headers, _ = cfg.validate_anthropic_messages_environment(
            headers={},
            model="DeepSeek-V4-Pro",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        assert headers["authorization"] == "Bearer sk-env-key"
        assert headers["anthropic-version"] == "2023-06-01"


class TestWattCatalog:
    def test_every_marketplace_model_has_a_cost_entry(self):
        for model_id in WATT_CATALOG_MODELS:
            key = f"watt/{model_id}"
            assert key in litellm.model_cost, key
            info = litellm.model_cost[key]
            assert info["litellm_provider"] == "watt"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > info["input_cost_per_token"]
            assert info["max_input_tokens"] > 0
            for endpoint in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
                assert endpoint in info["supported_endpoints"]

    def test_catalog_costs_match_the_published_marketplace(self):
        # Source: https://www.watt-cloud.com/models marketplace API, 2026-10-27
        pro = litellm.model_cost["watt/DeepSeek-V4-Pro"]
        assert pro["input_cost_per_token"] == 1.6e-06
        assert pro["cache_read_input_token_cost"] == 1.4e-07
        assert pro["output_cost_per_token"] == 3.38e-06
        assert pro["max_input_tokens"] == 1048576

        flash = litellm.model_cost["watt/DeepSeek-V4-Flash"]
        assert flash["input_cost_per_token"] == 1.4e-07
        assert flash["output_cost_per_token"] == 2.8e-07

    def test_vision_models_declare_image_input(self):
        assert litellm.model_cost["watt/DeepSeek-V4-Pro"].get("supports_vision") is not True
        assert litellm.model_cost["watt/Kimi-K3"]["supports_vision"] is True
        assert litellm.model_cost["watt/GLM-5.3"]["supports_vision"] is not True

    def test_watt_supported_endpoints_matrix(self):
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        endpoints = matrix["providers"]["watt"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True
        assert endpoints["embeddings"] is False
