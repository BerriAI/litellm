"""
Tests for the OpenInfer LLM provider configuration and integration.
"""

import json
from pathlib import Path

import pytest

import litellm

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRICE_FILES = (
    _REPO_ROOT / "model_prices_and_context_window.json",
    _REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)


class TestOpenInferProviderConfig:
    def test_openinfer_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.OPENINFER.value == "openinfer"
        assert "openinfer" in litellm.provider_list

    def test_openinfer_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("openinfer")
        assert provider is not None
        assert provider.base_url == "https://api.openinfer.ai/v1"
        assert provider.api_key_env == "OPENINFER_API_KEY"
        assert provider.api_base_env == "OPENINFER_API_BASE"
        assert not JSONProviderRegistry.supports_responses_api("openinfer")

    def test_openinfer_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "openinfer" in openai_compatible_providers

    def test_provider_prefixed_model_routes_to_openinfer(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openinfer/@oi/Llama-3.2-1B-Instruct",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "@oi/Llama-3.2-1B-Instruct"
        assert provider == "openinfer"
        assert api_key == "sk-test"
        assert api_base == "https://api.openinfer.ai/v1"

    def test_api_key_and_base_resolved_from_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")
        monkeypatch.setenv("OPENINFER_API_BASE", "https://proxy.internal/v1")

        _, provider, api_key, api_base = get_llm_provider(
            model="openinfer/@oi/Qwen3.5-9B",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "openinfer"
        assert api_key == "sk-env-key"
        assert api_base == "https://proxy.internal/v1"

    def test_url_autodetection_from_api_base(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")

        _, provider, api_key, api_base = get_llm_provider(
            model="@oi/Llama-3.2-1B-Instruct",
            custom_llm_provider=None,
            api_base="https://api.openinfer.ai/v1",
            api_key=None,
        )

        assert provider == "openinfer"
        assert api_key == "sk-env-key"

    def test_chat_completions_url(self):
        config = litellm.ProviderConfigManager.get_provider_chat_config(
            model="@oi/Llama-3.2-1B-Instruct", provider=litellm.LlmProviders.OPENINFER
        )
        assert config is not None
        assert (
            config.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="@oi/Llama-3.2-1B-Instruct",
                optional_params={},
                litellm_params={},
            )
            == "https://api.openinfer.ai/v1/chat/completions"
        )

    def test_outgoing_chat_request_contract(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        model, provider, api_key, api_base = get_llm_provider(
            model="openinfer/@oi/Llama-3.2-1B-Instruct",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )
        assert model == "@oi/Llama-3.2-1B-Instruct"
        assert provider == "openinfer"
        assert api_key == "sk-test"
        assert api_base == "https://api.openinfer.ai/v1"

        provider_cfg = JSONProviderRegistry.get("openinfer")
        assert provider_cfg is not None
        config = create_config_class(provider_cfg)()

        assert (
            config.get_complete_url(
                api_base=None,
                api_key=api_key,
                model=model,
                optional_params={},
                litellm_params={},
            )
            == "https://api.openinfer.ai/v1/chat/completions"
        )

        headers = config.validate_environment(
            headers={},
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=api_base,
        )
        assert headers["Authorization"] == "Bearer sk-test"

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 128},
            optional_params={},
            model=model,
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 128
        assert "max_completion_tokens" not in optional_params

    @pytest.mark.parametrize(
        ("model", "input_cost_per_token", "output_cost_per_token"),
        (
            ("openinfer/@oi/Llama-3.2-1B-Instruct", 2e-08, 2e-08),
            ("openinfer/@oi/Qwen3.5-9B", 1.5e-07, 1.8e-07),
            ("openinfer/@oi/Qwen3.5-27B", 7.2e-07, 7.2e-07),
            ("openinfer/@oi/Gemma4-31B-It", 5.2e-07, 7.5e-07),
        ),
    )
    def test_catalog_token_rates_match_vendor_per_million_prices(
        self, model: str, input_cost_per_token: float, output_cost_per_token: float
    ):
        for path in _PRICE_FILES:
            catalog = json.loads(path.read_text())
            row = catalog[model]
            assert row["input_cost_per_token"] == input_cost_per_token
            assert row["output_cost_per_token"] == output_cost_per_token
            assert row["input_cost_per_token"] > 0
            assert row["output_cost_per_token"] > 0
