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
_CATALOG_MODELS = ("openinfer/@oi/GLM-5.3-Flash", "openinfer/@oi/Qwen3.5-27B")


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
            model="openinfer/@oi/Qwen3.5-27B",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "@oi/Qwen3.5-27B"
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
            model="@oi/Qwen3.5-27B",
            custom_llm_provider=None,
            api_base="https://api.openinfer.ai/v1",
            api_key=None,
        )

        assert provider == "openinfer"
        assert api_key == "sk-env-key"

    def test_chat_completions_url(self):
        config = litellm.ProviderConfigManager.get_provider_chat_config(
            model="@oi/Qwen3.5-27B", provider=litellm.LlmProviders.OPENINFER
        )
        assert config is not None
        assert (
            config.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="@oi/Qwen3.5-27B",
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
            model="openinfer/@oi/Qwen3.5-27B",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )
        assert model == "@oi/Qwen3.5-27B"
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

    @pytest.mark.parametrize("model", _CATALOG_MODELS)
    def test_catalog_rows_are_positive_and_backup_agrees(self, model: str):
        primary = json.loads(_PRICE_FILES[0].read_text())[model]
        backup = json.loads(_PRICE_FILES[1].read_text())[model]

        assert primary == backup
        assert primary["litellm_provider"] == "openinfer"
        assert primary["mode"] == "chat"
        assert primary["max_tokens"] == primary["max_output_tokens"]
        assert primary["max_input_tokens"] > 0
        assert primary["max_output_tokens"] > 0
        assert primary["input_cost_per_token"] > 0
        assert primary["output_cost_per_token"] > 0

    @pytest.mark.parametrize("model", _CATALOG_MODELS)
    def test_chat_response_is_priced_at_its_catalog_row(self, model: str):
        row = litellm.model_cost[model]
        response = litellm.ModelResponse(
            model=model.removeprefix("openinfer/"),
            usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )

        cost = litellm.completion_cost(completion_response=response, model=model)

        assert cost == pytest.approx(1000 * row["input_cost_per_token"] + 200 * row["output_cost_per_token"])
