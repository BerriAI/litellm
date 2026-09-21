"""
Tests for the uRun provider (JSON-configured openai_like fast track).
"""

import json
from pathlib import Path

import pytest

import litellm


URUN_MODELS: tuple[str, ...] = (
    "urun/qwen3.8-27b:nvfp4",
)


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


class TestUrunProviderConfig:
    def test_urun_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders("urun") is LlmProviders.URUN
        assert "urun" in litellm.provider_list

    def test_urun_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "urun" in openai_compatible_providers

    def test_urun_json_config_loads(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("urun")

        urun = JSONProviderRegistry.get("urun")
        assert urun is not None
        assert urun.base_url == "https://inference.urun.sh/v1"
        assert urun.api_key_env == "URUN_API_KEY"
        assert urun.api_base_env == "URUN_API_BASE"

    def test_urun_not_registered_for_text_completion(self):
        """uRun does not serve /v1/completions, so it must stay out of the text-completion list."""
        from litellm.constants import openai_text_completion_compatible_providers

        assert "urun" not in openai_text_completion_compatible_providers


class TestUrunProviderResolution:
    def test_provider_prefix_stripped_and_colon_survives(self):
        """The quantization suffix is part of the served model id, not a deployment marker."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="urun/qwen3.8-27b:nvfp4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.8-27b:nvfp4"
        assert provider == "urun"
        assert api_base == "https://inference.urun.sh/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="urun/qwen3.8-27b:nvfp4",
            custom_llm_provider=None,
            api_base="https://custom.urun.sh/v1",
            api_key="sk-test",
        )

        assert provider == "urun"
        assert api_base == "https://custom.urun.sh/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetection_resolves_urun(self):
        """A bare model plus the uRun api_base must resolve to the urun provider via the
        JSON registry fallback, without a provider-specific branch in
        get_llm_provider_logic."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="qwen3.8-27b:nvfp4",
            custom_llm_provider=None,
            api_base="https://inference.urun.sh/v1",
            api_key=None,
        )

        assert model == "qwen3.8-27b:nvfp4"
        assert provider == "urun"
        assert api_base == "https://inference.urun.sh/v1"

    def test_router_accepts_urun_deployment(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "urun-chat",
                    "litellm_params": {
                        "model": "urun/qwen3.8-27b:nvfp4",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "urun-chat"


class TestUrunCompletion:
    def test_mocked_completion_keeps_colon_in_model_id(self):
        response = litellm.completion(
            model="urun/qwen3.8-27b:nvfp4",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from urun",
            api_key="sk-test",
        )

        assert response.choices[0].message.content == "hello from urun"
        assert response.model == "qwen3.8-27b:nvfp4"
        assert response._hidden_params["custom_llm_provider"] == "urun"


class TestUrunCostTracking:
    def test_every_price_row_bills_its_committed_rate(self, local_model_cost_map):
        """A missing or zero-cost row is a silent billing bug, so each served model must
        bill prompt + completion tokens at the rates committed in the price map."""
        for model in URUN_MODELS:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello",
                api_key="sk-test",
            )

            cost = litellm.completion_cost(completion_response=response)

            row = litellm.model_cost[model]
            expected = (
                response.usage.prompt_tokens * row["input_cost_per_token"]
                + response.usage.completion_tokens * row["output_cost_per_token"]
            )
            assert cost == pytest.approx(expected), model
            assert cost > 0, model


class TestUrunResponsesApiSupport:
    def test_responses_endpoint_declared(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        urun = JSONProviderRegistry.get("urun")
        assert urun is not None
        assert "/v1/responses" in urun.supported_endpoints
        assert JSONProviderRegistry.supports_responses_api("urun") is True

    def test_provider_config_manager_returns_responses_config(self):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="urun",
            model="qwen3.8-27b:nvfp4",
        )

        assert config is not None
        assert config.custom_llm_provider == "urun"


class TestUrunRegistryFiles:
    @staticmethod
    def _load_root_and_backup_price_rows() -> tuple[dict, dict]:
        repo_root = Path(__file__).parents[4]
        rows = json.loads((repo_root / "model_prices_and_context_window.json").read_text())
        backup = json.loads(
            (repo_root / "litellm" / "model_prices_and_context_window_backup.json").read_text()
        )
        return rows, backup

    def test_price_rows_feed_the_urun_wildcard_grant(self, local_model_cost_map):
        """An `urun/*` model-group grant lists models from litellm.models_by_provider, which
        only carries them when the price rows are wired into urun_models in litellm/__init__."""
        litellm.add_known_models()

        listed = litellm.models_by_provider["urun"]
        for model in URUN_MODELS:
            assert model in listed, model

    def test_price_rows_exist_and_twins_agree(self):
        rows, backup = self._load_root_and_backup_price_rows()

        for model in URUN_MODELS:
            assert model in rows, model
            assert model in backup, model
            assert backup[model] == rows[model], model
            assert rows[model]["litellm_provider"] == "urun"
            assert rows[model]["mode"] == "chat"

    def test_backup_endpoint_matrix_lists_urun(self):
        """GET /public/supported_endpoints serves the bundled backup matrix, so uRun must
        appear there with the endpoints it actually serves."""
        backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
        matrix = json.loads(backup_path.read_text())

        assert "urun" in matrix["providers"]
        endpoints = matrix["providers"]["urun"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is True
