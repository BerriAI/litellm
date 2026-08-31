"""
Tests for the Aquaduck provider identity.

Aquaduck serves an OpenAI-compatible /v1/chat/completions surface at
https://aqi.aquaduck.ai/v1. It must resolve to its own `aquaduck` provider so
OpenAI-specific pricing and provider-level reporting never apply to its traffic.
"""

import json
from pathlib import Path

import pytest

import litellm


class TestAquaduckProviderIdentity:
    def test_aquaduck_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.AQUADUCK.value == "aquaduck"
        assert "aquaduck" in litellm.provider_list

    def test_aquaduck_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        aquaduck = JSONProviderRegistry.get("aquaduck")
        assert aquaduck is not None
        assert aquaduck.base_url == "https://aqi.aquaduck.ai/v1"
        assert aquaduck.api_key_env == "AQUADUCK_API_KEY"
        assert aquaduck.api_base_env == "AQUADUCK_API_BASE"
        assert aquaduck.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert aquaduck.supported_endpoints == ["/v1/chat/completions"]

    def test_aquaduck_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "aquaduck" in openai_compatible_providers
        assert "https://aqi.aquaduck.ai/v1" in openai_compatible_endpoints

    def test_prefixed_model_resolves_to_aquaduck_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="aquaduck/zai-org/glm-4.7-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "zai-org/glm-4.7-flash"
        assert provider == "aquaduck"
        assert api_base == "https://aqi.aquaduck.ai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="aquaduck/zai-org/glm-4.7-flash",
            custom_llm_provider=None,
            api_base="https://aquaduck.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "aquaduck"
        assert api_base == "https://aquaduck.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_aquaduck(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("AQUADUCK_API_KEY", "sk-aquaduck-env")

        _, provider, api_key, api_base = get_llm_provider(
            model="zai-org/glm-4.7-flash",
            custom_llm_provider=None,
            api_base="https://aqi.aquaduck.ai/v1",
            api_key=None,
        )

        assert provider == "aquaduck"
        assert api_base == "https://aqi.aquaduck.ai/v1"
        assert api_key == "sk-aquaduck-env"

    def test_autodetected_api_base_keeps_the_caller_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("AQUADUCK_API_KEY", "sk-aquaduck-env")

        _, provider, api_key, _ = get_llm_provider(
            model="zai-org/glm-4.7-flash",
            custom_llm_provider=None,
            api_base="https://aqi.aquaduck.ai/v1",
            api_key="sk-aquaduck-caller",
        )

        assert provider == "aquaduck"
        assert api_key == "sk-aquaduck-caller"

    def test_env_api_key_is_read_from_aquaduck_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AQUADUCK_API_KEY", "sk-aquaduck-env")

        provider = JSONProviderRegistry.get("aquaduck")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://aqi.aquaduck.ai/v1"
        assert api_key == "sk-aquaduck-env"

    def test_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("aquaduck")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="zai-org/glm-4.7-flash",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params


class TestAquaduckCostTracking:
    AQUADUCK_MODELS = (
        "aquaduck/zai-org/glm-4.7-flash",
        "aquaduck/qwen/qwen3-14b",
        "aquaduck/qwen/qwen3.8-27b",
        "aquaduck/google/gemma-4-26b-a4b-it",
        "aquaduck/ornith-ai/ornith-1.5-9b",
    )
    VISION_MODELS = (
        "aquaduck/qwen/qwen3.8-27b",
        "aquaduck/google/gemma-4-26b-a4b-it",
    )

    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.get_model_info.cache_clear()
        yield
        litellm.get_model_info.cache_clear()

    @staticmethod
    def _load(path_parts):
        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "model, input_cost, output_cost",
        [
            ("aquaduck/zai-org/glm-4.7-flash", 5e-08, 2e-07),
            ("aquaduck/qwen/qwen3-14b", 5e-08, 1.2e-07),
            ("aquaduck/qwen/qwen3.8-27b", 2e-07, 1.5e-06),
            ("aquaduck/google/gemma-4-26b-a4b-it", 5e-08, 2e-07),
            ("aquaduck/ornith-ai/ornith-1.5-9b", 5e-08, 1e-07),
        ],
    )
    def test_cost_map_entries(self, model: str, input_cost: float, output_cost: float):
        info = litellm.get_model_info(model=model)

        assert info["litellm_provider"] == "aquaduck"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == input_cost
        assert info["output_cost_per_token"] == output_cost
        assert info.get("supports_vision", False) is (model in self.VISION_MODELS)

    def test_aquaduck_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.AQUADUCK_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"

    @pytest.mark.parametrize(
        "model, expected_prompt_cost, expected_completion_cost",
        [
            ("aquaduck/zai-org/glm-4.7-flash", 0.05, 0.2),
            ("aquaduck/qwen/qwen3.8-27b", 0.2, 1.5),
        ],
    )
    def test_cost_per_million_tokens(
        self, model: str, expected_prompt_cost: float, expected_completion_cost: float
    ):
        from litellm.cost_calculator import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            custom_llm_provider="aquaduck",
        )

        assert prompt_cost == pytest.approx(expected_prompt_cost)
        assert completion_cost == pytest.approx(expected_completion_cost)

    def test_supported_endpoints_matrix(self):
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        endpoints = matrix["providers"]["aquaduck"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is False
        assert endpoints["responses"] is False
        assert endpoints["embeddings"] is False


class TestAquaduckRouting:
    @pytest.fixture(autouse=True)
    def _use_local_model_cost_map(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.get_model_info.cache_clear()
        yield
        litellm.get_model_info.cache_clear()

    @pytest.mark.asyncio
    async def test_router_spend_is_attributed_to_aquaduck_pricing(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "glm-flash",
                    "litellm_params": {
                        "model": "aquaduck/zai-org/glm-4.7-flash",
                        "api_key": "sk-test",
                    },
                }
            ]
        )

        response = await router.acompletion(
            model="glm-flash",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello from aquaduck",
        )

        usage = response.usage
        expected = usage.prompt_tokens * 5e-08 + usage.completion_tokens * 2e-07
        assert response._hidden_params["response_cost"] == pytest.approx(expected)


class TestAquaduckDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_aquaduck_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "aquaduck"]
        assert len(entries) == 1, "aquaduck must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "Aquaduck"
        assert entry["provider_display_name"] == "Aquaduck AI"
        assert entry["default_model_placeholder"].startswith("aquaduck/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
        assert fields["api_base"]["placeholder"] == "https://aqi.aquaduck.ai/v1"
