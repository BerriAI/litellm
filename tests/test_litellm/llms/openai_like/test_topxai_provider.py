"""
Tests for the TopxAI provider configuration and integration.
"""

import litellm

TOPXAI_BASE_URL = "https://ai.topxea.com/v1"


class TestTopxAIProviderConfig:
    def test_topxai_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "TOPXAI")
        assert LlmProviders.TOPXAI.value == "topxai"
        assert "topxai" in litellm.provider_list

    def test_topxai_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("topxai")

        topxai = JSONProviderRegistry.get("topxai")
        assert topxai is not None
        assert topxai.base_url == TOPXAI_BASE_URL
        assert topxai.api_key_env == "TOPXAI_API_KEY"
        assert topxai.api_base_env == "TOPXAI_API_BASE"
        assert "/v1/responses" in topxai.supported_endpoints

    def test_topxai_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "topxai" in openai_compatible_providers

    def test_topxai_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "claude-sonnet-5"
        assert provider == "topxai"
        assert api_base == TOPXAI_BASE_URL

    def test_topxai_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="topxai/claude-sonnet-5",
            custom_llm_provider=None,
            api_base="https://relay.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "topxai"
        assert api_base == "https://relay.example.com/v1"
        assert api_key == "sk-test"

    def test_topxai_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gpt-5.6-sol",
            custom_llm_provider=None,
            api_base=TOPXAI_BASE_URL,
            api_key=None,
        )
        assert provider == "topxai"
        assert api_base == TOPXAI_BASE_URL

    def test_topxai_resolves_env_api_key(self, monkeypatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("topxai")
        assert provider is not None
        config = create_config_class(provider)()
        monkeypatch.setenv("TOPXAI_API_KEY", "sk-test")
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == TOPXAI_BASE_URL
        assert api_key == "sk-test"

    def test_topxai_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("topxai")
        assert provider is not None
        config = create_config_class(provider)()
        url = config.get_complete_url(
            api_base=TOPXAI_BASE_URL,
            api_key="sk-test",
            model="topxai/kimi-k3",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == f"{TOPXAI_BASE_URL}/chat/completions"

    def test_topxai_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "topxai/claude-sonnet-5",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "sonnet"


class TestTopxAIModelMetadata:
    TOPXAI_MODELS = (
        "topxai/claude-sonnet-5",
        "topxai/claude-opus-5",
        "topxai/claude-fable-5-1",
        "topxai/claude-fable-5",
        "topxai/gpt-5.6-sol",
        "topxai/gpt-6-astra",
        "topxai/grok-4.6",
        "topxai/kimi-k3",
        "topxai/GLM-5.3-Abliterated",
    )
    TEXT_ONLY_MODELS = ("topxai/GLM-5.3-Abliterated",)
    TIERED_MODELS = {
        "topxai/gpt-5.6-sol": "272k",
        "topxai/gpt-6-astra": "272k",
        "topxai/grok-4.6": "200k",
    }

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_topxai_models_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.TOPXAI_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "topxai"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_tool_choice"] is True
            assert info["supports_reasoning"] is True
            assert info["supports_response_schema"] is True
            assert info["supports_vision"] is (model not in self.TEXT_ONLY_MODELS)

            assert info["supports_prompt_caching"] is True
            assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"]
            assert info["cache_creation_input_token_cost"] >= info["input_cost_per_token"]

            assert info["max_tokens"] == info["max_output_tokens"]
            assert info["max_input_tokens"] >= 500_000
            assert info["source"].startswith("https://ai.topxea.com/pricing/")

    def test_topxai_long_context_tiers_double_the_input_rate(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model, tier in self.TIERED_MODELS.items():
            info = model_cost[model]
            assert info[f"input_cost_per_token_above_{tier}_tokens"] == 2 * info["input_cost_per_token"]
            assert info[f"cache_read_input_token_cost_above_{tier}_tokens"] == 2 * info["cache_read_input_token_cost"]
            assert info[f"output_cost_per_token_above_{tier}_tokens"] > info["output_cost_per_token"]

    def test_topxai_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.TOPXAI_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"


class TestTopxAIDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_topxai_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "topxai"]
        assert len(entries) == 1, "topxai must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "TOPXAI"
        assert entry["provider_display_name"] == "TopxAI"
        assert entry["default_model_placeholder"].startswith("topxai/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
