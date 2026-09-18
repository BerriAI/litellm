"""
Tests for Vispark provider configuration and integration.
"""

import litellm


class TestVisparkProviderConfig:
    def test_vispark_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "VISPARK")
        assert LlmProviders.VISPARK.value == "vispark"
        assert "vispark" in litellm.provider_list

    def test_vispark_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("vispark")

        vispark = JSONProviderRegistry.get("vispark")
        assert vispark is not None
        assert vispark.base_url == "https://api.lab.vispark.in/v1"
        assert vispark.api_key_env == "VISPARK_LAB_API_KEY"
        assert vispark.api_base_env == "VISPARK_LAB_API_BASE"
        assert vispark.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_vispark_in_openai_compatible_providers(self):
        from litellm.constants import (
            openai_compatible_providers,
            openai_text_completion_compatible_providers,
        )

        assert "vispark" in openai_compatible_providers
        assert "vispark" in openai_text_completion_compatible_providers

    def test_vispark_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="vispark/vision-medium",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "vision-medium"
        assert provider == "vispark"
        assert api_base == "https://api.lab.vispark.in/v1"

    def test_vispark_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="vispark/vision-medium",
            custom_llm_provider=None,
            api_base="https://custom.vispark.in/v1",
            api_key="sk-test",
        )

        assert provider == "vispark"
        assert api_base == "https://custom.vispark.in/v1"
        assert api_key == "sk-test"

    def test_vispark_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="vision-medium",
            custom_llm_provider=None,
            api_base="https://api.lab.vispark.in/v1",
            api_key=None,
        )
        assert provider == "vispark"
        assert api_base == "https://api.lab.vispark.in/v1"

    def test_vispark_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("vispark")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="vision-medium",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_vispark_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "vispark-chat",
                    "litellm_params": {
                        "model": "vispark/vision-medium",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "vispark-chat"


class TestVisparkModelMetadata:
    VISPARK_MODELS = (
        "vispark/vision-small",
        "vispark/vision-medium",
        "vispark/vision-large",
    )

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_vispark_models_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.VISPARK_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "vispark"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_tool_choice"] is True
            assert info["supports_reasoning"] is True
            assert info["supports_response_schema"] is True
            assert info["supports_vision"] is True

            assert info["max_output_tokens"] == 65536
            assert info["max_tokens"] == info["max_output_tokens"]
            assert info["max_input_tokens"] == 1000000

    def test_vispark_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.VISPARK_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"


class TestVisparkDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_vispark_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "vispark"]
        assert len(entries) == 1, "vispark must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "VISPARK"
        assert entry["provider_display_name"] == "Vispark"
        assert entry["default_model_placeholder"].startswith("vispark/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
