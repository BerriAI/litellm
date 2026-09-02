"""
Tests for OpenZoo provider configuration and integration.
"""

import litellm


class TestOpenZooProviderConfig:
    def test_openzoo_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "OPENZOO")
        assert LlmProviders.OPENZOO.value == "openzoo"
        assert "openzoo" in litellm.provider_list

    def test_openzoo_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("openzoo")

        openzoo = JSONProviderRegistry.get("openzoo")
        assert openzoo is not None
        assert openzoo.base_url == "http://localhost:8402/v1"
        assert openzoo.api_key_env == "OPENZOO_API_KEY"
        assert openzoo.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_openzoo_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "openzoo" in openai_compatible_providers

    def test_openzoo_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openzoo/z-ai/glm-5.3-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "z-ai/glm-5.3-flash"
        assert provider == "openzoo"
        assert api_base == "http://localhost:8402/v1"

    def test_openzoo_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="openzoo/z-ai/glm-5.3-flash",
            custom_llm_provider=None,
            api_base="https://api.openzoo.fun/v1",
            api_key="ozk_live_example",
        )

        assert provider == "openzoo"
        assert api_base == "https://api.openzoo.fun/v1"
        assert api_key == "ozk_live_example"

    def test_openzoo_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="z-ai/glm-5.3-flash",
            custom_llm_provider=None,
            api_base="http://localhost:8402/v1",
            api_key=None,
        )
        assert provider == "openzoo"
        assert api_base == "http://localhost:8402/v1"

    def test_openzoo_temperature_passthrough(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("openzoo")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"temperature": 0.4},
            optional_params={},
            model="z-ai/glm-5.3-flash",
            drop_params=False,
        )
        assert optional_params["temperature"] == 0.4

    def test_openzoo_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("openzoo")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="z-ai/glm-5.3-flash",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_openzoo_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "openzoo-chat",
                    "litellm_params": {
                        "model": "openzoo/z-ai/glm-5.3-flash",
                        "api_key": "sk-openzoo",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "openzoo-chat"


class TestOpenZooModelMetadata:
    OPENZOO_MODELS = (
        "openzoo/z-ai/glm-5.3-flash",
        "openzoo/qwen/qwen3.7-flash",
        "openzoo/nvidia/nemotron-3.5-lightning",
    )
    MIN_INPUT_TOKENS = {
        "openzoo/z-ai/glm-5.3-flash": 1310720,
        "openzoo/qwen/qwen3.7-flash": 1000000,
        "openzoo/nvidia/nemotron-3.5-lightning": 262144,
    }

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_openzoo_models_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.OPENZOO_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "openzoo"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0
            assert info["supports_function_calling"] is True
            assert info["supports_tool_choice"] is True
            assert info["supports_reasoning"] is True
            assert info.get("supports_vision", False) is False

            assert info["max_output_tokens"] > 0
            assert info["max_tokens"] == info["max_output_tokens"]
            assert info["max_input_tokens"] == self.MIN_INPUT_TOKENS[model]

    def test_openzoo_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.OPENZOO_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"


class TestOpenZooDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_openzoo_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "openzoo"]
        assert len(entries) == 1, "openzoo must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "OpenZoo"
        assert entry["provider_display_name"] == "OpenZoo"
        assert entry["default_model_placeholder"].startswith("openzoo/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
