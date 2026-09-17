"""
Tests for the ainetcafe (Kimi K3) provider configuration and integration.
"""

import litellm


class TestAinetcafeProviderConfig:
    def test_ainetcafe_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "AINETCAFE")
        assert LlmProviders.AINETCAFE.value == "ainetcafe"
        assert "ainetcafe" in litellm.provider_list

    def test_ainetcafe_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("ainetcafe")

        provider = JSONProviderRegistry.get("ainetcafe")
        assert provider is not None
        assert provider.base_url == "https://microquickjs.com/v1"
        assert provider.api_key_env == "AINETCAFE_API_KEY"
        assert provider.api_base_env == "AINETCAFE_API_BASE"
        assert "/v1/chat/completions" in provider.supported_endpoints

    def test_ainetcafe_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "ainetcafe" in openai_compatible_providers

    def test_ainetcafe_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="ainetcafe/Kimi-K3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "Kimi-K3"
        assert provider == "ainetcafe"
        assert api_base == "https://microquickjs.com/v1"

    def test_ainetcafe_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="ainetcafe/Kimi-K3",
            custom_llm_provider=None,
            api_base="https://dedicated.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "ainetcafe"
        assert api_base == "https://dedicated.example.com/v1"
        assert api_key == "sk-test"

    def test_ainetcafe_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="Kimi-K3",
            custom_llm_provider=None,
            api_base="https://microquickjs.com/v1",
            api_key=None,
        )
        assert provider == "ainetcafe"
        assert api_base == "https://microquickjs.com/v1"

    def test_ainetcafe_passes_reasoning_effort_and_max_completion_tokens(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("ainetcafe")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256, "reasoning_effort": "high"},
            optional_params={},
            model="Kimi-K3",
            drop_params=False,
        )
        # The endpoint accepts both fields natively; nothing is renamed.
        assert optional_params["max_completion_tokens"] == 256
        assert optional_params["reasoning_effort"] == "high"

    def test_ainetcafe_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "k3",
                    "litellm_params": {
                        "model": "ainetcafe/Kimi-K3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "k3"


class TestAinetcafeModelMetadata:
    MODEL = "ainetcafe/Kimi-K3"

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_ainetcafe_model_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        info = model_cost.get(self.MODEL)
        assert info is not None, f"{self.MODEL} missing from model_prices_and_context_window.json"
        assert info["litellm_provider"] == "ainetcafe"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == 2.1e-06
        assert info["output_cost_per_token"] == 1.05e-05
        assert info["cache_read_input_token_cost"] == 3e-07
        assert info["supports_function_calling"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_response_schema"] is True
        assert info["supports_vision"] is True
        assert info["supports_prompt_caching"] is True
        assert info["max_tokens"] == info["max_output_tokens"]
        assert info["max_input_tokens"] == 262144

    def test_ainetcafe_model_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        assert self.MODEL in backup, f"{self.MODEL} missing from backup json"
        assert backup[self.MODEL] == model_cost[self.MODEL]


class TestAinetcafeDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_ainetcafe_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "ainetcafe"]
        assert len(entries) == 1, "ainetcafe must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "AINETCAFE"
        assert entry["provider_display_name"] == "ainetcafe"
        assert entry["default_model_placeholder"] == "ainetcafe/Kimi-K3"

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
