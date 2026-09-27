"""
Tests for Wallaby provider configuration and integration.
"""

import litellm


class TestWallabyProviderConfig:
    def test_wallaby_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "WALLABY")
        assert LlmProviders.WALLABY.value == "wallaby"
        assert "wallaby" in litellm.provider_list

    def test_wallaby_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("wallaby")

        wallaby = JSONProviderRegistry.get("wallaby")
        assert wallaby is not None
        assert wallaby.base_url == "https://api.wallabytoken.com/v1"
        assert wallaby.api_key_env == "WALLABY_API_KEY"

    def test_wallaby_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "wallaby" in openai_compatible_providers

    def test_wallaby_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="wallaby/kimi-k3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "kimi-k3"
        assert provider == "wallaby"
        assert api_base == "https://api.wallabytoken.com/v1"

    def test_wallaby_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="wallaby/kimi-k3",
            custom_llm_provider=None,
            api_base="https://custom.wallabytoken.com/v1",
            api_key="sk-test",
        )

        assert provider == "wallaby"
        assert api_base == "https://custom.wallabytoken.com/v1"
        assert api_key == "sk-test"

    def test_wallaby_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="kimi-k3",
            custom_llm_provider=None,
            api_base="https://api.wallabytoken.com/v1",
            api_key=None,
        )
        assert provider == "wallaby"
        assert api_base == "https://api.wallabytoken.com/v1"

    def test_wallaby_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "wallaby-chat",
                    "litellm_params": {
                        "model": "wallaby/kimi-k3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "wallaby-chat"


class TestWallabyModelMetadata:
    WALLABY_MODELS = ("wallaby/kimi-k3",)

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_wallaby_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.WALLABY_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], (
                f"{model} differs between root and backup json"
            )


class TestWallabyDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = (
            Path(litellm.__file__).parent
            / "proxy"
            / "public_endpoints"
            / "provider_create_fields.json"
        )
        with open(path) as f:
            return json.load(f)

    def test_wallaby_is_selectable_in_the_add_model_form(self):
        entries = [
            e for e in self._provider_create_fields() if e["litellm_provider"] == "wallaby"
        ]
        assert len(entries) == 1, "wallaby must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "WALLABY"
        assert entry["provider_display_name"] == "Wallaby"
        assert entry["default_model_placeholder"].startswith("wallaby/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
