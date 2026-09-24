"""
Tests for the APIMODELS provider configuration.

APIMODELS (https://apimodels.app) is an OpenAI-compatible gateway, so it is
registered through the JSON provider system rather than a bespoke Python module.
These tests are pure configuration assertions and mocks — no network calls.
"""

import litellm


class TestAPIModelsProviderConfig:
    def test_apimodels_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "APIMODELS")
        assert LlmProviders.APIMODELS.value == "apimodels"
        assert "apimodels" in litellm.provider_list

    def test_apimodels_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("apimodels")

        provider = JSONProviderRegistry.get("apimodels")
        assert provider is not None
        assert provider.base_url == "https://api.apimodels.app/v1"
        assert provider.api_key_env == "APIMODELS_API_KEY"

    def test_apimodels_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "apimodels" in openai_compatible_providers

    def test_apimodels_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("apimodels")

    def test_apimodels_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="apimodels/gpt-5.6-luna",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-5.6-luna"
        assert provider == "apimodels"
        assert api_base == "https://api.apimodels.app/v1"

    def test_apimodels_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="apimodels/gpt-5.6-luna",
            custom_llm_provider=None,
            api_base="https://custom.apimodels.app/v1",
            api_key="sk-test",
        )

        assert provider == "apimodels"
        assert api_base == "https://custom.apimodels.app/v1"
        assert api_key == "sk-test"

    def test_apimodels_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gpt-5.6-luna",
            custom_llm_provider=None,
            api_base="https://api.apimodels.app/v1",
            api_key=None,
        )

        assert provider == "apimodels"
        assert api_base == "https://api.apimodels.app/v1"

    def test_apimodels_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "apimodels-chat",
                    "litellm_params": {
                        "model": "apimodels/gpt-5.6-luna",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "apimodels-chat"


class TestAPIModelsDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        path = (
            Path(litellm.__file__).parent
            / "proxy"
            / "public_endpoints"
            / "provider_create_fields.json"
        )
        with open(path) as f:
            return json.load(f)

    def test_apimodels_is_selectable_in_the_add_model_form(self):
        entries = [
            e for e in self._provider_create_fields() if e["litellm_provider"] == "apimodels"
        ]
        assert len(entries) == 1, "apimodels must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "APIMODELS"
        assert entry["default_model_placeholder"].startswith("apimodels/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
