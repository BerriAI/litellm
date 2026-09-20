"""
Tests for Viktor provider configuration and integration.
"""

import json
from pathlib import Path

import pytest

import litellm

VIKTOR_API_BASE = "https://api.viktor.com/api/compat/v1"


class TestViktorProviderConfig:
    def test_viktor_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "VIKTOR")
        assert LlmProviders.VIKTOR.value == "viktor"
        assert "viktor" in litellm.provider_list

    def test_viktor_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("viktor")

        viktor = JSONProviderRegistry.get("viktor")
        assert viktor is not None
        assert viktor.base_url == VIKTOR_API_BASE
        assert viktor.api_key_env == "VIKTOR_API_KEY"
        assert viktor.api_base_env == "VIKTOR_API_BASE"
        assert viktor.param_mappings == {}
        assert viktor.constraints == {}

    def test_viktor_supports_responses_api(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("viktor") is True

    def test_viktor_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "viktor" in openai_compatible_providers
        assert VIKTOR_API_BASE in openai_compatible_endpoints

    def test_viktor_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="viktor/viktor",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "viktor"
        assert provider == "viktor"
        assert api_base == VIKTOR_API_BASE

    def test_viktor_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="viktor/viktor",
            custom_llm_provider=None,
            api_base="https://viktor-gateway.internal.example/api/compat/v1",
            api_key="zt_test_sk_example",
        )

        assert provider == "viktor"
        assert api_base == "https://viktor-gateway.internal.example/api/compat/v1"
        assert api_key == "zt_test_sk_example"

    def test_viktor_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        # Viktor serves one agent and does not validate the `model` field, so any
        # un-prefixed model string sent to its base url should route to the provider
        model, provider, api_key, api_base = get_llm_provider(
            model="default",
            custom_llm_provider=None,
            api_base=VIKTOR_API_BASE,
            api_key=None,
        )
        assert model == "default"
        assert provider == "viktor"
        assert api_base == VIKTOR_API_BASE

    def test_viktor_env_api_key_is_read_from_viktor_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VIKTOR_API_BASE", raising=False)
        monkeypatch.setenv("VIKTOR_API_KEY", "zt_test_sk_env")

        provider = JSONProviderRegistry.get("viktor")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == VIKTOR_API_BASE
        assert api_key == "zt_test_sk_env"

    def test_viktor_max_completion_tokens_passed_through(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("viktor")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="viktor",
            drop_params=False,
        )
        assert optional_params["max_completion_tokens"] == 256

    def test_viktor_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "viktor",
                    "litellm_params": {
                        "model": "viktor/viktor",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "viktor"


class TestViktorEndpointMatrix:
    @staticmethod
    def _endpoints(path: Path) -> dict:
        return json.loads(path.read_text())["providers"]["viktor"]["endpoints"]

    def test_viktor_supported_endpoints_matrix(self):
        endpoints = self._endpoints(Path(litellm.__file__).parent / "provider_endpoints_support_backup.json")

        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is True
        assert endpoints["embeddings"] is False

    def test_viktor_endpoint_matrix_synced_to_backup(self):
        root = Path(__file__).parents[4] / "provider_endpoints_support.json"
        backup = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"

        assert self._endpoints(root) == self._endpoints(backup)
