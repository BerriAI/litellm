"""
Tests for the Gondola provider configuration and integration.

Gondola (https://gondola-ai.com) is an OpenAI-compatible gateway in front of
Venice AI's model catalog. Besides ``/v1/chat/completions`` it natively serves
the Anthropic Messages API at ``/v1/messages``, so the JSON provider opts into
the untranslated passthrough.
"""

import json
from pathlib import Path

import litellm

REPO_ROOT = Path(__file__).parents[4]


class TestGondolaProviderConfig:
    def test_gondola_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "GONDOLA")
        assert LlmProviders.GONDOLA.value == "gondola"
        assert "gondola" in litellm.provider_list

    def test_gondola_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("gondola")

        gondola = JSONProviderRegistry.get("gondola")
        assert gondola is not None
        assert gondola.base_url == "https://api.gondola-ai.com/v1"
        assert gondola.api_key_env == "GONDOLA_API_KEY"
        assert gondola.api_base_env == "GONDOLA_API_BASE"
        assert gondola.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert set(gondola.supported_endpoints) == {"/v1/chat/completions", "/v1/messages"}

    def test_gondola_in_openai_compatible_lists(self):
        from litellm.constants import openai_compatible_endpoints, openai_compatible_providers

        assert "gondola" in openai_compatible_providers
        assert "https://api.gondola-ai.com/v1" in openai_compatible_endpoints

    def test_gondola_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gondola/deepseek-v3.2",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "deepseek-v3.2"
        assert provider == "gondola"
        assert api_base == "https://api.gondola-ai.com/v1"

    def test_gondola_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="gondola/deepseek-v3.2",
            custom_llm_provider=None,
            api_base="https://proxy.example.com/v1",
            api_key="gnd_test",
        )

        assert provider == "gondola"
        assert api_base == "https://proxy.example.com/v1"
        assert api_key == "gnd_test"

    def test_gondola_api_base_from_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("GONDOLA_API_BASE", "https://proxy.example.com/v1")

        _, provider, _, api_base = get_llm_provider(
            model="gondola/deepseek-v3.2",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "gondola"
        assert api_base == "https://proxy.example.com/v1"

    def test_gondola_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="deepseek-v3.2",
            custom_llm_provider=None,
            api_base="https://api.gondola-ai.com/v1",
            api_key=None,
        )

        assert provider == "gondola"
        assert api_base == "https://api.gondola-ai.com/v1"

    def test_gondola_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("gondola")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="deepseek-v3.2",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_gondola_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "gondola-chat",
                    "litellm_params": {
                        "model": "gondola/deepseek-v3.2",
                        "api_key": "gnd_test",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "gondola-chat"


class TestGondolaAnthropicMessages:
    def test_gondola_resolves_native_messages_config(self):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="deepseek-v3.2", provider=litellm.LlmProviders.GONDOLA
        )
        assert isinstance(cfg, JSONProviderAnthropicMessagesConfig)

    def test_complete_url_targets_gondola_messages(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        provider = JSONProviderRegistry.get("gondola")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        url = cfg.get_complete_url(
            api_base=None,
            api_key="gnd_test",
            model="deepseek-v3.2",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.gondola-ai.com/v1/messages"

        override_url = cfg.get_complete_url(
            api_base="https://proxy.example.com/v1",
            api_key="gnd_test",
            model="deepseek-v3.2",
            optional_params={},
            litellm_params={},
        )
        assert override_url == "https://proxy.example.com/v1/messages"

    def test_api_key_resolved_from_env(self, monkeypatch):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        monkeypatch.setenv("GONDOLA_API_KEY", "gnd_env_key")
        provider = JSONProviderRegistry.get("gondola")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        headers, _ = cfg.validate_anthropic_messages_environment(
            headers={},
            model="deepseek-v3.2",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        assert headers["authorization"] == "Bearer gnd_env_key"
        assert headers["anthropic-version"] == "2023-06-01"


class TestGondolaEndpointMatrix:
    @staticmethod
    def _providers(path_parts):
        with open(REPO_ROOT.joinpath(*path_parts)) as f:
            return json.load(f)["providers"]

    def test_gondola_listed_in_supported_endpoints_and_backup(self):
        root = self._providers(("provider_endpoints_support.json",))
        backup = self._providers(("litellm", "provider_endpoints_support_backup.json"))

        assert "gondola" in root, "gondola missing from provider_endpoints_support.json"
        assert "gondola" in backup, "gondola missing from the backup matrix"
        assert backup["gondola"] == root["gondola"]

        endpoints = root["gondola"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        # LiteLLM has no embedding dispatch for JSON-configured providers, so the
        # matrix must not advertise one even though the gateway serves it.
        assert endpoints["embeddings"] is False
        assert endpoints["responses"] is False


class TestGondolaDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            return json.load(f)

    def test_gondola_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "gondola"]
        assert len(entries) == 1, "gondola must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "GONDOLA"
        assert entry["provider_display_name"] == "Gondola"
        assert entry["default_model_placeholder"].startswith("gondola/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
        assert fields["api_base"]["placeholder"] == "https://api.gondola-ai.com/v1"
