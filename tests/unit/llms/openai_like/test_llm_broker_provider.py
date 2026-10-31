"""
Tests for the LLM Broker provider configuration.
"""

import json
import os

import litellm


class TestLLMBrokerProviderConfig:
    def test_in_provider_list(self):
        from litellm import LlmProviders

        assert LlmProviders.LLM_BROKER.value == "llm_broker"
        assert "llm_broker" in litellm.provider_list

    def test_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        cfg = JSONProviderRegistry.get("llm_broker")
        assert cfg is not None
        assert cfg.base_url == "https://api.llm-broker.net/api/v1"
        assert cfg.api_key_env == "LLM_BROKER_API_KEY"
        assert cfg.api_base_env == "LLM_BROKER_API_BASE"

    def test_prefix_resolution(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("LLM_BROKER_API_KEY", "env-key")
        monkeypatch.delenv("LLM_BROKER_API_BASE", raising=False)
        model, provider, api_key, api_base = get_llm_provider(model="llm_broker/taylor")

        assert (model, provider) == ("taylor", "llm_broker")
        assert api_base == "https://api.llm-broker.net/api/v1"
        assert api_key == "env-key"

    def test_explicit_key_and_base_win_over_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("LLM_BROKER_API_KEY", "env-key")
        _, provider, api_key, api_base = get_llm_provider(
            model="llm_broker/taylor",
            api_base="https://custom.example.com/v1",
            api_key="caller-key",
        )

        assert provider == "llm_broker"
        assert api_key == "caller-key"
        assert api_base == "https://custom.example.com/v1"

    def test_api_base_env_override(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("LLM_BROKER_API_BASE", "https://eu.example.com/v1")
        _, _, _, api_base = get_llm_provider(model="llm_broker/taylor")

        assert api_base == "https://eu.example.com/v1"

    def test_api_base_autodetection_keeps_caller_key(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("LLM_BROKER_API_KEY", "env-key")
        _, provider, api_key, _ = get_llm_provider(
            model="taylor",
            api_base="https://api.llm-broker.net/api/v1",
            api_key="caller-key",
        )

        assert provider == "llm_broker"
        assert api_key == "caller-key"

    def test_support_matrix_lists_chat_only(self):
        root = os.path.dirname(os.path.dirname(litellm.__file__))
        with open(os.path.join(root, "provider_endpoints_support.json")) as f:
            providers = json.load(f)["providers"]

        endpoints = providers["llm_broker"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert not any(v for k, v in endpoints.items() if k != "chat_completions")
