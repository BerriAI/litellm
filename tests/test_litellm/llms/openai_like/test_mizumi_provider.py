"""
Tests for the Mizumi provider identity.

Mizumi (https://mizumi.co) serves an OpenAI-compatible /v1/chat/completions surface, but it
must resolve to its own `mizumi` provider so OpenAI-specific pricing and provider-level
reporting never apply to its traffic.

All tests are offline: no network access is required.
"""

import json
from pathlib import Path

import pytest

import litellm


class TestMizumiProviderIdentity:
    def test_mizumi_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.MIZUMI.value == "mizumi"
        assert "mizumi" in litellm.provider_list

    def test_mizumi_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        mizumi = JSONProviderRegistry.get("mizumi")
        assert mizumi is not None
        assert mizumi.base_url == "https://api.mizumi.co/v1"
        assert mizumi.api_key_env == "MIZUMI_API_KEY"
        assert mizumi.api_base_env == "MIZUMI_API_BASE"

    def test_mizumi_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "mizumi" in openai_compatible_providers

    def test_prefixed_model_resolves_to_mizumi_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="mizumi/gpt-5.5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-5.5"
        assert provider == "mizumi"
        assert api_base == "https://api.mizumi.co/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="mizumi/gpt-5.5",
            custom_llm_provider=None,
            api_base="https://mizumi.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "mizumi"
        assert api_base == "https://mizumi.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_mizumi(self, monkeypatch: pytest.MonkeyPatch):
        """Endpoint autodetection works off the JSON config alone.

        Regression guard: no provider-specific branch exists in
        get_llm_provider_logic.py — the generic JSONProviderRegistry fallback
        must pick Mizumi up from its registered base URL.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("MIZUMI_API_KEY", "sk-mizumi-env")

        _, provider, api_key, api_base = get_llm_provider(
            model="gpt-5.5",
            custom_llm_provider=None,
            api_base="https://api.mizumi.co/v1",
            api_key=None,
        )

        assert provider == "mizumi"
        assert api_base == "https://api.mizumi.co/v1"
        assert api_key == "sk-mizumi-env"

    def test_autodetected_api_base_keeps_the_caller_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("MIZUMI_API_KEY", "sk-mizumi-env")

        _, provider, api_key, _ = get_llm_provider(
            model="gpt-5.5",
            custom_llm_provider=None,
            api_base="https://api.mizumi.co/v1",
            api_key="sk-mizumi-caller",
        )

        assert provider == "mizumi"
        assert api_key == "sk-mizumi-caller"

    def test_env_api_key_is_read_from_mizumi_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("MIZUMI_API_KEY", "sk-mizumi-env")

        provider = JSONProviderRegistry.get("mizumi")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.mizumi.co/v1"
        assert api_key == "sk-mizumi-env"

    def test_complete_url_appends_chat_completions(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("mizumi")
        assert provider is not None
        config = create_config_class(provider)()

        url = config.get_complete_url(
            api_base="https://api.mizumi.co/v1",
            api_key="sk-test",
            model="mizumi/gpt-5.5",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.mizumi.co/v1/chat/completions"

    def test_supported_endpoints_matrix(self):
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        endpoints = matrix["providers"]["mizumi"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["embeddings"] is False
