"""
Tests for the Cognition provider identity.

Cognition serves an OpenAI-compatible /v1/chat/completions surface, but it must resolve to its
own `cognition` provider so OpenAI-specific pricing and provider-level reporting never apply to
its traffic.
"""

import json
from pathlib import Path

import pytest

import litellm


class TestCognitionProviderIdentity:
    def test_cognition_is_a_registered_provider(self):
        from litellm import LlmProviders

        assert LlmProviders.COGNITION.value == "cognition"
        assert "cognition" in litellm.provider_list

    def test_cognition_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        cognition = JSONProviderRegistry.get("cognition")
        assert cognition is not None
        assert cognition.base_url == "https://api.cognition.ai/v1"
        assert cognition.api_key_env == "COGNITION_API_KEY"
        assert cognition.api_base_env == "COGNITION_API_BASE"

    def test_cognition_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "cognition" in openai_compatible_providers

    def test_prefixed_model_resolves_to_cognition_not_openai(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="cognition/swe-1.7",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "swe-1.7"
        assert provider == "cognition"
        assert api_base == "https://api.cognition.ai/v1"

    def test_explicit_api_base_and_key_win(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="cognition/swe-1.7",
            custom_llm_provider=None,
            api_base="https://cognition.internal.example/v1",
            api_key="sk-test",
        )

        assert provider == "cognition"
        assert api_base == "https://cognition.internal.example/v1"
        assert api_key == "sk-test"

    def test_api_base_autodetects_cognition(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        _, provider, api_key, api_base = get_llm_provider(
            model="swe-1.7",
            custom_llm_provider=None,
            api_base="https://api.cognition.ai/v1",
            api_key=None,
        )

        assert provider == "cognition"
        assert api_base == "https://api.cognition.ai/v1"
        assert api_key == "sk-cognition-env"

    def test_autodetected_api_base_keeps_the_caller_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        _, provider, api_key, _ = get_llm_provider(
            model="swe-1.7",
            custom_llm_provider=None,
            api_base="https://api.cognition.ai/v1",
            api_key="sk-cognition-caller",
        )

        assert provider == "cognition"
        assert api_key == "sk-cognition-caller"

    def test_env_api_key_is_read_from_cognition_variable(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("COGNITION_API_KEY", "sk-cognition-env")

        provider = JSONProviderRegistry.get("cognition")
        assert provider is not None

        api_base, api_key = create_config_class(provider)()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.cognition.ai/v1"
        assert api_key == "sk-cognition-env"


class TestCognitionCostTracking:
    def test_supported_endpoints_matrix(self):
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        endpoints = matrix["providers"]["cognition"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True
        assert endpoints["embeddings"] is False
