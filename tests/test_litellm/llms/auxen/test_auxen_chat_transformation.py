"""
Unit tests for the Auxen provider.

Auxen (https://auxen.ai) hosts per-customer dedicated LLM endpoints with an
OpenAI-compatible chat completions surface. The provider is a thin pass-through
over OpenAIGPTConfig — these tests verify the URL composition, env-var
resolution, and provider routing.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm import get_llm_provider
from litellm.llms.auxen.chat.transformation import AuxenChatConfig

SAMPLE_API_BASE = "https://api.auxen.ai/v1/inst_test/v1"
SAMPLE_API_KEY = "auxk_test_fake"


class TestAuxenChatConfig:
    def test_get_complete_url_appends_chat_completions(self):
        config = AuxenChatConfig()
        url = config.get_complete_url(
            api_base=SAMPLE_API_BASE,
            api_key=SAMPLE_API_KEY,
            model="llama-3.1-8b",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{SAMPLE_API_BASE}/chat/completions"

    def test_get_complete_url_strips_trailing_slash(self):
        config = AuxenChatConfig()
        url = config.get_complete_url(
            api_base=f"{SAMPLE_API_BASE}/",
            api_key=SAMPLE_API_KEY,
            model="llama-3.1-8b",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{SAMPLE_API_BASE}/chat/completions"

    def test_get_complete_url_idempotent_when_already_full(self):
        config = AuxenChatConfig()
        full = f"{SAMPLE_API_BASE}/chat/completions"
        url = config.get_complete_url(
            api_base=full,
            api_key=SAMPLE_API_KEY,
            model="llama-3.1-8b",
            optional_params={},
            litellm_params={},
        )
        assert url == full

    def test_get_complete_url_raises_without_api_base(self):
        config = AuxenChatConfig()
        with pytest.raises(ValueError, match="Auxen requires an `api_base`"):
            config.get_complete_url(
                api_base=None,
                api_key=SAMPLE_API_KEY,
                model="llama-3.1-8b",
                optional_params={},
                litellm_params={},
            )

    def test_provider_info_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("AUXEN_API_BASE", SAMPLE_API_BASE)
        monkeypatch.setenv("AUXEN_API_KEY", SAMPLE_API_KEY)
        config = AuxenChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )
        assert api_base == SAMPLE_API_BASE
        assert api_key == SAMPLE_API_KEY

    def test_provider_info_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AUXEN_API_BASE", "https://wrong.example/v1")
        monkeypatch.setenv("AUXEN_API_KEY", "wrong-key")
        config = AuxenChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=SAMPLE_API_BASE, api_key=SAMPLE_API_KEY
        )
        assert api_base == SAMPLE_API_BASE
        assert api_key == SAMPLE_API_KEY


class TestAuxenProviderRouting:
    def test_get_llm_provider_resolves_auxen_prefix(self):
        model, provider, _, api_base = get_llm_provider(
            model="auxen/llama-3.1-8b",
            api_base=SAMPLE_API_BASE,
            api_key=SAMPLE_API_KEY,
        )
        assert provider == "auxen"
        assert model == "llama-3.1-8b"
        assert api_base == SAMPLE_API_BASE

    def test_auxen_provider_in_enum(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders.AUXEN.value == "auxen"

    def test_auxen_config_in_provider_config_manager(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_chat_config(
            model="auxen/llama-3.1-8b", provider=LlmProviders.AUXEN
        )
        assert isinstance(cfg, AuxenChatConfig)
