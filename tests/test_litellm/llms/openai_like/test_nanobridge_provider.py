"""
Mock-only unit tests for Nanobridge JSON-configured OpenAI-compatible provider.
Live / network tests: tests/llm_translation/test_nanobridge.py
"""

import os
import sys

import litellm
from litellm import LlmProviders
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)


class TestNanobridgeProvider:
    def test_nanobridge_in_llm_providers_enum(self):
        assert hasattr(LlmProviders, "NANOBRIDGE")
        assert LlmProviders.NANOBRIDGE.value == "nanobridge"

    def test_nanobridge_json_config_exists(self):
        assert JSONProviderRegistry.exists("nanobridge")
        cfg = JSONProviderRegistry.get("nanobridge")
        assert cfg is not None
        assert cfg.base_url == "https://api.nanobridge.net/v1"
        assert cfg.api_key_env == "NANOBRIDGE_API_KEY"
        assert cfg.api_base_env == "NANOBRIDGE_API_BASE"

    def test_nanobridge_provider_resolution(self):
        model, provider, _api_key, api_base = get_llm_provider(
            model="nanobridge/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert model == "deepseek-v4-flash"
        assert provider == "nanobridge"
        assert api_base == "https://api.nanobridge.net/v1"

    def test_nanobridge_api_base_env_override(self, monkeypatch):
        monkeypatch.setenv("NANOBRIDGE_API_BASE", "https://api-sg.nanobridge.net/v1")
        _model, _provider, _api_key, api_base = get_llm_provider(
            model="nanobridge/deepseek-v4-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert api_base == "https://api-sg.nanobridge.net/v1"

    def test_nanobridge_completion_mocked(self):
        response = litellm.completion(
            model="nanobridge/deepseek-v4-flash",
            messages=[{"role": "user", "content": "ping"}],
            mock_response="PONG",
        )
        assert response.choices[0].message.content == "PONG"
