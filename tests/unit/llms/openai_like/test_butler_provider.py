import pytest

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

BUTLER_BASE_URL = "https://api.butlercompute.com/v1"


def test_butler_provider_registered():
    provider = JSONProviderRegistry.get("butler")
    assert provider is not None
    assert provider.base_url == BUTLER_BASE_URL
    assert provider.api_key_env == "BUTLER_API_KEY"
    assert provider.api_base_env == "BUTLER_API_BASE"
    assert provider.supported_endpoints == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]


def test_butler_resolves_the_environment_connection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUTLER_API_KEY", "test-key")
    monkeypatch.delenv("BUTLER_API_BASE", raising=False)
    assert litellm.get_llm_provider("butler/cicero") == ("cicero", "butler", "test-key", BUTLER_BASE_URL)

    monkeypatch.setenv("BUTLER_API_BASE", "https://gateway.example/v1")
    assert litellm.get_llm_provider("butler/cicero")[3] == "https://gateway.example/v1"
