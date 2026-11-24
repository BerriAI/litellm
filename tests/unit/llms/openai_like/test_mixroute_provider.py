from typing import Final

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry


def test_mixroute_prefix_preserves_model_and_resolves_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXROUTE_API_KEY", "mixroute-test-key")
    monkeypatch.delenv("MIXROUTE_API_BASE", raising=False)
    config: Final = JSONProviderRegistry.get("mixroute")
    assert config is not None
    model, provider, key, base = get_llm_provider(model="mixroute/example-model")
    assert (model, provider, key, base) == ("example-model", "mixroute", "mixroute-test-key", config.base_url)


def test_mixroute_endpoint_detection_preserves_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXROUTE_API_KEY", "environment-key")
    config: Final = JSONProviderRegistry.get("mixroute")
    assert config is not None
    model, provider, key, base = get_llm_provider(
        model="example-model", api_base=config.base_url, api_key="caller-key"
    )
    assert (model, provider, key, base) == ("example-model", "mixroute", "caller-key", config.base_url)


def test_mixroute_base_override_and_chat_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIXROUTE_API_KEY", "environment-key")
    monkeypatch.setenv("MIXROUTE_API_BASE", "https://mixroute.example.test/v1")
    config: Final = JSONProviderRegistry.get("mixroute")
    assert config is not None
    handler: Final = create_config_class(config)()
    base, key = handler.get_openai_compatible_provider_info(None, None)
    assert (base, key) == ("https://mixroute.example.test/v1", "environment-key")
    assert handler.get_complete_url(base, key, "example-model", {}, {}) == (
        "https://mixroute.example.test/v1/chat/completions"
    )
    assert handler.get_openai_compatible_provider_info("https://override.example.test/v1", "explicit-key") == (
        "https://override.example.test/v1", "explicit-key"
    )


def test_mixroute_native_endpoints_do_not_apply_to_other_models() -> None:
    assert JSONProviderRegistry.serves_endpoint("mixroute", "claude-opus-5-5", "/v1/messages")
    assert JSONProviderRegistry.serves_endpoint("mixroute", "mixroute/claude-sonnet-5-5", "/v1/messages")
    assert JSONProviderRegistry.serves_endpoint("mixroute", "gpt-6.1-sol", "/v1/responses")
    assert not JSONProviderRegistry.serves_endpoint("mixroute", "unlisted-model", "/v1/messages")
    assert not JSONProviderRegistry.serves_endpoint("mixroute", "unlisted-model", "/v1/responses")
    assert JSONProviderRegistry.supports_embeddings("mixroute")
