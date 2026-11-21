import os
from unittest.mock import patch

TRUSTEDROUTER_API_BASE = "https://api.trustedrouter.com/v1"


def test_trustedrouter_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.TRUSTEDROUTER.value == "trustedrouter"
    assert litellm.LlmProviders("trustedrouter") == litellm.LlmProviders.TRUSTEDROUTER
    assert JSONProviderRegistry.exists("trustedrouter")
    config = JSONProviderRegistry.get("trustedrouter")
    assert config is not None
    assert config.base_url == TRUSTEDROUTER_API_BASE
    assert config.api_key_env == "TRUSTEDROUTER_API_KEY"
    assert config.api_base_env == "TRUSTEDROUTER_API_BASE"
    assert "/v1/chat/completions" in config.supported_endpoints
    assert "/v1/responses" in config.supported_endpoints


def test_trustedrouter_listed_in_openai_compatible_providers():
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    assert "trustedrouter" in openai_compatible_providers
    assert TRUSTEDROUTER_API_BASE in openai_compatible_endpoints


def test_trustedrouter_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("trustedrouter"))()

    with patch.dict(
        os.environ,
        {
            "TRUSTEDROUTER_API_KEY": "test-key",
            "TRUSTEDROUTER_API_BASE": "https://example.com/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://example.com/v1"
    assert api_key == "test-key"


def test_trustedrouter_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("trustedrouter/zdr")

    assert model == "zdr"
    assert provider == "trustedrouter"
    assert api_base == TRUSTEDROUTER_API_BASE


_TRUSTEDROUTER_CHAT_URL = f"{TRUSTEDROUTER_API_BASE}/chat/completions"
_TRUSTEDROUTER_CHAT_COMPLETION = {
    "id": "chatcmpl_trustedrouter",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "zdr",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}


def test_trustedrouter_chat_never_borrows_the_openai_key(monkeypatch):
    import pytest
    import respx

    import litellm

    monkeypatch.delenv("TRUSTEDROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TRUSTEDROUTER_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    with respx.mock(assert_all_called=False) as upstream:
        route = upstream.post(_TRUSTEDROUTER_CHAT_URL).respond(
            200, json=_TRUSTEDROUTER_CHAT_COMPLETION
        )
        with pytest.raises(litellm.AuthenticationError, match="TRUSTEDROUTER_API_KEY"):
            litellm.completion(
                model="trustedrouter/zdr",
                messages=[{"role": "user", "content": "Say hello"}],
            )

    assert route.call_count == 0


def test_trustedrouter_chat_sends_the_explicit_key(monkeypatch):
    import respx

    import litellm

    monkeypatch.delenv("TRUSTEDROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TRUSTEDROUTER_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    with respx.mock() as upstream:
        route = upstream.post(_TRUSTEDROUTER_CHAT_URL).respond(
            200, json=_TRUSTEDROUTER_CHAT_COMPLETION
        )
        response = litellm.completion(
            model="trustedrouter/zdr",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sk-tr-test",
        )

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-tr-test"
    assert response.choices[0].message.content == "Hello"
