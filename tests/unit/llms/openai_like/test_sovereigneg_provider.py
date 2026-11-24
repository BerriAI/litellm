import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_BASE_URL: Final = "https://backend.sovereigneg.com/v1"
_CHAT_URL: Final = f"{_BASE_URL}/chat/completions"
_EMBEDDINGS_URL: Final = f"{_BASE_URL}/embeddings"


def _chat_body(content: str = "Hello from SovereignEG") -> dict[str, object]:
    return {
        "id": "chatcmpl_sovereigneg",
        "object": "chat.completion",
        "created": 1_792_000_000,
        "model": "gpt-oss-20b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def test_sovereigneg_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOVEREIGNEG_API_KEY", "sovereigneg-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="sovereigneg/gpt-oss-20b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-oss-20b"
    assert provider == "sovereigneg"
    assert api_key == "sovereigneg-test-key"
    assert api_base == _BASE_URL


def test_sovereigneg_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOVEREIGNEG_API_KEY", "sovereigneg-test-key")
    monkeypatch.setenv("SOVEREIGNEG_API_BASE", "https://sovereigneg.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="sovereigneg/gpt-oss-20b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "sovereigneg"
    assert api_base == "https://sovereigneg.internal.example/v1"


def test_sovereigneg_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOVEREIGNEG_API_KEY", "sovereigneg-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="sovereigneg/gpt-oss-20b",
        custom_llm_provider=None,
        api_base="https://sovereigneg.internal.example/v1",
        api_key="sovereigneg-explicit-key",
    )

    assert provider == "sovereigneg"
    assert api_key == "sovereigneg-explicit-key"
    assert api_base == "https://sovereigneg.internal.example/v1"


def test_sovereigneg_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    sovereigneg: Final = next(provider for provider in providers if provider["litellm_provider"] == "sovereigneg")

    assert sovereigneg["provider"] == "SOVEREIGNEG"
    assert sovereigneg["provider_display_name"] == "SovereignEG"
    assert sovereigneg["default_model_placeholder"] == "sovereigneg/gpt-oss-20b"
    assert {field["key"]: field["required"] for field in sovereigneg["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_sovereigneg_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_chat_body())
        response: Final = litellm.completion(
            model="sovereigneg/gpt-oss-20b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sovereigneg-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_URL
    assert request.headers["authorization"] == "Bearer sovereigneg-test-key"
    assert body["model"] == "gpt-oss-20b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from SovereignEG"


def test_sovereigneg_sends_max_completion_tokens_as_max_tokens():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_chat_body())
        litellm.completion(
            model="sovereigneg/gpt-oss-20b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="sovereigneg-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


def test_sovereigneg_forwards_tools():
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_chat_body())
        litellm.completion(
            model="sovereigneg/gpt-oss-20b",
            messages=[{"role": "user", "content": "ping"}],
            tools=tools,
            api_key="sovereigneg-test-key",
        )

    assert json.loads(route.calls.last.request.content)["tools"] == tools


def test_sovereigneg_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_EMBEDDINGS_URL).respond(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "embeddinggemma-300m",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
        response: Final = litellm.embedding(
            model="sovereigneg/embeddinggemma-300m",
            input=["مرحبا"],
            api_key="sovereigneg-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert str(request.url) == _EMBEDDINGS_URL
    assert request.headers["authorization"] == "Bearer sovereigneg-test-key"
    assert body["model"] == "embeddinggemma-300m"
    assert body["input"] == ["مرحبا"]
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
