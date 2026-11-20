import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "http://localhost:8000/v1"
_MODEL: Final = "Qwen/Qwen2.5-1.5B-Instruct"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_turing",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Turing"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_TEXT_COMPLETION: Final = {
    "id": "cmpl_turing",
    "object": "text_completion",
    "created": 1_790_000_000,
    "model": _MODEL,
    "choices": [{"index": 0, "text": "Hello from Turing", "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 0.75]}],
    "model": _MODEL,
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


def test_turing_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURING_API_KEY", "turing-local")
    monkeypatch.delenv("TURING_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(model=f"turing/{_MODEL}")

    assert model == _MODEL
    assert provider == "turing"
    assert api_key == "turing-local"
    assert api_base == _DEFAULT_API_BASE


def test_turing_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURING_API_KEY", "turing-local")
    monkeypatch.setenv("TURING_API_BASE", "http://gpu-box.internal:8011/v1")

    _, provider, _, api_base = get_llm_provider(model=f"turing/{_MODEL}")

    assert provider == "turing"
    assert api_base == "http://gpu-box.internal:8011/v1"


def test_turing_explicit_credentials_win_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURING_API_KEY", "turing-env-key")
    monkeypatch.setenv("TURING_API_BASE", "http://gpu-box.internal:8011/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"turing/{_MODEL}",
        api_base="http://127.0.0.1:9000/v1",
        api_key="turing-explicit-key",
    )

    assert provider == "turing"
    assert api_key == "turing-explicit-key"
    assert api_base == "http://127.0.0.1:9000/v1"


def test_turing_chat_completion_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURING_API_KEY", "turing-local")
    monkeypatch.delenv("TURING_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"turing/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=8,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer turing-local"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 8
    assert response.choices[0].message.content == "Hello from Turing"


def test_turing_chat_completion_streams_from_the_env_api_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TURING_API_KEY", "turing-local")
    monkeypatch.setenv("TURING_API_BASE", "http://gpu-box.internal:8011/v1")
    chunks: Final = (
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1_790_000_000,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
        },
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1_790_000_000,
            "model": _MODEL,
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        },
    )
    sse: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    with respx.mock() as upstream:
        route: Final = upstream.post("http://gpu-box.internal:8011/v1/chat/completions").respond(
            200, text=sse, headers={"content-type": "text/event-stream"}
        )
        text: Final = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in litellm.completion(
                model=f"turing/{_MODEL}",
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
            )
        )

    assert json.loads(route.calls.last.request.content)["stream"] is True
    assert text == "Hello"


def test_turing_text_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/completions").respond(200, json=_TEXT_COMPLETION)
        response: Final = litellm.text_completion(
            model=f"turing/{_MODEL}",
            prompt="Say hello",
            max_tokens=8,
            api_key="turing-local",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer turing-local"
    assert body["model"] == _MODEL
    assert body["prompt"] == "Say hello"
    assert body["max_tokens"] == 8
    assert response.choices[0].text == "Hello from Turing"


def test_turing_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/embeddings").respond(200, json=_EMBEDDING)
        response: Final = litellm.embedding(model=f"turing/{_MODEL}", input=["hello"], api_key="turing-local")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer turing-local"
    assert body["model"] == _MODEL
    assert body["input"] == ["hello"]
    assert response.data[0]["embedding"] == [0.25, -0.5, 0.75]
