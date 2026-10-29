import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

DEFAULT_API_BASE: Final = "http://127.0.0.1:17434/v1"
PLACEHOLDER_API_KEY: Final = "fake-api-key"
ENV: Final = {"LLMMAN_API_BASE": "http://gpu-box:17434/v1", "LLMMAN_API_KEY": "fake-environment-key"}


@pytest.fixture(autouse=True)
def _clean_llmman_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMMAN_API_BASE", raising=False)
    monkeypatch.delenv("LLMMAN_API_KEY", raising=False)


@pytest.mark.parametrize(
    "api_base, api_key, env, expected_base, expected_key",
    [
        (None, None, {}, DEFAULT_API_BASE, PLACEHOLDER_API_KEY),
        ("", "", {}, DEFAULT_API_BASE, PLACEHOLDER_API_KEY),
        (None, None, ENV, ENV["LLMMAN_API_BASE"], ENV["LLMMAN_API_KEY"]),
        ("http://other:17434/v1", "fake-explicit-key", ENV, "http://other:17434/v1", "fake-explicit-key"),
        ("http://other:17434/v1", None, ENV, "http://other:17434/v1", ENV["LLMMAN_API_KEY"]),
        (None, "fake-explicit-key", ENV, ENV["LLMMAN_API_BASE"], "fake-explicit-key"),
    ],
)
def test_llmman_base_and_key_precedence(
    monkeypatch: pytest.MonkeyPatch,
    api_base: str | None,
    api_key: str | None,
    env: dict[str, str],
    expected_base: str,
    expected_key: str,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert get_llm_provider(model="llmman/qwen3.8", api_base=api_base, api_key=api_key) == (
        "qwen3.8",
        "llmman",
        expected_key,
        expected_base,
    )


def test_llmman_chat_reaches_the_local_server() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{DEFAULT_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3.8",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
        response: Final = litellm.completion(
            model="llmman/qwen3.8", messages=[{"role": "user", "content": "Say hello"}], max_tokens=100
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert response.choices[0].message.content == "hi"
    assert request.headers["authorization"] == f"Bearer {PLACEHOLDER_API_KEY}"
    assert body["model"] == "qwen3.8"
    assert body["max_tokens"] == 100


def test_llmman_embedding_reaches_the_local_server() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{DEFAULT_API_BASE}/embeddings").respond(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "bge-m3",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
        response: Final = litellm.embedding(model="llmman/bge-m3", input=["hello"])
    request: Final = route.calls.last.request
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert request.headers["authorization"] == f"Bearer {PLACEHOLDER_API_KEY}"
    assert json.loads(request.content)["model"] == "bge-m3"
