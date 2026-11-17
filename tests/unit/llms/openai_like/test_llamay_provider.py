import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_llamay_needs_no_key_on_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMAY_API_KEY", raising=False)
    monkeypatch.delenv("LLAMAY_API_BASE", raising=False)
    assert get_llm_provider(model="llamay/qwen2.5:0.5b") == (
        "qwen2.5:0.5b",
        "llamay",
        "fake-api-key",
        "http://localhost:11435/v1",
    )


def test_llamay_reads_a_key_and_base_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMAY_API_KEY", "fake-environment-key")
    monkeypatch.setenv("LLAMAY_API_BASE", "http://gpu-box:11435/v1")
    assert get_llm_provider(model="llamay/hf.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M") == (
        "hf.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M",
        "llamay",
        "fake-environment-key",
        "http://gpu-box:11435/v1",
    )


def test_an_explicit_key_and_base_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMAY_API_KEY", "fake-environment-key")
    assert get_llm_provider(
        model="llamay/qwen2.5:0.5b", api_key="fake-explicit-key", api_base="http://other:11435/v1"
    ) == ("qwen2.5:0.5b", "llamay", "fake-explicit-key", "http://other:11435/v1")


def test_llamay_chat_reaches_the_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMAY_API_KEY", raising=False)
    monkeypatch.delenv("LLAMAY_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post("http://localhost:11435/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen2.5:0.5b",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )
        response: Final = litellm.completion(
            model="llamay/qwen2.5:0.5b", messages=[{"role": "user", "content": "Say hello"}]
        )
    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "qwen2.5:0.5b"
    assert response.choices[0].message.content == "hello"
