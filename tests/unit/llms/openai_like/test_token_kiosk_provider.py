import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_token_kiosk_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKEN_KIOSK_API_KEY", "fake-token-kiosk-key")
    assert get_llm_provider(model="token_kiosk/claude-3-5-sonnet") == (
        "claude-3-5-sonnet",
        "token_kiosk",
        "fake-token-kiosk-key",
        "https://api-token-kiosk.gaib.ai/v1",
    )
    assert get_llm_provider(
        model="token_kiosk/claude-3-5-sonnet",
        api_key="explicit-key",
        api_base="https://custom-router.example/v1",
    ) == (
        "claude-3-5-sonnet",
        "token_kiosk",
        "explicit-key",
        "https://custom-router.example/v1",
    )


def test_token_kiosk_chat_completion_dispatch() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api-token-kiosk.gaib.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-token-kiosk-test",
                "object": "chat.completion",
                "created": 1724000000,
                "model": "claude-3-5-sonnet",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Token Kiosk!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        response: Final = litellm.completion(
            model="token_kiosk/claude-3-5-sonnet",
            api_key="test-key",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert route.called
        assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "claude-3-5-sonnet"
        assert response.choices[0].message.content == "Hello from Token Kiosk!"
