import json
from typing import Final

import pytest
import respx

import litellm


def test_modal_completion_uses_proxy_token_and_bare_endpoint_id(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("MODAL_PROXY_TOKEN", "wk-test.ws-test")
    endpoint: Final = "test-endpoint.us-west.modal.direct"
    messages: Final = [{"role": "user", "content": "Hello"}]
    tools: Final = [
        {
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    url: Final = "https://inference.us-west.modal.direct/v1/chat/completions"
    route: Final = respx_mock.post(url).respond(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": endpoint,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    response: Final = litellm.completion(model=f"modal/{endpoint}", messages=messages, tools=tools, tool_choice="auto")

    assert response.choices[0].message.content == "Hello"
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer wk-test.ws-test"
    assert str(request.url) == url, "https://modal.com/docs/guide/endpoint-integrations.md (2026-10-28)"
    assert json.loads(request.content) == {
        "model": endpoint,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
