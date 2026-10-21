import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "claude-sonnet-4-5-20250929"
KEY: Final = "synthetic-anthropic-key"
SIGNATURE_ERROR: Final = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages.2.content.0.thinking.signature.str: Input should be a valid string",
        },
    }
).encode()
TOOLS: Final = ({"name": "lookup", "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}}},)


def _history_with_unsigned_thinking(identity: str) -> tuple[dict[str, object], ...]:
    return (
        {"role": "user", "content": [{"type": "text", "text": f"first question {identity}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "second question"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "replayed from another provider", "signature": None},
                {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "found"}]},
    )


@pytest.mark.covers("providers.anthropic_messages.missing_thinking_signature_400_retries_without_thinking_blocks")
def test_missing_thinking_signature_400_retries_once_without_thinking_blocks_and_returns_200(
    gateway: Gateway,
) -> None:
    identity: Final = "thinking-signature-" + uuid.uuid4().hex
    history: Final = _history_with_unsigned_thinking(identity)
    tool_use_only_turn: Final = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": "value"}}],
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == KEY
        body: Final = json.loads(request.body)
        assert body["model"] == MODEL
        assert body["tools"] == list(TOOLS), body
        if body["messages"][3]["content"][0]["type"] == "thinking":
            assert body["messages"] == list(history), body
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}, body
            return Reply(status=400, body=SIGNATURE_ERROR)
        assert body["messages"] == [*history[:3], tool_use_only_turn, history[4]], body
        assert "thinking" not in body, body
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [{"type": "text", "text": "recovered without thinking history"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 30, "output_tokens": 6},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key=KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tools": list(TOOLS),
                "messages": list(history),
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["id"] == identity, response.text
        assert body["content"] == [{"type": "text", "text": "recovered without thinking history"}], response.text
        assert body["stop_reason"] == "end_turn", response.text
        assert [request.target for request in wire.drain()] == ["/v1/messages", "/v1/messages"]
