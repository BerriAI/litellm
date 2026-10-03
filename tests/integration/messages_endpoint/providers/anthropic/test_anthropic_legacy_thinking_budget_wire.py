import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-sonnet-4-6"
_KEY: Final = "synthetic-anthropic-key"
_THINKING: Final = {"type": "enabled", "budget_tokens": 8000}
_TOOL: Final = {
    "name": "read_file",
    "description": "read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
_NEXT_CALL: Final = {"type": "tool_use", "id": "call-2", "name": "read_file", "input": {"path": "schema.prisma"}}


def _tool_loop_history(identity: str) -> tuple[dict[str, object], ...]:
    return (
        {"role": "user", "content": f"open the config for {identity}"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-1", "name": "read_file", "input": {"path": "config.yaml"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "model_list: []"}]},
    )


def _tool_use_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg-{identity}",
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [_NEXT_CALL],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 40, "output_tokens": 12},
            }
        ).encode()
    )


@pytest.mark.covers("providers.anthropic_messages.claude_4_6_legacy_thinking_budget_reaches_the_wire_unchanged")
def test_claude_4_6_thinking_budget_tokens_on_messages_is_forwarded_instead_of_rewritten_to_adaptive(
    gateway: Gateway,
) -> None:
    identity: Final = "legacy-thinking-" + uuid.uuid4().hex
    history: Final = _tool_loop_history(identity)

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _KEY
        body: Final = json.loads(request.body)
        assert body["thinking"] == _THINKING, body
        assert "output_config" not in body, body
        assert body["max_tokens"] == 32768, body
        assert body["messages"] == list(history), body
        assert body["tools"] == [_TOOL], body
        return _tool_use_reply(identity)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 32768, "thinking": _THINKING, "messages": history, "tools": [_TOOL]},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["content"] == [_NEXT_CALL], response.text
        assert body["stop_reason"] == "tool_use", response.text
        assert len(wire.drain()) == 1
