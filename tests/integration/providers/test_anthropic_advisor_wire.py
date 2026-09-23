import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_ADVISOR_KEY: Final = "synthetic-advisor-key"
_QUESTION: Final = "which index should this query use"
_ADVICE: Final = "use the composite index on (tenant_id, created_at)"
_FINAL_ANSWER: Final = "done, the composite index is the right one"


_ADVISOR_CALL_MESSAGE: Final = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "advisor-call",
            "type": "function",
            "function": {"name": "advisor", "arguments": json.dumps({"question": _QUESTION})},
        }
    ],
}
_FINAL_MESSAGE: Final = {"role": "assistant", "content": _FINAL_ANSWER}


def _chat_completion(identity: str, message: dict[str, object], finish_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{identity}",
                "object": "chat.completion",
                "created": 1,
                "model": "llama-3.3-70b-versatile",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ).encode()
    )


def _executor_reply(body: dict[str, object], identity: str) -> Reply:
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    if any(message.get("role") == "tool" for message in messages):
        assert messages[-1]["content"] == _ADVICE
        return _chat_completion(identity, _FINAL_MESSAGE, "stop")
    tools: Final = body["tools"]
    assert isinstance(tools, list)
    assert tools[0]["function"]["name"] == "advisor"
    return _chat_completion(identity, _ADVISOR_CALL_MESSAGE, "tool_calls")


@pytest.mark.covers("providers.anthropic_messages_advisor.sub_call_uses_the_configured_advisor_deployment")
def test_advisor_sub_call_reaches_the_router_deployment_with_its_key_instead_of_anthropic_unauthenticated(
    gateway: Gateway,
) -> None:
    identity: Final = "advisor-wire-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        if request.target == "/v1/chat/completions":
            assert request.headers["authorization"] == "Bearer integration-provider-key"
            return _executor_reply(body, identity)
        assert request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _ADVISOR_KEY
        assert body["model"] == "claude-opus-4-1-20250805"
        assert body["messages"] == [
            {"role": "user", "content": "please plan the migration"},
            {"role": "user", "content": _QUESTION},
        ]
        assert "tools" not in body
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg-{identity}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-1-20250805",
                    "content": [{"type": "text", "text": _ADVICE}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 6},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        executor: Final = scenario.model(model="hosted_vllm/llama-3.3-70b", api_base=wire.url + "/v1")
        advisor: Final = scenario.model(
            model="anthropic/claude-opus-4-1-20250805", api_base=wire.url, api_key=_ADVISOR_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": executor,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "please plan the migration"}],
                "tools": [{"type": "advisor_20260301", "name": "advisor", "model": advisor}],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["content"] == [{"type": "text", "text": _FINAL_ANSWER}], response.text
        assert body["stop_reason"] == "end_turn", response.text
        assert [request.target for request in wire.drain()] == [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/chat/completions",
        ]
