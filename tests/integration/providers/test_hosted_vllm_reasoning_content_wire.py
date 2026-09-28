import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3-reasoning"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_REASONING: Final = "I compared the two invoices and the totals differ by 42."
_TOOL_CALL_ID: Final = "call_reasoning_wire_1"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        }
    ).encode()


def test_hosted_vllm_assistant_reasoning_content_reaches_the_wire(gateway: Gateway) -> None:
    identity: Final = f"hosted-vllm-reasoning-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [
            {"role": "user", "content": "Compare these invoices."},
            {
                "role": "assistant",
                "content": "Checking the totals.",
                "reasoning_content": _REASONING,
                "tool_calls": [
                    {
                        "id": _TOOL_CALL_ID,
                        "type": "function",
                        "function": {"name": "lookup_invoice", "arguments": json.dumps({"id": "inv-7"})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "invoice total is 1042"},
        ], body["messages"]
        return Reply(body=_completion(identity, "The totals differ by 42."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Compare these invoices."},
                    {
                        "role": "assistant",
                        "content": "Checking the totals.",
                        "reasoning_content": _REASONING,
                        "tool_calls": [
                            {
                                "id": _TOOL_CALL_ID,
                                "type": "function",
                                "function": {"name": "lookup_invoice", "arguments": json.dumps({"id": "inv-7"})},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": "invoice total is 1042"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]
