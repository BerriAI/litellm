import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_CORRECTION: Final = "Stop refactoring the parser and only fix the failing test instead."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _responses_reply(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": _BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
        }
    ).encode()


@pytest.mark.covers("providers.anthropic_messages_openai_bridge.midturn_system_correction_reaches_the_wire")
def test_midturn_system_correction_is_forwarded_to_openai_responses(gateway: Gateway) -> None:
    identity: Final = f"openai-midturn-system-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["instructions"] == "You are a coding agent."
        assert body["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the failing test."}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will start by refactoring the parser."}],
            },
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": _CORRECTION}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Continue."}]},
        ], body
        return Reply(body=_responses_reply(identity, "Understood."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "system": "You are a coding agent.",
                "messages": [
                    {"role": "user", "content": "Fix the failing test."},
                    {"role": "assistant", "content": "I will start by refactoring the parser."},
                    {"role": "system", "content": _CORRECTION},
                    {"role": "user", "content": "Continue."},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": "Understood."}], response.text
        assert payload["stop_reason"] == "end_turn", response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
