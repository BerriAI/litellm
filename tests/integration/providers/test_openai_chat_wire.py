import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_PROMPT: Final = "Summarize this conversation in one sentence."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
        }
    ).encode()


@pytest.mark.covers("providers.openai_chat_wire.tool_choice_without_tools_is_dropped_before_the_wire")
def test_openai_chat_tool_choice_without_tools_is_not_forwarded(gateway: Gateway) -> None:
    identity: Final = f"openai-toolless-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        assert "tool_choice" not in body, body
        assert "tools" not in body, body
        return Reply(body=_completion(identity, "One sentence."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "tool_choice": "none"},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "One sentence.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
