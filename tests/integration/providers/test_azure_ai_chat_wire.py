import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "kimi-k2-thinking"
_API_KEY: Final = "synthetic-azure-ai-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_THINKING_BLOCK: Final[JsonValue] = {
    "type": "thinking",
    "thinking": "The user wants the sum of 17 and 26.",
    "signature": "synthetic-signature",
}
_HISTORY_WITH_ANTHROPIC_FIELDS: Final[JsonValue] = [
    {
        "role": "system",
        "content": "You are a calculator.",
        "cache_control": {"type": "ephemeral"},
    },
    {"role": "user", "content": "What is 17 + 26?"},
    {
        "role": "assistant",
        "content": "43",
        "thinking_blocks": [_THINKING_BLOCK],
        "provider_specific_fields": {"citations": None},
    },
    {"role": "user", "content": "And doubled?"},
]
_HISTORY_AS_OPENAI_SPEC: Final[JsonValue] = [
    {"role": "system", "content": "You are a calculator."},
    {"role": "user", "content": "What is 17 + 26?"},
    {"role": "assistant", "content": "43"},
    {"role": "user", "content": "And doubled?"},
]


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "86"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 31, "completion_tokens": 2, "total_tokens": 33},
        }
    ).encode()


@pytest.mark.covers("providers.azure_ai.anthropic_message_fields_are_stripped_before_foundry")
def test_azure_ai_strips_thinking_blocks_and_cache_control_from_forwarded_messages(gateway: Gateway) -> None:
    identity: Final = f"azure-ai-strip-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["messages"] == _HISTORY_AS_OPENAI_SPEC
        return Reply(body=_completion(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"azure_ai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _HISTORY_WITH_ANTHROPIC_FIELDS},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "86"},
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
