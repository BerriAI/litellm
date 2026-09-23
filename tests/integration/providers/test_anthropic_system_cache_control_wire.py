import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "claude-sonnet-4-5-20250929"
_API_KEY: Final = "synthetic-anthropic-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _anthropic_reply(identity: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": _MODEL,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 3, "cache_creation_input_tokens": 12},
        }
    ).encode()


def _assert_system_block(body: dict[str, JsonValue], policy: str) -> None:
    assert body["model"] == _MODEL, body
    assert body["system"] == [{"type": "text", "text": policy, "cache_control": {"type": "ephemeral"}}], body


def test_chat_completions_system_block_list_carries_cache_control_to_anthropic_system(gateway: Gateway) -> None:
    identity: Final = f"anthropic-system-cc-{uuid.uuid4().hex}"
    policy: Final = f"policy {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        assert request.headers["x-api-key"] == _API_KEY
        _assert_system_block(_JSON_OBJECT.validate_json(request.body), policy)
        return Reply(body=_anthropic_reply(identity, "done"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": policy, "cache_control": {"type": "ephemeral"}}
                        ],
                    },
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_chat_completions_system_string_with_message_cache_control_reaches_anthropic_system(
    gateway: Gateway,
) -> None:
    identity: Final = f"anthropic-system-str-{uuid.uuid4().hex}"
    policy: Final = f"policy {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        _assert_system_block(_JSON_OBJECT.validate_json(request.body), policy)
        return Reply(body=_anthropic_reply(identity, "done"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [
                    {"role": "system", "content": policy, "cache_control": {"type": "ephemeral"}},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_responses_system_input_item_carries_cache_control_to_anthropic_system(gateway: Gateway) -> None:
    identity: Final = f"responses-system-cc-{uuid.uuid4().hex}"
    policy: Final = f"policy {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages"
        _assert_system_block(_JSON_OBJECT.validate_json(request.body), policy)
        return Reply(body=_anthropic_reply(identity, "done"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": policy, "cache_control": {"type": "ephemeral"}}
                        ],
                    },
                    {"role": "user", "content": "hi"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["status"] == "completed", response.text
        assert any(item.get("type") == "message" for item in payload.get("output", []) if isinstance(item, dict))
        assert len(wire.drain()) == 1

