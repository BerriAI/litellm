import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _claude_code_user_id(device_id: str, session_id: str) -> str:
    return json.dumps({"device_id": device_id, "account_uuid": "", "session_id": session_id})


def _responses_reply(identity: str) -> bytes:
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
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }
    ).encode()


def test_prompt_cache_key_is_derived_from_claude_code_session_id_not_device_id(gateway: Gateway) -> None:
    identity: Final = f"claude-code-cache-key-{uuid.uuid4().hex}"
    device_one: Final = "a" * 64
    device_two: Final = "b" * 64
    session_one: Final = str(uuid.uuid4())
    session_two: Final = str(uuid.uuid4())

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)

        def send(user_id: str, probe: str) -> None:
            response: Final = gateway.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 16,
                    "metadata": {"user_id": user_id},
                    "messages": [{"role": "user", "content": probe}],
                },
            )
            assert response.status_code == 200, response.text

        send(_claude_code_user_id(device_one, session_one), f"probe one {identity}")
        send(_claude_code_user_id(device_one, session_two), f"probe two {identity}")
        send(_claude_code_user_id(device_two, session_two), f"probe three {identity}")

        keys: Final = [
            _JSON_OBJECT.validate_json(request.body).get("prompt_cache_key") for request in wire.drain()
        ]
        assert keys[0] == session_one, keys
        assert keys[1] == session_two, keys
        assert keys[2] == session_two, keys
        assert keys[0] != keys[1] and keys[1] == keys[2]


def test_explicit_prompt_cache_key_wins_over_derived_session_key(gateway: Gateway) -> None:
    identity: Final = f"claude-code-explicit-key-{uuid.uuid4().hex}"
    explicit: Final = "explicit-client-cache-key"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["prompt_cache_key"] == explicit, body
        return Reply(body=_responses_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "prompt_cache_key": explicit,
                "metadata": {"user_id": _claude_code_user_id("c" * 64, str(uuid.uuid4()))},
                "messages": [{"role": "user", "content": f"explicit key probe {identity}"}],
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
