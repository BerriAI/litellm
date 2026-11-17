import json
from collections.abc import Callable
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "openai.gpt-5.6-sol"
_API_KEY: Final = "synthetic-mantle-bearer"
_PROMPT: Final = "synthetic long conversation control"
_PROMPT_TOKENS: Final = 1055489
_MODEL_MAXIMUM: Final = 1050000
_RESPONSES_PATH: Final = "/openai/v1/responses"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OVERFLOW_BODY: Final = json.dumps(
    {
        "error": {
            "code": "validation_error",
            "message": f"prompt tokens ({_PROMPT_TOKENS}) exceed model maximum ({_MODEL_MAXIMUM}) for {_BACKEND}",
            "type": "invalid_request_error",
        }
    }
).encode()


def _overflow_peer(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target == _RESPONSES_PATH
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _BACKEND
    assert _PROMPT in json.dumps(body["input"]), body
    return Reply(status=400, body=_OVERFLOW_BODY)


@pytest.mark.covers("other.provider_wire.bedrock_mantle.context_overflow_is_reported_as_prompt_too_long")
def test_bedrock_mantle_context_overflow_returns_400_saying_prompt_is_too_long(gateway: Gateway) -> None:
    with wire_server(_overflow_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        )
        assert response.status_code == 400, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert error["code"] == "400", response.text
        message: Final = error["message"]
        assert isinstance(message, str), response.text
        assert f"prompt is too long: {_PROMPT_TOKENS} tokens > {_MODEL_MAXIMUM} maximum" in message, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _RESPONSES_PATH)]


_ACCESS_KEY: Final = "AKIAINTEGRATION000003"
_SIGV4_PROMPT: Final = "synthetic sigv4 bridge control"
_SIGV4_RESPONSE: Final = json.dumps(
    {
        "id": "resp_synthetic_mantle_sigv4",
        "object": "response",
        "created_at": 1789788253,
        "status": "completed",
        "model": _BACKEND,
        "output": [
            {
                "type": "message",
                "id": "msg_synthetic_mantle_sigv4",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "mantle sigv4 wire control", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 21, "output_tokens": 4, "total_tokens": 25},
    }
).encode()


def _sigv4_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
    assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={_ACCESS_KEY}/"), dict(
        request.headers
    )
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _BACKEND, body
    assert _SIGV4_PROMPT in json.dumps(body["input"]), body
    return Reply(body=_SIGV4_RESPONSE)


@pytest.mark.covers("providers.bedrock_mantle.chat_bridge_keeps_deployment_aws_credentials_for_sigv4")
def test_chat_completions_bridge_signs_mantle_responses_request_with_deployment_aws_keys(gateway: Gateway) -> None:
    with wire_server(_sigv4_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock_mantle/{_BACKEND}",
            api_base=wire.url,
            api_key=None,
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key="synthetic-secret-key-for-testing",
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _SIGV4_PROMPT}]},
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        choices: Final = body["choices"]
        assert isinstance(choices, list) and len(choices) == 1, response.text
        choice: Final = choices[0]
        assert isinstance(choice, dict), response.text
        assert choice["message"] == {"role": "assistant", "content": "mantle sigv4 wire control"}, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _RESPONSES_PATH)]


_CLAUDE_BACKEND: Final = "anthropic.claude-sonnet-5-v1:0"
_MESSAGES_PATH: Final = "/anthropic/v1/messages"
_STREAM_EVENTS: Final = (
    (
        "message_start",
        {
            "message": {
                "id": "msg_mantle_stream",
                "type": "message",
                "role": "assistant",
                "model": _CLAUDE_BACKEND,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 1},
            }
        },
    ),
    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "mantle "}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "stream control"}}),
    ("content_block_stop", {"index": 0}),
    ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 4}}),
    ("message_stop", {}),
)
_STREAM_FRAMES: Final = tuple(
    f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode() for kind, payload in _STREAM_EVENTS
)


def _streaming_messages_peer(prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == _MESSAGES_PATH
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _CLAUDE_BACKEND, body
        assert body["stream"] is True, body
        assert body["messages"] == [{"role": "user", "content": prompt}], body
        return Reply(content_type="text/event-stream", chunks=_STREAM_FRAMES)

    return respond


@pytest.mark.covers("providers.bedrock_mantle.messages_stream_sends_stream_true_and_relays_sse_events")
def test_bedrock_mantle_messages_stream_relays_anthropic_sse_instead_of_failing_on_event_stream_decode(
    gateway: Gateway,
) -> None:
    prompt: Final = f"synthetic mantle stream control {uuid4().hex}"
    with wire_server(_streaming_messages_peer(prompt)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock_mantle/{_CLAUDE_BACKEND}", api_base=wire.url, api_key=_API_KEY, aws_region_name="us-east-1"
        )
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
            events: Final = tuple(
                _JSON_OBJECT.validate_json(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            )
        assert tuple(event["type"] for event in events) == tuple(kind for kind, _ in _STREAM_EVENTS), events
        assert (
            "".join(str(event["delta"]["text"]) for event in events if event["type"] == "content_block_delta")
            == "mantle stream control"
        ), events
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]
