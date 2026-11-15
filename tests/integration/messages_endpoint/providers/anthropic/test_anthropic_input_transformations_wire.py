"""Anthropic's ``input_transformations`` report reaches the caller on chat completions and ``/v1/messages``."""

import json
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_API_KEY: Final = "scripted-anthropic-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CHOICES: Final = TypeAdapter(list[dict[str, JsonValue]])
_INPUT_TRANSFORMATIONS: Final[list[JsonValue]] = [{"type": "prefix_binding_mismatch", "index": 1}]
_CONTENT: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "text", "text": "Ok"},)
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 8, "output_tokens": 2}
_IDENTITY: Final = "msg_input_transformations"


def _reply_body() -> bytes:
    body: Final = _JSON_OBJECT.validate_json(cc.message_reply(_IDENTITY, _BACKEND, _CONTENT, _USAGE))
    return json.dumps({**body, "input_transformations": _INPUT_TRANSFORMATIONS}).encode()


def _stream_reply() -> tuple[bytes, ...]:
    start, *rest = cc.message_stream(_IDENTITY, _BACKEND, _CONTENT, _USAGE)
    event_name, data = cc.sse_events(start.decode())[0]
    message: Final = _JSON_OBJECT.validate_python(data["message"])
    started: Final = cc.sse_frame(
        event_name, {**data, "message": {**message, "input_transformations": _INPUT_TRANSFORMATIONS}}
    )
    return (started, *rest)


def _anthropic_upstream(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    assert request.headers["x-api-key"] == _API_KEY, request.headers
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_stream_reply())
    return Reply(body=_reply_body())


def test_chat_completion_message_carries_input_transformations(gateway: Gateway) -> None:
    with wire_server(_anthropic_upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok"}]},
        )
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["provider_specific_fields"]["input_transformations"] == _INPUT_TRANSFORMATIONS, response.text


def test_first_chat_completion_chunk_carries_input_transformations(gateway: Gateway) -> None:
    with wire_server(_anthropic_upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 16, "stream": True, "messages": [{"role": "user", "content": "Say ok"}]},
        )
        assert response.status_code == 200, response.text
        chunks: Final = [
            _JSON_OBJECT.validate_json(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        first_delta: Final = _JSON_OBJECT.validate_python(_CHOICES.validate_python(chunks[0]["choices"])[0]["delta"])
        assert first_delta["provider_specific_fields"] == {"input_transformations": _INPUT_TRANSFORMATIONS}, chunks[0]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_messages_endpoint_keeps_input_transformations(gateway: Gateway, stream: bool) -> None:
    with wire_server(_anthropic_upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "stream": stream, "messages": [{"role": "user", "content": "Say ok"}]},
        )
        assert response.status_code == 200, response.text
        if stream:
            started: Final = next(data for name, data in cc.sse_events(response.text) if name == "message_start")
            assert _JSON_OBJECT.validate_python(started["message"])["input_transformations"] == _INPUT_TRANSFORMATIONS
        else:
            assert response.json()["input_transformations"] == _INPUT_TRANSFORMATIONS, response.text
