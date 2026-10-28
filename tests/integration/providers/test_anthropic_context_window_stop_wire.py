import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

MODEL: Final = "claude-haiku-4-5"


def _sse(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _message(identity: str, stop_reason: str | None) -> dict[str, JsonValue]:
    return {
        "id": "msg_" + identity,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "alpha bravo"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 199878, "output_tokens": 12799},
    }


def _stream(identity: str, stop_reason: str) -> tuple[bytes, ...]:
    return (
        _sse({"type": "message_start", "message": {**_message(identity, None), "content": []}}),
        _sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "alpha bravo"}}),
        _sse({"type": "content_block_stop", "index": 0}),
        _sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 12799},
            }
        ),
        _sse({"type": "message_stop"}),
    )


def _anthropic(identity: str, stop_reason: str):
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        if json.loads(request.body).get("stream"):
            return Reply(content_type="text/event-stream", chunks=_stream(identity, stop_reason))
        return Reply(body=json.dumps(_message(identity, stop_reason)).encode())

    return respond


def _events(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    with gateway.client.stream(
        "POST", path, json={**body, "stream": True}, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return tuple(
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )


def _chat_finish_reason(gateway: Gateway, model: str, *, stream: bool) -> JsonValue:
    body: Final = {"model": model, "max_tokens": 64000, "messages": [{"role": "user", "content": "copy the list"}]}
    if stream:
        reasons: Final = tuple(
            choice["finish_reason"]
            for chunk in _events(gateway, "/v1/chat/completions", body)
            for choice in chunk["choices"]  # comprehension-ok: chunk -> choices flatten
            if choice.get("finish_reason")
        )
        assert len(reasons) == 1, reasons
        return reasons[0]
    response: Final = gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["finish_reason"]


def _responses_status(gateway: Gateway, model: str, *, stream: bool) -> JsonValue:
    body: Final = {"model": model, "max_output_tokens": 64000, "input": "copy the list"}
    if stream:
        terminal: Final = tuple(
            event["response"]
            for event in _events(gateway, "/v1/responses", body)
            if event["type"] in ("response.completed", "response.incomplete", "response.failed")
        )
        assert len(terminal) == 1, terminal
        return terminal[0]["status"]
    response: Final = gateway.request("POST", "/v1/responses", body)
    assert response.status_code == 200, response.text
    return response.json()["status"]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [("model_context_window_exceeded", "length"), ("max_tokens", "length"), ("end_turn", "stop")],
)
def test_chat_maps_anthropic_stop_reason(gateway: Gateway, stop_reason: str, finish_reason: str, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_anthropic(identity, stop_reason)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        assert _chat_finish_reason(gateway, model, stream=stream) == finish_reason


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [("model_context_window_exceeded", "incomplete"), ("max_tokens", "incomplete"), ("end_turn", "completed")],
)
def test_responses_maps_anthropic_stop_reason(gateway: Gateway, stop_reason: str, status: str, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_anthropic(identity, stop_reason)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        assert _responses_status(gateway, model, stream=stream) == status
