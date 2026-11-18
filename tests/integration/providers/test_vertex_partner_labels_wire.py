import base64
import json
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-5-5"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_BACKEND}"
_TARGET: Final = f"{_MODEL_PATH}:rawPredict"
_STREAM_TARGET: Final = f"{_MODEL_PATH}:streamRawPredict?alt=sse"
_LABELS_HEADER: Final = "x-vertex-ai-labels"
_REPLY_TEXT: Final = "labels control"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LABELS: Final[dict[str, JsonValue]] = {"team": "triage", "env": "prod"}

_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_vertex_labels",
    "type": "message",
    "role": "assistant",
    "model": _BACKEND,
    "content": [{"type": "text", "text": _REPLY_TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
_EVENTS: Final[tuple[dict[str, JsonValue], ...]] = (
    {"type": "message_start", "message": {**_REPLY, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _REPLY_TEXT}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    },
    {"type": "message_stop"},
)
_SSE: Final = tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in _EVENTS)


def _peer(request: Request) -> Reply:
    if request.target == "/_oauth/token":
        return Reply(
            body=json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()
        )
    if request.target == _STREAM_TARGET:
        return Reply(content_type="text/event-stream", chunks=_SSE)
    if request.target == _TARGET:
        return Reply(body=json.dumps(_REPLY).encode())
    return Reply(status=404, body=json.dumps({"error": f"unscripted target {request.target}"}).encode())


def _deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, api_base),
        num_retries=0,
    )


def _chat(gateway: Gateway, wire: Wire, scenario: Scenario, stream: bool, **extra: JsonValue) -> Request:
    model: Final = _deployment(scenario, wire.url)
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply ok"}],
            "max_tokens": 16,
            "stream": stream,
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    assert _REPLY_TEXT in response.text, response.text
    (sent,) = tuple(request for request in wire.drain() if request.target in (_TARGET, _STREAM_TARGET))
    assert sent.target == (_STREAM_TARGET if stream else _TARGET), sent.target
    return sent


def _decoded_labels(sent: Request) -> JsonValue:
    return json.loads(base64.b64decode(sent.headers[_LABELS_HEADER]))


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_regression_vertex_claude_labels_move_from_body_to_header(gateway: Gateway, stream: bool) -> None:
    with gateway.scenario() as scenario, wire_server(_peer) as wire:
        sent: Final = _chat(gateway, wire, scenario, stream, labels=_LABELS)

    assert "labels" not in _JSON_OBJECT.validate_json(sent.body), sent.body
    assert _decoded_labels(sent) == _LABELS, sent.headers


def test_guard_vertex_claude_without_labels_sends_no_labels_header(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, wire_server(_peer) as wire:
        sent: Final = _chat(gateway, wire, scenario, False)

    assert "labels" not in _JSON_OBJECT.validate_json(sent.body), sent.body
    assert _LABELS_HEADER not in sent.headers, sent.headers


def test_regression_vertex_claude_caller_labels_header_wins_over_labels_param(gateway: Gateway) -> None:
    own_header: Final = base64.b64encode(json.dumps({"team": "caller"}).encode()).decode()
    with gateway.scenario() as scenario, wire_server(_peer) as wire:
        sent: Final = _chat(
            gateway, wire, scenario, False, labels=_LABELS, extra_headers={"X-Vertex-AI-Labels": own_header}
        )

    assert "labels" not in _JSON_OBJECT.validate_json(sent.body), sent.body
    assert sent.headers[_LABELS_HEADER] == own_header, sent.headers
