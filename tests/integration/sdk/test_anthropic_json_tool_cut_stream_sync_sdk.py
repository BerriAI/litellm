import base64
import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_JSON_TOOL: Final = "json_tool_call"
_FRAGMENT_SIZE: Final = 20
_CUT_EVENTS: Final = 3
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"tier": {"type": "integer"}, "summary": {"type": "string"}},
    "required": ["tier", "summary"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "risk", "schema": _SCHEMA},
}
_ANTHROPIC_MODEL: Final = "anthropic/claude-scripted-9"
_BEDROCK_INVOKE_MODEL: Final = "bedrock/invoke/anthropic.claude-sonnet-5"


def _payload(marker_hex: str) -> dict[str, JsonValue]:
    return {"tier": 2, "summary": marker_hex}


def _tool_input(shape: str, marker_hex: str) -> dict[str, JsonValue]:
    if shape == "wrapped":
        return {"tool_name": _JSON_TOOL, "parameters": _payload(marker_hex)}
    return _payload(marker_hex)


def _events(marker_hex: str, shape: str, cut: bool) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    text: Final = json.dumps(_tool_input(shape, marker_hex))
    deltas: Final = tuple(
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": text[start : start + _FRAGMENT_SIZE]},
            },
        )
        for start in range(0, len(text), _FRAGMENT_SIZE)
    )
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_{marker_hex}",
        "type": "message",
        "role": "assistant",
        "model": "claude-scripted-9",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    complete: Final = (
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": f"toolu_{marker_hex}", "name": _JSON_TOOL, "input": {}},
            },
        ),
        *deltas,
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return complete[:-_CUT_EVENTS] if cut else complete


def _aws_chunk(data: Mapping[str, JsonValue]) -> bytes:
    payload: Final = json.dumps({"bytes": base64.b64encode(json.dumps(data).encode()).decode()}).encode()
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"}, payload
    )


def _peer(marker_hex: str, shape: str, cut: bool) -> Callable[[Request], Reply]:
    events: Final = _events(marker_hex, shape, cut)

    def respond(request: Request) -> Reply:
        assert marker_hex in request.body.decode(), request.body
        if request.target.endswith("/invoke-with-response-stream"):
            return Reply(
                content_type="application/vnd.amazon.eventstream",
                chunks=tuple(_aws_chunk(data) for _, data in events),
            )
        assert request.target.endswith("/v1/messages"), request.target
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events),
        )

    return respond


def _credentials(model: str) -> dict[str, str]:
    if model == _ANTHROPIC_MODEL:
        return {"api_key": "synthetic-anthropic-key"}
    return {"api_key": "synthetic-bedrock-bearer", "aws_region_name": "us-east-1"}


@pytest.mark.parametrize("model", [_ANTHROPIC_MODEL, _BEDROCK_INVOKE_MODEL], ids=["anthropic", "bedrock_invoke"])
@pytest.mark.parametrize(
    ("shape", "cut"),
    [("plain", False), ("plain", True), ("wrapped", True)],
    ids=["plain_complete", "plain_cut", "wrapped_cut"],
)
def test_sync_stream_cut_before_the_json_tool_block_closes_still_yields_the_whole_answer(
    model: str, shape: str, cut: bool
) -> None:
    marker_hex: Final = uuid.uuid4().hex
    with wire_server(_peer(marker_hex, shape, cut)) as wire:
        response: Final = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": f"Rate the risk {marker_hex}"}],
            response_format=_RESPONSE_FORMAT,
            max_tokens=64,
            stream=True,
            api_base=wire.url,
            num_retries=0,
            **_credentials(model),
        )
        assert isinstance(response, CustomStreamWrapper), type(response)
        text: Final = "".join(str(chunk.choices[0].delta.content or "") for chunk in tuple(response))
        assert len(wire.drain()) == 1
    assert json.loads(text) == _payload(marker_hex), text
