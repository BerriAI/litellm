import base64
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Final

from integration._support.claude_code import sse_frame
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "claude-sonnet-5-5"
BEDROCK_MODEL: Final = "anthropic.claude-sonnet-5-5"
THINKING_PARTS: Final = ("alpha ", "beta")
THINKING: Final = "alpha beta"
SIGNATURE: Final = "scripted-signature-" + "s" * 32
NO_CACHE: Final = {"cache": {"no-cache": True}}
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_LIST: Final = TypeAdapter(list[JsonValue])
BLOCKS: Final = TypeAdapter(list[dict[str, JsonValue]])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STREAMING_TARGETS: Final = ("/invoke-with-response-stream", ":streamRawPredict")

Event = dict[str, JsonValue]


def prompt(marker: str) -> str:
    return f"think it through for marker-{marker}"


def answer(marker: str) -> str:
    return f"answer marker-{marker}"


def identity(marker: str) -> str:
    return f"msg_{marker}"


def marker_of(request: Request) -> str:
    found: Final = _MARKER.findall(request.body.decode())
    assert found, request.body
    return found[-1]


def _event(**fields: JsonValue) -> Event:
    return dict(fields)


def thinking_events(index: int, parts: Sequence[JsonValue], signatures: Sequence[JsonValue]) -> tuple[Event, ...]:
    start: Final = _event(
        type="content_block_start", index=index, content_block={"type": "thinking", "thinking": "", "signature": ""}
    )
    thought: Final = tuple(
        _event(type="content_block_delta", index=index, delta={"type": "thinking_delta", "thinking": part})
        for part in parts
    )
    signed: Final = tuple(
        _event(type="content_block_delta", index=index, delta={"type": "signature_delta", "signature": signature})
        for signature in signatures
    )
    return (start, *thought, *signed, _event(type="content_block_stop", index=index))


def redacted_events(index: int, data: str) -> tuple[Event, ...]:
    return (
        _event(type="content_block_start", index=index, content_block={"type": "redacted_thinking", "data": data}),
        _event(type="content_block_stop", index=index),
    )


def text_events(index: int, text: str) -> tuple[Event, ...]:
    return (
        _event(type="content_block_start", index=index, content_block={"type": "text", "text": ""}),
        _event(type="content_block_delta", index=index, delta={"type": "text_delta", "text": text}),
        _event(type="content_block_stop", index=index),
    )


def message_events(marker: str, blocks: Sequence[Sequence[Event]]) -> tuple[Event, ...]:
    start: Final = _event(
        type="message_start",
        message={
            "id": identity(marker),
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    )
    delta: Final = _event(
        type="message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 9}
    )
    return (start, *chain.from_iterable(blocks), delta, _event(type="message_stop"))


def standard_events(
    marker: str,
    *,
    parts: Sequence[JsonValue] = THINKING_PARTS,
    signatures: Sequence[JsonValue] = (SIGNATURE,),
) -> tuple[Event, ...]:
    return message_events(marker, (thinking_events(0, parts, signatures), text_events(1, answer(marker))))


def sse_chunks(events: Sequence[Event]) -> tuple[bytes, ...]:
    return tuple(sse_frame(str(event["type"]), event) for event in events)


def aws_chunks(events: Sequence[Event]) -> tuple[bytes, ...]:
    return tuple(
        _aws_event_frame(
            "chunk",
            {"bytes": base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()},
            "sc",
            "u",
        )
        for event in events
    )


def message_body(marker: str) -> bytes:
    return json.dumps(
        {
            "id": identity(marker),
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [
                {"type": "thinking", "thinking": THINKING, "signature": SIGNATURE},
                {"type": "text", "text": answer(marker)},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 9},
        }
    ).encode()


def streams(request: Request) -> bool:
    if request.target.endswith(_STREAMING_TARGETS):
        return True
    return JSON_OBJECT.validate_json(request.body).get("stream") is True


def stream_reply(request: Request, events: Sequence[Event], *, abort_after: int | None = None) -> Reply:
    if request.target.endswith("/invoke-with-response-stream"):
        return Reply(content_type=EVENT_STREAM, chunks=aws_chunks(events), abort_after=abort_after)
    return Reply(content_type="text/event-stream", chunks=sse_chunks(events), abort_after=abort_after)


def standard_peer(request: Request) -> Reply:
    marker: Final = marker_of(request)
    if streams(request):
        return stream_reply(request, standard_events(marker))
    return Reply(body=message_body(marker))


def chunks_of(text: str) -> tuple[Event, ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def delta_of(chunk: Mapping[str, JsonValue]) -> Event:
    choices: Final = JSON_LIST.validate_python(chunk.get("choices") or [])
    if not choices:
        return {}
    return JSON_OBJECT.validate_python(JSON_OBJECT.validate_python(choices[0]).get("delta") or {})


def deltas_of(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    return tuple(delta_of(chunk) for chunk in chunks)


def blocks_of(delta: Mapping[str, JsonValue]) -> tuple[Event, ...]:
    return tuple(BLOCKS.validate_python(delta.get("thinking_blocks") or []))


def all_blocks(deltas: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    return tuple(chain.from_iterable(blocks_of(delta) for delta in deltas))


def signed_blocks(deltas: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    return tuple(block for block in all_blocks(deltas) if block.get("signature"))


def reasoning_text(deltas: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(str(delta.get("reasoning_content") or "") for delta in deltas)


def content_text(deltas: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(str(delta.get("content") or "") for delta in deltas)


def thinking_block(thinking: str, signature: JsonValue) -> Event:
    return {"type": "thinking", "thinking": thinking, "signature": signature}


def signature_only(signature: JsonValue = SIGNATURE) -> Event:
    return thinking_block("", signature)


@dataclass(frozen=True, slots=True)
class _Accumulated:
    closed: tuple[Event, ...]
    text: str


def _fold(state: _Accumulated, block: Mapping[str, JsonValue]) -> _Accumulated:
    if block.get("type") == "redacted_thinking":
        redacted: Event = {"type": "redacted_thinking", "data": block.get("data")}
        return _Accumulated((*state.closed, redacted), state.text)
    text: Final = state.text + str(block.get("thinking") or "")
    signature: Final = block.get("signature")
    if not signature:
        return _Accumulated(state.closed, text)
    return _Accumulated((*state.closed, thinking_block(text, signature)), "")


def accumulate(deltas: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    return reduce(_fold, all_blocks(deltas), _Accumulated((), "")).closed


def logged_thinking(response: Mapping[str, JsonValue]) -> tuple[Event, ...]:
    if "choices" in response:
        choice: Final = JSON_OBJECT.validate_python(JSON_LIST.validate_python(response["choices"])[0])
        message: Final = JSON_OBJECT.validate_python(choice.get("message") or {})
        return tuple(BLOCKS.validate_python(message.get("thinking_blocks") or []))
    content: Final = BLOCKS.validate_python(response.get("content") or [])
    return tuple(block for block in content if block.get("type") in ("thinking", "redacted_thinking"))
