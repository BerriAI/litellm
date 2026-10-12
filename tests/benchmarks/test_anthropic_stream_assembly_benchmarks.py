"""Measure post-stream reconstruction for 5000 real SSE delta events."""

import json
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from pydantic import BaseModel, TypeAdapter

from litellm.llms.anthropic.pass_through.stream_assembly import (
    build_complete_streaming_response,
    build_complete_streaming_response_legacy,
)
from litellm.types.utils import ModelResponse

_MODEL: Final = "claude-sonnet-5-5"
_DELTA_COUNT: Final = 5000
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_EMPTY_INPUT: Final[Mapping[str, object]] = {}


def _frame(data: Mapping[str, object]) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"


def _delta_part(kind: str, index: int) -> str:
    return (
        '{"value":"'
        if kind == "tool_use" and index == 0
        else '"}'
        if kind == "tool_use" and index == _DELTA_COUNT - 1
        else "x"
    )


def _block_events(kind: str) -> Iterator[str]:
    block: Final[Mapping[str, object]] = (
        {"type": "tool_use", "id": "tool_benchmark", "name": "lookup", "input": _EMPTY_INPUT}
        if kind == "tool_use"
        else {"type": kind, kind: ""}
    )
    yield _frame({"type": "content_block_start", "index": 0, "content_block": block})
    delta_type, field = {
        "text": ("text_delta", "text"),
        "thinking": ("thinking_delta", "thinking"),
        "tool_use": ("input_json_delta", "partial_json"),
    }[kind]
    for index in range(_DELTA_COUNT):
        yield _frame(
            {"type": "content_block_delta", "index": 0, "delta": {"type": delta_type, field: _delta_part(kind, index)}}
        )
    if kind == "thinking":
        yield _frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "benchmark-signature"},
            }
        )
    yield _frame({"type": "content_block_stop", "index": 0})


def _build(chunks: tuple[str, ...], *, legacy: bool = False) -> ModelResponse:
    builder: Final = build_complete_streaming_response_legacy if legacy else build_complete_streaming_response
    response: Final = builder(chunks, None, _MODEL)  # pyright: ignore[reportArgumentType]  # synthetic SSE frames need no logging object
    assert isinstance(response, ModelResponse)
    return response


def _payload(response: ModelResponse) -> Mapping[str, object]:
    return _JSON_OBJECT.validate_json(BaseModel.model_dump_json(response, exclude={"id", "created"}))


def _completion_tokens(response: ModelResponse) -> object:
    return _JSON_OBJECT.validate_python(_payload(response)["usage"])["completion_tokens"]


@pytest.fixture
def assembled_chunks(kind: str) -> tuple[str, ...]:
    start: Final = _frame(
        {
            "type": "message_start",
            "message": {
                "id": "msg_benchmark",
                "type": "message",
                "role": "assistant",
                "model": _MODEL,
                "content": [],
                "usage": {"input_tokens": 13, "output_tokens": 1},
            },
        }
    )
    end: Final = (
        _frame(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": _DELTA_COUNT}}
        ),
        _frame({"type": "message_stop"}),
    )
    chunks: Final = (start,) + tuple(_block_events(kind)) + end
    native: Final = _build(chunks, legacy=True)
    optimized: Final = _build(chunks)
    assert _payload(optimized) == _payload(native)
    assert _completion_tokens(optimized) == _DELTA_COUNT
    return chunks


@pytest.mark.benchmark
@pytest.mark.parametrize("kind", ["text", "thinking", "tool_use"], ids=["pure-text", "signed-thinking", "tool-json"])
def test_anthropic_stream_assembly_5000_deltas(assembled_chunks: tuple[str, ...]) -> None:
    _build(assembled_chunks)
