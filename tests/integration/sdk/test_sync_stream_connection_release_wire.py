from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import openai
import pytest
from integration._support.client import eventually
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm.types.utils import ModelResponseStream

_API_KEY: Final = "synthetic-stream-release-key"
_PROMPT: Final = "does the socket go back to the pool once the stream is done with it?"
_FIRST_WORDS: Final = "hello "
_LAST_WORD: Final = "world"
_RELEASE_SECONDS: Final = 2


def _anthropic_event(event: str, payload: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _openai_delta(text: str, finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-stream-release-wire",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


_ANTHROPIC_OPENING: Final = b"".join(
    (
        _anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream_release_wire",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        _anthropic_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _anthropic_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _FIRST_WORDS}},
        ),
    )
)
_ANTHROPIC_CLOSING: Final = (
    _anthropic_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _LAST_WORD}},
    ),
    _anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
    _anthropic_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
    ),
    _anthropic_event("message_stop", {"type": "message_stop"}),
)
_OPENAI_OPENING: Final = _openai_delta(_FIRST_WORDS, None)
_OPENAI_CLOSING: Final = (_openai_delta(_LAST_WORD, None), _openai_delta("", "stop"), b"data: [DONE]\n\n")


@dataclass(frozen=True, slots=True)
class _Shape:
    name: str
    model: str
    opening: bytes
    closing: tuple[bytes, ...]


_SHAPES: Final = (
    _Shape("anthropic", "anthropic/claude-sonnet-5-5", _ANTHROPIC_OPENING, _ANTHROPIC_CLOSING),
    _Shape("openai", "openai/gpt-5.6", _OPENAI_OPENING, _OPENAI_CLOSING),
    _Shape("databricks", "databricks/databricks-llama", _OPENAI_OPENING, _OPENAI_CLOSING),
)


def _peer(shape: _Shape, gate_after_first: threading.Event | None = None):
    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream", chunks=(shape.opening, *shape.closing), gate_after_first=gate_after_first
        )

    return respond


def _request_kwargs(shape: _Shape, wire: Wire) -> Mapping[str, object]:
    client: Final = openai.OpenAI(api_key=_API_KEY, base_url=wire.url) if shape.name == "openai" else None
    return {
        "model": shape.model,
        "api_base": wire.url,
        "api_key": _API_KEY,
        "messages": ({"role": "user", "content": _PROMPT},),
        "stream": True,
        **({"client": client} if client is not None else {}),
    }


def _text_of(chunk: ModelResponseStream) -> str:
    return chunk.choices[0].delta.content or ""


def _open_sessions_settling_to(wire: Wire, at_most: int) -> int:
    return eventually(wire.open_connections, lambda open_sessions: open_sessions <= at_most, seconds=_RELEASE_SECONDS)


async def _open_sessions_settling_to_while_the_loop_runs(wire: Wire, at_most: int) -> int:
    deadline: Final = time.monotonic() + _RELEASE_SECONDS
    while wire.open_connections() > at_most and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return wire.open_connections()


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda shape: shape.name)
def test_sync_stream_read_to_its_last_chunk_releases_the_connection(shape: _Shape) -> None:
    with wire_server(_peer(shape), keep_alive=True) as wire:
        request: Final = _request_kwargs(shape, wire)
        first_text: Final = "".join(_text_of(chunk) for chunk in litellm.completion(**request))
        second_text: Final = "".join(_text_of(chunk) for chunk in litellm.completion(**request))
        assert (first_text, second_text) == (_FIRST_WORDS + _LAST_WORD, _FIRST_WORDS + _LAST_WORD)
        assert _open_sessions_settling_to(wire, at_most=1) <= 1, "a finished stream left its TCP session open"


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda shape: shape.name)
def test_sync_stream_closed_after_its_first_chunk_releases_the_connection(shape: _Shape) -> None:
    gate: Final = threading.Event()
    with wire_server(_peer(shape, gate_after_first=gate), keep_alive=True) as wire:
        request: Final = _request_kwargs(shape, wire)
        abandoned: Final = litellm.completion(**request)
        first_chunk: Final = next(iter(abandoned))
        assert _text_of(first_chunk) == _FIRST_WORDS
        abandoned.close()
        gate.set()
        assert _open_sessions_settling_to(wire, at_most=0) == 0, "close() left the TCP session open"
        gate.clear()
        follow_up: Final = litellm.completion(**request)
        assert _text_of(next(iter(follow_up))) == _FIRST_WORDS
        gate.set()
        assert "".join(_text_of(chunk) for chunk in follow_up) == _LAST_WORD


@pytest.mark.asyncio
async def test_async_stream_read_to_its_last_chunk_releases_the_connection() -> None:
    shape: Final = _SHAPES[0]
    with wire_server(_peer(shape), keep_alive=True) as wire:
        request: Final = _request_kwargs(shape, wire)
        first_text: Final = "".join([_text_of(chunk) async for chunk in await litellm.acompletion(**request)])
        second_text: Final = "".join([_text_of(chunk) async for chunk in await litellm.acompletion(**request)])
        assert (first_text, second_text) == (_FIRST_WORDS + _LAST_WORD, _FIRST_WORDS + _LAST_WORD)
        assert await _open_sessions_settling_to_while_the_loop_runs(wire, at_most=1) <= 1, (
            "a finished stream left its TCP session open"
        )


@pytest.mark.asyncio
async def test_async_stream_aclosed_after_its_first_chunk_releases_the_connection() -> None:
    shape: Final = _SHAPES[0]
    gate: Final = threading.Event()
    with wire_server(_peer(shape, gate_after_first=gate), keep_alive=True) as wire:
        request: Final = _request_kwargs(shape, wire)
        abandoned: Final = await litellm.acompletion(**request)
        assert _text_of(await abandoned.__anext__()) == _FIRST_WORDS
        await abandoned.aclose()
        gate.set()
        assert await _open_sessions_settling_to_while_the_loop_runs(wire, at_most=0) == 0, (
            "aclose() left the TCP session open"
        )
        gate.clear()
        follow_up: Final = await litellm.acompletion(**request)
        assert _text_of(await follow_up.__anext__()) == _FIRST_WORDS
        gate.set()
        assert "".join([_text_of(chunk) async for chunk in follow_up]) == _LAST_WORD


_ANTHROPIC_ERROR_MIDDLE: Final = _anthropic_event(
    "error", {"type": "error", "error": {"type": "overloaded_error", "message": "peer overloaded mid stream"}}
)


def test_sync_stream_that_raises_mid_stream_releases_the_connection() -> None:
    shape: Final = _SHAPES[0]
    broken: Final = _Shape(shape.name, shape.model, shape.opening, (_ANTHROPIC_ERROR_MIDDLE, *shape.closing))
    with wire_server(_peer(broken), keep_alive=True) as wire:
        request: Final = _request_kwargs(broken, wire)
        with pytest.raises(Exception, match="overloaded"):
            "".join(_text_of(chunk) for chunk in litellm.completion(**request))
        assert _open_sessions_settling_to(wire, at_most=0) == 0, (
            "a stream that raised mid-way left its TCP session open"
        )


_STUCK: Final = _Shape(
    "hosted_vllm",
    "hosted_vllm/stuck-model",
    _OPENAI_OPENING,
    (_openai_delta("the model is stuck on this", None),) * (10 * litellm.REPEATED_STREAMING_CHUNK_LIMIT),
)


def _still_generating_peer(request: Request) -> Reply:
    return Reply(content_type="text/event-stream", chunks=(_STUCK.opening, *_STUCK.closing), pause_between_chunks=0.01)


def test_sync_stream_stuck_repeating_one_chunk_releases_the_connection() -> None:
    with wire_server(_still_generating_peer, keep_alive=True) as wire:
        stuck: Final = litellm.completion(**_request_kwargs(_STUCK, wire))
        with pytest.raises(Exception, match="repeating the same chunk"):
            for _ in stuck:
                pass
        assert _open_sessions_settling_to(wire, at_most=0) == 0, "a stream stuck on one chunk left its TCP session open"


@pytest.mark.asyncio
async def test_async_stream_stuck_repeating_one_chunk_releases_the_connection() -> None:
    with wire_server(_still_generating_peer, keep_alive=True) as wire:
        stuck: Final = await litellm.acompletion(**_request_kwargs(_STUCK, wire))
        with pytest.raises(Exception, match="repeating the same chunk"):
            async for _ in stuck:
                pass
        assert await _open_sessions_settling_to_while_the_loop_runs(wire, at_most=0) == 0, (
            "a stream stuck on one chunk left its TCP session open"
        )
