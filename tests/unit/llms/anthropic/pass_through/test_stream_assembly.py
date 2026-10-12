import copy
import json
from collections.abc import Iterator
from typing import Final
from unittest.mock import patch

import pytest

from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream,
    build_complete_streaming_response,
    build_complete_streaming_response_legacy,
    collapse_pure_text_chunks,
    compact_content_block_deltas,
    is_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)
from litellm.types.utils import Choices, ModelResponse, ModelResponseStream


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


MESSAGE_ID: Final = "msg_upstream_1"
MODEL: Final = "claude-upstream"

ANTHROPIC_STREAM: Final = (
    _frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": MESSAGE_ID,
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ),
    _frame(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
    ),
    _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}},
    ),
    _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
    _frame(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    ),
    _frame("message_stop", {"type": "message_stop"}),
)

GOOGLE_STREAM: Final = (b'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n\n',)
ANTHROPIC_ERROR: Final = _frame("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
CHAT_ERROR: Final = b'data: {"error": {"message": "blocked by guardrail"}}\n\n'
INVALID_UTF8: Final = (b"\xff\xfe data",)


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((b"data: {}\n\n",), True),
        (("data: {}\n\n",), True),
        ((object(), b"data: {}\n\n"), True),
        ((object(),), False),
        ((), False),
    ],
)
def test_is_raw_sse_stream_is_true_only_when_a_chunk_is_str_or_bytes(chunks: tuple[object, ...], expected: bool):
    assert is_raw_sse_stream(chunks) is expected


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (ANTHROPIC_STREAM, True),
        ((ANTHROPIC_ERROR,), True),
        (tuple(chunk.decode() for chunk in ANTHROPIC_STREAM), True),
        (GOOGLE_STREAM, False),
        (INVALID_UTF8, False),
        ((), False),
    ],
)
def test_is_anthropic_sse_stream_decides_on_the_event_types_present(chunks: tuple[object, ...], expected: bool):
    assert is_anthropic_sse_stream(chunks) is expected


def test_assemble_anthropic_sse_stream_joins_the_text_deltas():
    assembled = assemble_anthropic_sse_stream(ANTHROPIC_STREAM)

    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    assert choice.message.content == "Hello there"


def test_assemble_anthropic_sse_stream_restores_upstream_identity_only_when_asked():
    plain = assemble_anthropic_sse_stream(ANTHROPIC_STREAM)
    restored = assemble_anthropic_sse_stream(ANTHROPIC_STREAM, restore_identity=True)

    assert plain is not None and restored is not None
    assert plain.id != MESSAGE_ID
    assert (restored.id, restored.model) == (MESSAGE_ID, MODEL)


@pytest.mark.parametrize(
    "chunks",
    [GOOGLE_STREAM, INVALID_UTF8, (ANTHROPIC_ERROR,), ()],
    ids=["no-message-start", "undecodable", "error-only", "empty"],
)
def test_assemble_anthropic_sse_stream_returns_none_without_a_message_to_build(chunks: tuple[object, ...]):
    assert assemble_anthropic_sse_stream(chunks) is None


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ((ANTHROPIC_ERROR,), True),
        ((CHAT_ERROR,), True),
        ((ANTHROPIC_ERROR, CHAT_ERROR), True),
        ((ANTHROPIC_ERROR, ANTHROPIC_STREAM[0]), False),
        ((object(), ANTHROPIC_ERROR), False),
        (ANTHROPIC_STREAM, False),
        (INVALID_UTF8, False),
        ((b"\n\n",), False),
    ],
    ids=[
        "anthropic-error",
        "chat-error",
        "both-error-forms",
        "error-then-message",
        "typed-chunk-mixed-in",
        "normal-stream",
        "undecodable",
        "no-events",
    ],
)
def test_is_sse_error_stream_is_true_only_for_frames_that_are_all_errors(chunks: tuple[object, ...], expected: bool):
    assert is_sse_error_stream(chunks) is expected


PRICING_MODEL: Final = "claude-sonnet-5-5"


def _block_frames(index: int, kind: str, parts: tuple[str, ...], *, complete: bool, signed: bool) -> Iterator[bytes]:
    fields: Final = {"text": "text", "thinking": "thinking", "tool_use": "partial_json"}
    types: Final = {"text": "text_delta", "thinking": "thinking_delta", "tool_use": "input_json_delta"}
    content: Final = (
        {"type": kind, "id": f"tool_{index}", "name": f"lookup_{index}", "input": {}}
        if kind == "tool_use"
        else {"type": kind, kind: ""}
    )
    yield _frame("content_block_start", {"type": "content_block_start", "index": index, "content_block": content})
    for part in parts:
        yield _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": types[kind], fields[kind]: part}},
        )
    if kind == "thinking" and signed:
        for signature in ("signature-head", "signature-tail"):
            yield _frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                },
            )
    if complete:
        yield _frame("content_block_stop", {"type": "content_block_stop", "index": index})


def _all_block_frames(
    blocks: tuple[tuple[str, tuple[str, ...]], ...], *, complete: bool, signed: bool
) -> Iterator[bytes]:
    for index, (kind, parts) in enumerate(blocks):
        yield from _block_frames(index, kind, parts, complete=complete, signed=signed)


def _mixed_stream(
    blocks: tuple[tuple[str, tuple[str, ...]], ...], *, complete: bool = True, signed: bool = True
) -> tuple[bytes, ...]:
    start: Final = _frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": MESSAGE_ID,
                "type": "message",
                "role": "assistant",
                "model": PRICING_MODEL,
                "content": [],
                "usage": {
                    "input_tokens": 13,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 17,
                    "cache_creation_input_tokens": 23,
                    "cache_creation": {"ephemeral_5m_input_tokens": 11, "ephemeral_1h_input_tokens": 12},
                    "inference_geo": "us",
                },
            },
        },
    )
    end: Final = (
        _frame(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 31}},
        ),
        _frame("message_stop", {"type": "message_stop"}),
    )
    return (start,) + tuple(_all_block_frames(blocks, complete=complete, signed=signed)) + (end if complete else ())


def _stream_response(
    chunks: tuple[str | bytes, ...], *, legacy: bool = False, speed: str | None = None
) -> ModelResponse:
    builder: Final = build_complete_streaming_response_legacy if legacy else build_complete_streaming_response
    response: Final = builder(
        chunks,
        None,  # pyright: ignore[reportArgumentType]  # synthetic frames do not need a logging object
        PRICING_MODEL,
        speed,
    )
    assert isinstance(response, ModelResponse)
    return response


def _response_payload(response: ModelResponse) -> tuple[object, object]:
    return response.model_dump(exclude={"id", "created"}), copy.deepcopy(response._hidden_params)


@pytest.mark.parametrize(
    "blocks",
    [
        (("thinking", ("hel", "lo", "", "世界")),),
        (("thinking", ("", "", "")),),
        (("thinking", ("a", "b")), ("thinking", ("c", "d"))),
        (("tool_use", ('{"x":"', "世界", '"}')),),
        (("tool_use", ("", "", "")),),
        (("tool_use", ('{"x":', "1}")), ("tool_use", ('{"y":', "2}"))),
        (
            ("thinking", ("a", "b")),
            ("text", ("hello", " world")),
            ("thinking", ("c", "d")),
            ("tool_use", ('{"x":', "1}")),
        ),
    ],
)
@pytest.mark.parametrize("transport", ["bytes", "string", "combined"])
def test_mixed_delta_compaction_preserves_complete_response_and_hidden_usage(
    blocks: tuple[tuple[str, tuple[str, ...]], ...], transport: str
) -> None:
    raw: Final = _mixed_stream(blocks)
    chunks: Final = (
        (b"".join(raw),)
        if transport == "combined"
        else tuple(frame.decode() for frame in raw)
        if transport == "string"
        else raw
    )
    original: Final = tuple(chunks)
    optimized: Final = _stream_response(chunks)
    legacy: Final = _stream_response(chunks, legacy=True)
    assert _response_payload(optimized) == _response_payload(legacy)
    assert chunks == original


@pytest.mark.parametrize("kind", ["thinking", "tool_use"])
@pytest.mark.parametrize("speed", [None, "fast"])
def test_mixed_compaction_preserves_cache_duration_and_speed_pricing(kind: str, speed: str | None) -> None:
    import litellm

    chunks: Final = _mixed_stream(((kind, ("a", "b")),))
    optimized: Final = _stream_response(chunks, speed=speed)
    legacy: Final = _stream_response(chunks, legacy=True, speed=speed)
    assert optimized.usage.model_dump() == legacy.usage.model_dump()
    native_cost: Final = litellm.completion_cost(
        completion_response=legacy, model=PRICING_MODEL, custom_llm_provider="anthropic"
    )
    optimized_cost: Final = litellm.completion_cost(
        completion_response=optimized, model=PRICING_MODEL, custom_llm_provider="anthropic"
    )
    assert native_cost is not None and native_cost > 0
    assert optimized_cost == native_cost


@pytest.mark.parametrize("kind", ["thinking", "tool_use"])
def test_partial_mixed_stream_keeps_original_reconstruction(kind: str) -> None:
    chunks: Final = _mixed_stream(((kind, ("a", "b")),), complete=False, signed=False)
    assert compact_content_block_deltas(chunks) is None
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))


def test_unsigned_thinking_keeps_original_reconstruction() -> None:
    chunks: Final = _mixed_stream((("thinking", ("a", "b")),), signed=False)
    assert compact_content_block_deltas(chunks) is None
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))


@pytest.mark.parametrize(
    "extra",
    [
        b": keepalive\n\n",
        b"data: [DONE]\n\n",
        _frame("ping", {"type": "ping", "unexpected": True}),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": {}}},
        ),
    ],
)
def test_unfamiliar_frames_preserve_response_or_native_fallback(extra: bytes) -> None:
    original: Final = _mixed_stream((("thinking", ("a", "b")),))
    chunks: Final = original[:3] + (extra,) + original[3:]
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))


@pytest.mark.parametrize("kind", ["thinking", "tool_use"])
def test_long_mixed_runs_create_only_one_response_chunk_for_the_delta_run(kind: str) -> None:
    parts: Final = tuple("x" for _ in range(128))
    chunks: Final = _mixed_stream(((kind, parts),))
    with patch(
        "litellm.llms.anthropic.chat.handler.ModelResponseStream", wraps=ModelResponseStream
    ) as native_constructor:
        legacy: Final = _stream_response(chunks, legacy=True)
        native_count: Final = native_constructor.call_count
    with patch(
        "litellm.llms.anthropic.chat.handler.ModelResponseStream", wraps=ModelResponseStream
    ) as compact_constructor:
        optimized: Final = _stream_response(chunks)
        compact_count: Final = compact_constructor.call_count
    assert _response_payload(optimized) == _response_payload(legacy)
    assert native_count - compact_count == len(parts) - 1


def test_pure_text_preserves_existing_response_and_construction_count() -> None:
    chunks: Final = _mixed_stream((("text", ("hello", " world")),))
    collapsed: Final = collapse_pure_text_chunks(chunks)
    assert collapsed is not None
    with patch(
        "litellm.llms.anthropic.chat.handler.ModelResponseStream", wraps=ModelResponseStream
    ) as native_constructor:
        legacy: Final = _stream_response(tuple(collapsed), legacy=True)
        native_count: Final = native_constructor.call_count
    with patch(
        "litellm.llms.anthropic.chat.handler.ModelResponseStream", wraps=ModelResponseStream
    ) as compact_constructor:
        optimized: Final = _stream_response(chunks)
        compact_count: Final = compact_constructor.call_count
    assert _response_payload(optimized) == _response_payload(legacy)
    assert compact_count == native_count


@pytest.mark.parametrize("position", [0, 1, 3, 99])
def test_invalid_message_lifecycle_uses_original_frames(position: int) -> None:
    original: Final = _mixed_stream((("thinking", ("a", "b")),))
    extra: Final = original[0] if position in (0, 1) else original[-2]
    chunks: Final = original[:position] + (extra,) + original[position:]
    assert compact_content_block_deltas(chunks) is None
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))


def test_ping_splits_delta_runs_and_preserves_complete_response() -> None:
    original: Final = _mixed_stream((("thinking", ("a", "b", "c", "d")),))
    ping: Final = _frame("ping", {"type": "ping"})
    chunks: Final = original[:4] + (ping,) + original[4:]
    compacted: Final = compact_content_block_deltas(chunks)
    assert compacted is not None
    assert any('"type": "ping"' in event for event in compacted)
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))


@pytest.mark.parametrize("output_tokens", [None, 0, 97])
def test_multiple_message_usage_events_preserve_native_accounting(output_tokens: int | None) -> None:
    original: Final = _mixed_stream((("tool_use", ('{"x":', "1}")),))
    usage: Final = {} if output_tokens is None else {"output_tokens": output_tokens}
    extra: Final = _frame("message_delta", {"type": "message_delta", "delta": {}, "usage": usage})
    chunks: Final = original[:-1] + (extra,) + original[-1:]
    assert compact_content_block_deltas(chunks) is not None
    assert _response_payload(_stream_response(chunks)) == _response_payload(_stream_response(chunks, legacy=True))
