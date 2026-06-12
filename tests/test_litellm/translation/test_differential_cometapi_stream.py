"""Differential parity for cometapi streaming, pinned at the SSE line seam.

v1 side: raw ``data:`` lines through ``CometAPIChatCompletionStreamingHandler``
(the chunk_parser owns the cometapi behavior: KeyError raises on missing
envelope keys, error-chunk raises, the CONDITIONAL ``delta.reasoning`` ->
``reasoning_content`` copy that keeps BOTH keys, the id/created/usage/model/
choices-only rebuild) into ``CustomStreamWrapper("cometapi")`` — the same
line seam the xai gate uses, because the rewrites sit below the parsed-chunk
seam. v2 side: ``fold_lines`` with the cometapi parser and the shared
``xai`` chunk dialect (the httpx-wrapper dialect: reasoning-only deltas are
non-empty, the model is never re-read from chunks).

The usage tail is the pinned seam contract inherited from the openai/xai
ports: v1's wrapper swallows the ``choices: []`` tail and SYNTHESIZES a
final usage chunk under include_usage (dropping it entirely without); the
v2 fold passes the wire tail through verbatim and the future streaming seam
owns the synthesis.
"""

import copy
import json
import time

import pytest

from litellm.exceptions import BadRequestError, MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.cometapi.chat.transformation import (
    CometAPIChatCompletionStreamingHandler,
)

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.compat_httpx.stream import (
    cometapi_parse_event as parse_event,
)
from litellm.translation.providers.compat_httpx.stream import (
    cometapi_parse_line as parse_line,
)
from litellm.translation_seam import to_model_response_stream

MODEL = "gpt-4o-mini"
WIRE_MODEL = "gpt-4o-mini-2024-07-18"


def _chunk(
    delta: dict[str, object] | None = None,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
    choices: list[dict[str, object]] | None = None,
    **extra: object,
) -> dict[str, object]:
    payload = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish,
            }
        ],
        **extra,
    }
    if usage is not None:
        payload["usage"] = usage
    if choices is not None:
        payload["choices"] = choices
    return payload


_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": "He"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
    ],
    "reasoning_rename": [
        # the conditional copy: delta.reasoning -> reasoning_content with the
        # original key KEPT (v1 assigns without popping, replay-verified)
        _chunk({"role": "assistant", "reasoning": "thinking..."}),
        _chunk({"content": "Hi"}),
        _chunk({}, finish="stop"),
    ],
    "native_reasoning_content": [
        _chunk({"role": "assistant", "reasoning_content": "native"}),
        _chunk({"content": "Hi"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk(
            {
                # no "type" key: the dict-path default fills "function"
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '{"city":"Paris"}'}}
                ]
            }
        ),
        _chunk({}, finish="tool_calls"),
    ],
    "extras_dropped": [
        # system_fingerprint and top-level extras are DROPPED by the v1
        # rebuild (id/created/usage/model/choices only) — opposite of the
        # SDK family's verbatim extras passthrough
        _chunk(
            {"role": "assistant", "content": "Hi"},
            system_fingerprint="fp_x",
            citations=["https://a"],
        ),
        _chunk({}, finish="stop"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(choices=[], usage=_USAGE),
]


def _v1_chunks(
    events: list[dict[str, object]],
    stream_options: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = CometAPIChatCompletionStreamingHandler(
        streaming_response=iter(lines), sync_stream=True
    )
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-cometapi-stream",
        function_id="diff-cometapi-stream",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=MODEL,
        custom_llm_provider="cometapi",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _v2_chunks(events: list[dict[str, object]]) -> list[dict[str, object]]:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list[dict[str, object]]) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_reasoning_delta_keeps_both_keys(frozen_ambient) -> None:
    """Semantic pin on top of the byte row: the emitted delta carries BOTH
    ``reasoning`` and ``reasoning_content`` (v1's chunk_parser assigns
    without popping — the groq/openrouter variants differ here)."""
    v2 = _v2_chunks(STREAMS["reasoning_rename"])
    delta = v2[0]["choices"][0]["delta"]
    assert delta["reasoning"] == "thinking..."
    assert delta["reasoning_content"] == "thinking..."


def test_usage_tail_pins_the_seam_contract(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, stream_options={"include_usage": True})
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2)
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    assert v2[-1]["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1[-1]["usage"][key] == v2[-1]["usage"][key], key


def test_usage_tail_swallowed_without_stream_options(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, None)
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2) - 1
    assert _norm(v1) == _norm(v2[:-1])


def test_mid_stream_usage_withheld_like_v1(frozen_ambient) -> None:
    """Usage on a content-bearing chunk never reaches the emitted chunk in
    v1 (the wrapper strips it); the v2 parser attaches usage only to the
    ``choices: []`` tail, byte-identical both sides."""
    events = [
        _chunk({"role": "assistant", "content": "Hi"}, usage=_USAGE),
        _chunk({}, finish="stop"),
    ]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert _norm(v2) == _norm(v1)
    assert "usage" not in v2[0] or v2[0]["usage"] is None


def test_v2_line_and_event_folds_agree(frozen_ambient) -> None:
    events = STREAMS["text"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="xai")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_events) == _norm(_v2_chunks(events))


_LOUD_CHUNKS = {
    "error_chunk": (
        {"error": {"message": "boom", "code": 500}},
        "provider stream error",
    ),
    "missing_choices": (
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "model": WIRE_MODEL,
        },
        "KeyError",
    ),
    "missing_id": (
        {
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "missing_created": (
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "missing_model": (
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1718000000,
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}],
        },
        "KeyError",
    ),
    "multiple_choices": (
        _chunk(
            choices=[
                {"index": 0, "delta": {"content": "a"}, "finish_reason": None},
                {"index": 1, "delta": {"content": "b"}, "finish_reason": None},
            ]
        ),
        "multiple stream choices",
    ),
    "unknown_delta_key": (
        _chunk({"content": "x", "thinking_blocks": []}),
        "stream delta keys",
    ),
}


@pytest.mark.parametrize("name", sorted(_LOUD_CHUNKS))
def test_loud_chunk_shapes_error_on_both_sides(name: str, frozen_ambient) -> None:
    """v1 RAISES out of the iterator on error chunks and on chunks missing
    any required envelope key (KeyError -> CometAPIException), with the
    exact mapped exception types pinned (critic-wave2a N4); v2 must be a
    loud error value, never a served chunk. The two unreachable shapes
    (n>1, unknown delta keys) are v2-side typed errors like every family
    parser."""
    event, reason_fragment = _LOUD_CHUNKS[name]
    result = parse_event(copy.deepcopy(event))
    assert result.is_error(), f"{name} unexpectedly parsed"
    assert reason_fragment in result.error.summary, result.error.summary
    if name == "error_chunk":
        # the wrapper maps the in-stream CometAPIException onto the
        # mid-stream fallback contract
        with pytest.raises(MidStreamFallbackError):
            _v1_chunks([event])
    elif name.startswith("missing_"):
        # chunk_parser's KeyError -> CometAPIException(400) -> BadRequestError
        with pytest.raises(BadRequestError):
            _v1_chunks([event])
