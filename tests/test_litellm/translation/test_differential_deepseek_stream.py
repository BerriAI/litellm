"""Differential parity for deepseek streaming, pinned at the SSE line seam.

v1 side: raw ``data:`` lines through the BASE
``OpenAIChatCompletionStreamingHandler`` (DeepSeekChatConfig has no iterator
override — canary-pinned below) into ``CustomStreamWrapper("deepseek")``.
v2 side: ``fold_lines`` with the deepseek parser (the httpx_chunk FAMILY
policy: ``reasoning="rename"``) and the shared ``xai`` chunk dialect (the
generic httpx dict path). deepseek-reasoner wire streams carry NATIVE
``reasoning_content`` deltas — pinned verbatim alongside the base handler's
``reasoning`` pop-rename.

The usage tail is the pinned seam contract inherited from the openai/xai
ports; the mid-stream ``{"error": ...}`` chunk is the SAME deliberate
fail-closed PINNED DIVERGENCE the compat_httpx family carries (v1's base
handler silently swallows it, v2 is loud) — the report's single PINNED
DIVERGENCE row names deepseek among the base-handler consumers.
"""

import copy
import json

import pytest

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.deepseek import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

from ._own_module_corpus import provider_config, replay_v1_sse_lines

MODEL = "deepseek-chat"
WIRE_MODEL = "deepseek-chat"


def _chunk(delta, finish=None, usage=None, choices=None):
    payload = {
        "id": "ds-chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        ],
        "usage": usage,
    }
    if choices is not None:
        payload["choices"] = choices
    return payload


STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "Hello"}),
        _chunk({"content": " world"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk({"role": "assistant", "content": None}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '{"city":"Paris"}'}}
                ]
            }
        ),
        _chunk({}, finish="tool_calls"),
    ],
    # deepseek-reasoner's real wire shape: native reasoning_content deltas
    # interleaved before content
    "native_reasoning_content": [
        _chunk({"role": "assistant", "reasoning_content": "step "}),
        _chunk({"reasoning_content": "by step"}),
        _chunk({"content": "answer"}),
        _chunk({}, finish="stop"),
    ],
    # the base handler's unconditional delta.reasoning pop-rename
    "reasoning_rename": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"reasoning": "thinking..."}),
        _chunk({"content": "answer"}),
        _chunk({}, finish="stop"),
    ],
    "empty_keepalive_swallowed": [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({}),
        _chunk({"content": "hi"}),
        _chunk({}, finish="stop"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(
        {},
        choices=[],
        usage={
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
            "prompt_cache_hit_tokens": 3,
            "prompt_cache_miss_tokens": 4,
        },
    ),
]


def _v1_chunks(events, stream_options=None):
    return replay_v1_sse_lines("deepseek", events, MODEL, stream_options)


def _v2_chunks(events: list) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_reasoning_rename_pops_the_wire_key(frozen_ambient) -> None:
    """Semantic pin: the emitted delta carries reasoning_content ONLY (the
    base handler POPS delta.reasoning — cometapi's copy_both differs)."""
    v2 = _v2_chunks(STREAMS["reasoning_rename"])
    delta = v2[1]["choices"][0]["delta"]
    assert delta["reasoning_content"] == "thinking..."
    assert "reasoning" not in delta


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


def test_v2_line_and_event_folds_agree(frozen_ambient) -> None:
    events = STREAMS["native_reasoning_content"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="xai")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_events) == _norm(_v2_chunks(events))


_ERROR_CHUNK_STREAM = [
    _chunk({"role": "assistant", "content": ""}),
    {
        "id": "ds-chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": WIRE_MODEL,
        "error": {"message": "upstream exploded", "code": 502},
    },
    _chunk({"content": "after"}),
    _chunk({}, finish="stop"),
]


def test_error_chunk_divergence_two_sided(frozen_ambient) -> None:
    """The compat_httpx family's deliberate fail-closed divergence extends
    to deepseek (same BASE handler): a mid-stream ``{"error": {...}}`` chunk
    is SILENTLY SWALLOWED by v1 (no error surface in the emitted sequence),
    while v2's parse_line surfaces a LOUD typed boundary error naming the
    chunk. Flag-on cannot lose data (v2 is louder, not quieter). If the v1
    half fails, the base handler learned to raise: re-decide the divergence
    and update the report row in the same commit."""
    v1 = _v1_chunks(_ERROR_CHUNK_STREAM)
    assert "error" not in json.dumps(v1), v1
    assert [c["choices"][0]["delta"].get("content") for c in v1[:2]] == ["", "after"]
    lines = [f"data: {json.dumps(e)}" for e in copy.deepcopy(_ERROR_CHUNK_STREAM)]
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
    assert folded.is_error()
    assert "provider stream error" in folded.error.summary, folded.error.summary
    assert "upstream exploded" in folded.error.summary


def test_iterator_is_the_base_openai_handler() -> None:
    """The one-dialect claim: DeepSeekChatConfig streams through the base
    OpenAIChatCompletionStreamingHandler (no override). If this canary
    fails, deepseek grew a custom dialect: re-pin every stream row against
    the new handler before trusting the family policy."""
    handler = provider_config("deepseek", MODEL).get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
    assert type(handler) is OpenAIChatCompletionStreamingHandler
