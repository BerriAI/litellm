"""Differential parity for hosted_vllm streaming, pinned at the SSE line seam.

v1 side: raw ``data:`` lines through the BASE
``OpenAIChatCompletionStreamingHandler`` (HostedVLLMChatConfig has no
iterator override — canary-pinned below) into
``CustomStreamWrapper("hosted_vllm")``. v2 side: ``fold_lines`` with the
hosted_vllm parser (the httpx_chunk FAMILY policy: ``reasoning="rename"``)
and the shared ``xai`` chunk dialect. vLLM reasoning-parser streams carry
NATIVE ``reasoning_content`` deltas — pinned verbatim alongside the base
handler's ``reasoning`` pop-rename. The mid-stream error chunk is the
family's single PINNED DIVERGENCE (v1 swallows, v2 is loud) — hosted_vllm
joins the report's one named row.
"""

import copy
import json

import pytest

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.hosted_vllm import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

from ._own_module_corpus import provider_config, replay_v1_sse_lines

MODEL = "qwen2.5-7b-instruct"
WIRE_MODEL = "qwen2.5-7b-instruct"


def _chunk(delta, finish=None, usage=None, choices=None):
    payload = {
        "id": "vllm-chunk-1",
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
    "native_reasoning_content": [
        _chunk({"role": "assistant", "reasoning_content": "step "}),
        _chunk({"reasoning_content": "by step"}),
        _chunk({"content": "answer"}),
        _chunk({}, finish="stop"),
    ],
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
        usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    ),
]


def _v1_chunks(events, stream_options=None):
    return replay_v1_sse_lines("hosted_vllm", events, MODEL, stream_options)


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
        "id": "vllm-chunk-1",
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
    to hosted_vllm (same BASE handler): v1 silently swallows the error
    chunk, v2's parse_line is a loud typed boundary error. If the v1 half
    fails, the base handler learned to raise: re-decide the divergence and
    update the report row in the same commit."""
    v1 = _v1_chunks(_ERROR_CHUNK_STREAM)
    assert "error" not in json.dumps(v1), v1
    assert [c["choices"][0]["delta"].get("content") for c in v1[:2]] == ["", "after"]
    lines = [f"data: {json.dumps(e)}" for e in copy.deepcopy(_ERROR_CHUNK_STREAM)]
    folded = fold_lines(lines, parse_line, initial_state(MODEL, dialect="xai"))
    assert folded.is_error()
    assert "provider stream error" in folded.error.summary, folded.error.summary
    assert "upstream exploded" in folded.error.summary


def test_iterator_is_the_base_openai_handler() -> None:
    handler = provider_config("hosted_vllm", MODEL).get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
    assert type(handler) is OpenAIChatCompletionStreamingHandler
