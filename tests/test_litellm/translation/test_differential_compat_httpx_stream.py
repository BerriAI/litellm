"""Differential parity for compat_httpx (wave-1b) streams, at the line seam.

v1 side: raw ``data:`` lines through each config's
``get_model_response_iterator`` — the BASE
``OpenAIChatCompletionStreamingHandler`` for all nine providers (the
canaries below pin that: bedrock_mantle's override returns exactly that
class and ovhcloud's custom handler is dead code) — into
``CustomStreamWrapper(custom_llm_provider=p)``. v2 side: ``fold_lines``
with the family parser and the ``xai`` chunk dialect (the generic httpx
dict path: reasoning_content-aware delta emptiness, no extras passthrough).

The usage tail row pins the inherited seam contract: v1's wrapper swallows
the ``choices: []`` usage chunk and synthesizes the final usage chunk under
``include_usage``; the v2 fold passes the wire tail through verbatim and
the future streaming seam owns the synthesis.
"""

import copy
import json

import pytest

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)

from litellm.translation.engine.stream import fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.compat_httpx.stream import parse_line
from litellm.translation_seam import to_model_response_stream

from ._compat_httpx_corpus import (
    PROVIDERS,
    STREAM_MODEL,
    provider_config,
    replay_v1_sse_lines,
)


def _chunk(delta, finish=None, usage=None, choices=None):
    payload = {
        "id": "cmpl-h1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": "wire-stream-model",
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
    "reasoning_rename": [
        _chunk({"role": "assistant", "content": ""}),
        # delta.reasoning -> reasoning_content (the base handler's
        # unconditional pop-rename; gpt-oss / MiniMax reasoning deltas)
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


def _v2_chunks(events: list) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    folded = fold_lines(lines, parse_line, initial_state(STREAM_MODEL, dialect="xai"))
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


def _rows():
    return sorted((provider, name) for provider in PROVIDERS for name in STREAMS)


@pytest.mark.parametrize("provider,name", _rows())
def test_v2_stream_matches_v1(provider: str, name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(replay_v1_sse_lines(provider, events))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_usage_tail_seam_contract(provider: str, frozen_ambient) -> None:
    v1 = replay_v1_sse_lines(
        provider, USAGE_STREAM, stream_options={"include_usage": True}
    )
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2)
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    assert v2[-1]["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1[-1]["usage"][key] == v2[-1]["usage"][key]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_iterator_is_the_base_openai_handler(provider: str) -> None:
    """The family's one-dialect claim: every config streams through the
    base OpenAIChatCompletionStreamingHandler (bedrock_mantle's override
    returns exactly it; nobody else overrides)."""
    handler = provider_config(provider, "m").get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
    assert type(handler) is OpenAIChatCompletionStreamingHandler, provider


def test_ovhcloud_custom_handler_is_dead_code() -> None:
    """OVHCloudChatCompletionStreamingHandler is defined but never wired
    (no get_model_response_iterator override, no other reference) — if this
    canary fails, ovhcloud grew a live custom dialect: re-pin its stream
    rows against the new handler before trusting the family parser."""
    from litellm.llms.ovhcloud.chat.transformation import (
        OVHCloudChatCompletionStreamingHandler,
    )

    handler = provider_config("ovhcloud", "m").get_model_response_iterator(
        streaming_response=iter(()), sync_stream=True
    )
    assert not isinstance(handler, OVHCloudChatCompletionStreamingHandler)
