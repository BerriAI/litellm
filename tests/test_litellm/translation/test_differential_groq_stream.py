"""Differential parity for groq streaming (wave-2b-beta), pinned at the
SSE line seam.

v1 side: ``data:``/[DONE] lines through ``GroqChatCompletionStreamingHandler``
(truthy-``error`` raise, the ``delta.reasoning`` POP rename, then the base
rebuild) into ``CustomStreamWrapper("groq")``. v2 side: the shared
httpx_chunk factory with ``reasoning="rename"`` — the longtail guidance's
prediction verified (groq's pop == the existing rename mode; NO new
ReasoningMode arm) — folded by the shared ``xai`` chunk dialect.
"""

import copy
import json
import time

import pytest

from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.groq.chat.transformation import GroqChatCompletionStreamingHandler

from litellm.translation.engine.stream import fold_events, fold_lines
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.groq import parse_event, parse_line
from litellm.translation_seam import to_model_response_stream

MODEL = "llama-3.3-70b-versatile"

_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _chunk(
    delta: dict | None = None,
    finish: str | None = None,
    usage: dict | None = None,
    choices: list | None = None,
    **extra: object,
) -> dict:
    payload: dict = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": MODEL,
        **extra,
    }
    payload["choices"] = (
        choices
        if choices is not None
        else [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish,
            }
        ]
    )
    if usage is not None:
        payload["usage"] = usage
    return payload


STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": "He"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
    ],
    "reasoning_pop_rename": [
        _chunk({"role": "assistant", "reasoning": "think"}),
        _chunk({"content": "x"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ],
    "x_groq_extras_dropped": [
        _chunk({"role": "assistant", "content": "Hi"}, x_groq={"id": "r"}),
        _chunk({}, finish="stop"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(choices=[], usage=_USAGE),
]


def _v1_chunks(events: list, stream_options: dict | None = None) -> list:
    lines = [f"data: {json.dumps(event)}" for event in copy.deepcopy(events)]
    lines.append("data: [DONE]")
    handler = GroqChatCompletionStreamingHandler(
        streaming_response=iter(lines), sync_stream=True
    )
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-groq-stream",
        function_id="diff-groq-stream",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=handler,
        model=MODEL,
        custom_llm_provider="groq",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


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


def test_reasoning_pop_emits_only_reasoning_content(frozen_ambient) -> None:
    """Semantic pin: groq POPS (the original ``reasoning`` key never reaches
    the delta — the cometapi copy_both variant differs here)."""
    v2 = _v2_chunks(STREAMS["reasoning_pop_rename"])
    delta = v2[0]["choices"][0]["delta"]
    assert delta["reasoning_content"] == "think"
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
    events = STREAMS["reasoning_pop_rename"]
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="xai")
    )
    assert folded.is_ok(), folded.error.summary
    via_events = [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]
    assert _norm(via_events) == _norm(_v2_chunks(events))


def test_error_chunk_loud_on_both_sides(frozen_ambient) -> None:
    """v1 raises OpenAIError -> the wrapper maps it onto
    MidStreamFallbackError (the truthy-VALUE check, the xai shape); v2 is a
    loud typed error."""
    event = {"error": {"message": "boom", "code": 500}}
    result = parse_event(copy.deepcopy(event))
    assert result.is_error()
    assert "provider stream error" in result.error.summary
    with pytest.raises(MidStreamFallbackError):
        _v1_chunks([event])


def test_falsy_error_value_is_served_like_v1(frozen_ambient) -> None:
    """The value-check pin: ``error: null`` does NOT raise in v1 (groq
    checks truthiness, not key presence — cometapi differs)."""
    events = [
        _chunk({"role": "assistant", "content": "Hi"}, error=None),
        _chunk({}, finish="stop"),
    ]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_non_str_reasoning_is_loud_where_v1_raises(key: str, frozen_ambient) -> None:
    """verifier-wave2b-beta F6: a non-str truthy reasoning value reaches
    v1's wrapper epilogue, whose stream_chunk_builder ``"".join`` raises
    TypeError -> APIError; the factory's _string_or_none quietly nulled it.
    Null reasoning stays served (the reasoning_null parity row holds)."""
    events = [
        _chunk({"role": "assistant", **{key: 5}}),
        _chunk({}, finish="stop"),
    ]
    result = parse_event(copy.deepcopy(events[0]))
    assert result.is_error()
    assert "is not a string" in result.error.summary
    with pytest.raises(Exception):
        _v1_chunks(events)


def test_null_reasoning_still_serves(frozen_ambient) -> None:
    """The F6 pre-step must not widen: null reasoning serves identically
    (the verifier's reasoning-null parity probe)."""
    events = [
        _chunk({"role": "assistant", "content": "", "reasoning": None}),
        _chunk({"content": "hi"}),
        _chunk({}, finish="stop"),
    ]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_non_str_refusal_nulling_handoff_to_the_factory_fix(frozen_ambient) -> None:
    """INTEGRATOR-FLIP (wave-2b-beta F6 handoff — sibling merge): a non-str
    ``refusal`` is nulled by the SHARED factory's _string_or_none
    (openai_compat/httpx_chunk.py) while v1 FORWARDS it on the wire — both
    serve, bodies differ. That file is the alpha fix round's concurrent
    edit (verifier-wave2b-alpha F1, same machinery), so this branch does
    NOT touch it; this row GUARDS THE CURRENT (divergent) behavior so the
    sibling-merge integrator flips the assertions to v1 parity (refusal
    forwarded verbatim) when the factory fix lands. See the merge notes in
    ~/sprint/reports/wave2b-beta-port.md (Fix round section)."""
    events = [
        _chunk({"role": "assistant", "refusal": 7, "content": "x"}),
        _chunk({}, finish="stop"),
    ]
    v1 = _v1_chunks(events)
    v2 = _v2_chunks(events)
    assert (
        v1[0]["choices"][0]["delta"]["refusal"] == 7
    ), "v1 stopped forwarding the non-str refusal; re-decide the handoff"
    # CURRENT factory behavior — the integrator flips this to == 7 with the
    # alpha httpx_chunk fix:
    assert v2[0]["choices"][0]["delta"]["refusal"] is None
