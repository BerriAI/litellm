"""groq SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1's decode is ``GroqChatCompletionStreamingHandler(
OpenAIChatCompletionStreamingHandler)`` over standard ``data:``/[DONE] SSE:
a TRUTHY ``error`` value raises OpenAIError (the xai-style VALUE check, not
cometapi's key-presence check), ``delta.reasoning`` is POPPED into
``reasoning_content``, then the BASE rebuild runs. That is EXACTLY the
shared httpx_chunk factory with ``reasoning="rename"`` — the longtail
guidance's prediction ("groq's pop semantics == the existing rename mode"),
verified against the v1 source: the pre-step pop followed by the base
handler's own rename is output-identical to one rename. NO new
ReasoningMode arm was needed (the no-consumer-no-arm rule holds).

verifier-wave2b-beta F6: a NON-STR truthy ``reasoning``/
``reasoning_content`` delta value is a groq-local LOUD pre-step — the
factory's ``_string_or_none`` quietly nulled it where v1 RAISES (the
wrapper's stream_chunk_builder epilogue ``"".join`` TypeErrors ->
APIError). Null reasoning stays served (verified identical). The
refusal-value half of F6 was DISCHARGED at the wave-2b sibling merge
(the named INTEGRATOR-FLIP handoff): the shared factory now forwards a
non-str refusal VERBATIM exactly like v1's Delta (re-probed two-sided);
a refusal riding the FINISH chunk is the loud finish-chunk fallback
where groq's v1 wrapper silently drops the value — both pinned in
test_differential_groq_stream.py.
"""

from __future__ import annotations

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import PlainJson, StreamEvent
from ..openai_compat.httpx_chunk import BASE_HANDLER_POLICY, make_parse_event
from ..openai_compat.stream import make_parse_line

_EventResult = Result[StreamEvent | None, TranslationError]

# groq's v1 handler is its OWN class, but its policy VALUE is exactly the
# family truth (rename + value-checked errors + no envelope keys) — ONE
# name, never a re-declared HttpxChunkPolicy(reasoning="rename") copy
# (sibling-merge consistency sweep; critic-wave2b-alpha NIT-1's rule).
_family_parse_event = make_parse_event(BASE_HANDLER_POLICY)


def parse_event(event: PlainJson) -> _EventResult:
    reason = _reasoning_reason(event)
    if reason is not None:
        return Error(
            TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))
        )
    return _family_parse_event(event)


def _reasoning_reason(event: PlainJson) -> str | None:
    if not isinstance(event, dict):
        return None  # the factory owns the non-object error
    choices = event.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("reasoning", "reasoning_content"):
            value = delta.get(key)
            if value is not None and not isinstance(value, str):
                return (
                    f"groq delta {key} is not a string (v1's wrapper "
                    'stream_chunk_builder epilogue "".join raises '
                    "TypeError -> APIError)"
                )
    return None


parse_line = make_parse_line(parse_event)
