"""Parameter gates for the cohere v2 chat serializer.

v1's gate is ``_check_valid_arg`` over ``CohereV2ChatConfig.
get_supported_openai_params`` (stream/temperature/max_tokens/mct/top_p/
frequency_penalty/presence_penalty/stop/n/tools/tool_choice/seed/
extra_headers): an unsupported OPENAI param RAISES UnsupportedParamsError
unless drop_params. v2 mirrors the supported-list truth as typed fallbacks
so v1 serves its own raise (the grok R2 rule).

Verified in-process at HEAD (dossier drift): ``top_k`` does NOT raise — it
is a non-openai param the generic passthrough places top-level in
optional_params, so v1 SERVES it on the wire verbatim (the serializer emits
it; differential row pins the wire merge). ``user`` is silently dropped
upstream (never raised) — typed fallback, v1 serves its own drop.
``tool_choice`` is IN the supported list but has NO map arm: v1 silently
drops it (serializer delta, not a fallback).
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    # deps is unused here — the uniform own-module gate signature
    # (critic-wave2b-beta N5's recorded convention), which lets the
    # serializer compose openai_compat.make_gated_serializer directly
    if request.response_format.is_some():
        return (
            "response_format is outside cohere v2's supported list; v1's "
            "get_optional_params raises UnsupportedParamsError (or drops it "
            "under drop_params)"
        )
    if request.parallel_tool_calls.is_some():
        return (
            "parallel_tool_calls is outside cohere v2's supported list; "
            "v1 raises or drops it"
        )
    if request.thinking.is_some():
        return "thinking is not a cohere v2 chat param; v1 raises or drops it"
    if request.reasoning_effort.is_some():
        return "reasoning_effort is not a cohere v2 chat param; v1 raises or drops it"
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "cohere v2; v1 serves its own drop"
        )
    return None
