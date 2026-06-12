"""xai (Grok) chat-completion response JSON -> IR ``ChatResponse``.

The live v1 normalizer is ``XAIChatConfig.transform_response`` (xai is on
the dedicated httpx branch, main.py:2289, so transform_response RUNS —
the inverse of the openai SDK path): the shared
``convert_to_model_response_object`` conversion, then the xai post-steps.
v2 mirrors that chain over the openai_compat parser's normalized wire body:

- finish_reason ``""`` (Grok's tool-call quirk) needs NO xai arm: v1's own
  ``_fix_choice_finish_reason_for_tool_calls`` is empirically dead
  (``Choices.__init__`` maps ``""`` -> ``"stop"`` before the check ever
  runs), so v1-as-executed emits ``"stop"`` WITH tool_calls. The openai
  parser already rides the native ``""`` on the wire body and the seam's
  ``Choices`` runs the same live ``map_finish_reason`` — identical chain.
- ``_enhance_usage_with_xai_web_search_fields``: ``usage.num_sources_used``
  > 0 copies into ``prompt_tokens_details.web_search_requests`` (the
  live-search billing hook).
- ``_fold_reasoning_tokens_into_completion``: fold reasoning into
  completion_tokens when ``total == prompt + completion + reasoning``.
- ``_normalize_openai_compatible_usage_totals``: bump total_tokens up to
  ``prompt + completion``.

``citations`` and any future live-search top-level keys ride the parser's
unknown-key mirror exactly like v1's cdr:727-729 setattr.
"""

from __future__ import annotations

import dataclasses

from expression import Result, Some

from ...errors import TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response
from ..openai_compat.response import semantic_usage as openai_semantic_usage

_ParseResult = Result[ChatResponse, TranslationError]


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return openai_parse_response(raw, request).map(_with_xai_usage_post_steps)


def _with_xai_usage_post_steps(response: ChatResponse) -> ChatResponse:
    wire = response.wire.default_value(None)
    if wire is None or not isinstance(wire.value, dict):
        return response
    usage = wire.value.get("usage")
    if not isinstance(usage, dict):
        return response
    transformed = normalize_usage_totals(
        fold_reasoning_tokens(_websearch_fields(usage))
    )
    if transformed == usage:
        return response
    body: dict[str, PlainJson] = {**wire.value, "usage": transformed}
    return dataclasses.replace(
        response,
        usage=openai_semantic_usage(transformed),
        wire=Some(JsonBlob(value=body)),
    )


def _websearch_fields(usage: dict[str, PlainJson]) -> dict[str, PlainJson]:
    sources = usage.get("num_sources_used")
    if not isinstance(sources, (int, float)) or isinstance(sources, bool):
        return usage
    if sources <= 0:
        return usage
    details = usage.get("prompt_tokens_details")
    seeded: dict[str, PlainJson] = dict(details) if isinstance(details, dict) else {}
    return {
        **usage,
        "num_sources_used": int(sources),
        "prompt_tokens_details": {**seeded, "web_search_requests": int(sources)},
    }


def fold_reasoning_tokens(usage: dict[str, PlainJson]) -> dict[str, PlainJson]:
    """Pure dict mirror of ``_fold_reasoning_tokens_into_completion`` (the
    same arithmetic the stream chunk variant runs)."""
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else 0
    reasoning_tokens = _int_of(reasoning)
    if reasoning_tokens <= 0:
        return usage
    prompt_tokens = _int_of(usage.get("prompt_tokens"))
    completion_tokens = _int_of(usage.get("completion_tokens"))
    total_tokens = _int_of(usage.get("total_tokens"))
    if total_tokens == prompt_tokens + completion_tokens:
        return usage
    if total_tokens != prompt_tokens + completion_tokens + reasoning_tokens:
        return usage  # v1's double-count guard
    return {**usage, "completion_tokens": completion_tokens + reasoning_tokens}


def normalize_usage_totals(usage: dict[str, PlainJson]) -> dict[str, PlainJson]:
    """Pure dict mirror of ``_normalize_openai_compatible_usage_totals``."""
    expected = _int_of(usage.get("prompt_tokens")) + _int_of(
        usage.get("completion_tokens")
    )
    if _int_of(usage.get("total_tokens")) >= expected:
        return usage
    return {**usage, "total_tokens": expected}


def _int_of(value: PlainJson) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0
