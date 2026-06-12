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
  live-search billing hook). v1 swallows non-numeric values (the whole
  post-step sits in a try/except-debug), so non-numerics skip silently.
- ``_fold_reasoning_tokens_into_completion``: fold reasoning into
  completion_tokens when ``total == prompt + completion + reasoning``.
- ``_normalize_openai_compatible_usage_totals``: bump total_tokens up to
  ``prompt + completion``.

Token coercion mirrors v1's ``int(x or 0)`` exactly: numeric strings fold
(``int("7") == 7``, matching the pydantic-lax ``Usage(**raw)`` on v1's
response path), bools coerce (``int(True or 0) == 1``), and an uncoercible
value is a typed boundary error — LOUD where v1 raises (``int("abc")``
inside the chunk_parser / the ``Usage`` validation inside cdr), never a
silent zero.

``citations`` and any future live-search top-level keys ride the parser's
unknown-key mirror exactly like v1's cdr:727-729 setattr.
"""

from __future__ import annotations

import dataclasses

from expression import Error, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, JsonBlob, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response
from ..openai_compat.response import semantic_usage as openai_semantic_usage

_ParseResult = Result[ChatResponse, TranslationError]
_Usage = dict[str, PlainJson]


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return openai_parse_response(raw, request).bind(_with_xai_usage_post_steps)


def _with_xai_usage_post_steps(response: ChatResponse) -> _ParseResult:
    wire = response.wire.default_value(None)
    if wire is None or not isinstance(wire.value, dict):
        return Ok(response)
    usage = wire.value.get("usage")
    if not isinstance(usage, dict):
        return Ok(response)
    folded = fold_reasoning_tokens(_websearch_fields(usage))
    if isinstance(folded, TranslationError):
        return Error(folded)
    transformed = normalize_usage_totals(folded)
    if isinstance(transformed, TranslationError):
        return Error(transformed)
    if transformed == usage:
        return Ok(response)
    body: dict[str, PlainJson] = {**wire.value, "usage": transformed}
    return Ok(
        dataclasses.replace(
            response,
            usage=openai_semantic_usage(transformed),
            wire=Some(JsonBlob(value=body)),
        )
    )


def _websearch_fields(usage: _Usage) -> _Usage:
    sources = usage.get("num_sources_used")
    if not isinstance(sources, (bool, int, float)):
        return usage  # v1's `n > 0` raises on non-numerics and is swallowed
    if sources <= 0:
        return usage
    details = usage.get("prompt_tokens_details")
    seeded: dict[str, PlainJson] = dict(details) if isinstance(details, dict) else {}
    return {
        **usage,
        "num_sources_used": int(sources),
        "prompt_tokens_details": {**seeded, "web_search_requests": int(sources)},
    }


def fold_reasoning_tokens(usage: _Usage) -> _Usage | TranslationError:
    """Mirror ``_fold_reasoning_tokens_into_completion``'s dict variant in
    v1's read order: reasoning first (early return at <= 0), then
    prompt/completion/total."""
    details = usage.get("completion_tokens_details")
    reasoning = _coerce_int(
        details.get("reasoning_tokens") if isinstance(details, dict) else None,
        "completion_tokens_details.reasoning_tokens",
    )
    if isinstance(reasoning, TranslationError):
        return reasoning
    if reasoning <= 0:
        return usage
    tokens = _coerced_token_triple(usage)
    if isinstance(tokens, TranslationError):
        return tokens
    prompt_tokens, completion_tokens, total_tokens = tokens
    if total_tokens == prompt_tokens + completion_tokens:
        return usage
    if total_tokens != prompt_tokens + completion_tokens + reasoning:
        return usage  # v1's double-count guard
    return {**usage, "completion_tokens": completion_tokens + reasoning}


def normalize_usage_totals(usage: _Usage) -> _Usage | TranslationError:
    """Mirror ``_normalize_openai_compatible_usage_totals``'s dict variant."""
    tokens = _coerced_token_triple(usage)
    if isinstance(tokens, TranslationError):
        return tokens
    prompt_tokens, completion_tokens, total_tokens = tokens
    expected = prompt_tokens + completion_tokens
    if total_tokens >= expected:
        return usage
    return {**usage, "total_tokens": expected}


def _coerced_token_triple(
    usage: _Usage,
) -> tuple[int, int, int] | TranslationError:
    prompt = _coerce_int(usage.get("prompt_tokens"), "prompt_tokens")
    if isinstance(prompt, TranslationError):
        return prompt
    completion = _coerce_int(usage.get("completion_tokens"), "completion_tokens")
    if isinstance(completion, TranslationError):
        return completion
    total = _coerce_int(usage.get("total_tokens"), "total_tokens")
    if isinstance(total, TranslationError):
        return total
    return prompt, completion, total


def _coerce_int(value: PlainJson, field: str) -> int | TranslationError:
    """v1's ``int(x or 0)``: None/""/0 -> 0, bools and numeric strings
    coerce, anything ``int()`` rejects is loud (v1 raises out of the
    post-step or the Usage validation; v2 returns the boundary error)."""
    if value is None:
        return 0
    if isinstance(value, (bool, int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value) if value else 0
        except ValueError:
            pass
    return TranslationError.of_boundary(
        BoundaryError.of(
            Block.of_seq([f"usage {field} is not int-coercible: {value!r} (v1 raises)"])
        )
    )
