"""openrouter SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1 decodes through its OWN ``OpenRouterChatCompletionStreamingHandler``
(NOT the base handler), so openrouter gets a per-provider policy over the
ONE shared httpx chunk normalizer:

- ``reasoning="unconditional"`` (the ReasoningMode arm added WITH this
  consumer): v1 assigns ``delta["reasoning_content"] =
  delta.get("reasoning")`` on EVERY delta — ``None`` when absent, the
  original ``reasoning`` key kept beside it, a native wire
  ``reasoning_content`` CLOBBERED.
- key-PRESENCE error chunks (``"error" in chunk`` -> OpenRouterException;
  the policy row mirrors that RAISE, like cometapi — openrouter is NOT part
  of the base-handler PINNED DIVERGENCE).
- the strict id/created/model/choices envelope (v1 subscripts them;
  KeyError -> OpenRouterException 400).

One pre-step the policy cannot express: v1 ALSO subscripts
``choice["delta"]`` inside the choices loop, so a choice MISSING ``delta``
raises in v1 while the shared normalizer would default it to ``{}`` and
serve — ``parse_event`` errors loudly on that shape first (v2 never serves
what v1 raises on; two-sided-pinned in the stream gate).

Folds use the ``"xai"`` ChunkDialect (the generic httpx dict path).
"""

from __future__ import annotations

from collections.abc import Sequence

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import PlainJson, StreamEvent
from ..openai_compat.httpx_chunk import (
    HttpxChunkPolicy,
    StrictEnvelope,
    make_parse_event,
)
from ..openai_compat.stream import make_parse_line

_EventResult = Result[StreamEvent | None, TranslationError]


def _missing_keys_reason(missing: Sequence[str]) -> str:
    return (
        f"stream chunk missing {list(missing)!r}; v1's chunk_parser raises "
        "KeyError (OpenRouterException 400) on it"
    )


_policy_parse_event = make_parse_event(
    HttpxChunkPolicy(
        reasoning="unconditional",
        error_on_key_presence=True,
        strict_envelope=StrictEnvelope(
            keys=("id", "created", "model", "choices"),
            reason=_missing_keys_reason,
        ),
    )
)


def parse_event(event: PlainJson) -> _EventResult:
    delta_error = _choice_missing_delta(event)
    if delta_error is not None:
        return Error(delta_error)
    return _policy_parse_event(event)


def _choice_missing_delta(event: PlainJson) -> TranslationError | None:
    if not isinstance(event, dict):
        return None  # the policy parser owns the non-object error
    choices = event.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if isinstance(choice, dict) and "delta" not in choice:
            return TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(
                        [
                            "stream choice missing 'delta'; v1's chunk_parser "
                            "subscripts it (KeyError -> OpenRouterException 400)"
                        ]
                    )
                )
            )
    return None


parse_line = make_parse_line(parse_event)
