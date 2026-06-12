"""compat_httpx SSE ``chat.completion.chunk`` payloads -> IR stream events.

Both parsers here are the ONE shared httpx chunk normalizer
(``openai_compat.httpx_chunk.make_parse_event`` — critic-wave1b B1: the
factory replaced this module's verbatim copy of the xai normalizer at the
sibling merge; no new policy axis was needed) composed with two policies:

- the FAMILY policy (nine shim providers): every config streams through the
  BASE ``OpenAIChatCompletionStreamingHandler`` (bedrock_mantle's override
  returns exactly that class, and ovhcloud's custom handler is DEAD CODE —
  both canary-pinned) into ``CustomStreamWrapper``'s default openai branch.
  That is the xai policy minus the usage fold: ``reasoning="rename"`` (the
  base handler's unconditional pop-rename), value-checked error chunks, no
  required envelope keys, wire usage verbatim (only ever on the
  ``choices: []`` tail). Provider error chunks are a LOUD v2 boundary error
  where v1's base handler silently swallows them — the family's one
  deliberate fail-closed divergence, two-sided-pinned by
  ``test_error_chunk_divergence_two_sided`` and carried as a named report
  row.
- the COMETAPI policy: v1 decodes through its OWN
  ``CometAPIChatCompletionStreamingHandler`` (NOT the base handler), so
  cometapi gets a per-provider parser row: ``reasoning="copy_both"`` (v1
  assigns without popping — both keys reach the Delta), key-PRESENCE error
  chunks, and the strict id/created/model/choices envelope (v1 KeyErrors ->
  CometAPIException). Pinned by the dedicated line-seam replays in
  test_differential_cometapi_stream.py.

``LINE_PARSERS`` is the family's per-provider stream truth (the PARSERS
-table precedent): the future streaming seam must select from it, never
assume the family default covers every member. Folds use the ``"xai"``
ChunkDialect — the GENERIC httpx-dict-path wrapper truths
(reasoning_content-aware delta-emptiness, no extras passthrough), not
anything Grok-specific.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import get_args

from expression import Result

from ...errors import TranslationError
from ...ir import StreamEvent
from ..openai_compat.httpx_chunk import (
    BASE_HANDLER_POLICY,
    HttpxChunkPolicy,
    StrictEnvelope,
    make_parse_event,
)
from ..openai_compat.stream import make_parse_line
from .params import CompatHttpxProvider

ParseLine = Callable[[str], Result[StreamEvent | None, TranslationError]]

parse_event = make_parse_event(BASE_HANDLER_POLICY)
parse_line = make_parse_line(parse_event)


def _cometapi_missing_keys_reason(missing: Sequence[str]) -> str:
    return (
        f"stream chunk missing {list(missing)!r}; v1's chunk_parser raises "
        "KeyError (CometAPIException) on it"
    )


cometapi_parse_event = make_parse_event(
    HttpxChunkPolicy(
        reasoning="copy_both",
        error_on_key_presence=True,
        strict_envelope=StrictEnvelope(
            keys=("id", "created", "model", "choices"),
            reason=_cometapi_missing_keys_reason,
        ),
    )
)
cometapi_parse_line = make_parse_line(cometapi_parse_event)

LINE_PARSERS: Mapping[CompatHttpxProvider, ParseLine] = MappingProxyType(
    {
        provider: (cometapi_parse_line if provider == "cometapi" else parse_line)
        for provider in get_args(CompatHttpxProvider)
    }
)
