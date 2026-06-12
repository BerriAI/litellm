"""xai (Grok) SSE ``chat.completion.chunk`` payloads -> IR stream events.

The v1 decode is the httpx dict path: ``BaseModelResponseIterator`` strips
``data:`` lines, ``XAIChatCompletionStreamingHandler.chunk_parser`` applies
the xai rewrites, and the rebuilt ``ModelResponseStream`` flows through
``CustomStreamWrapper``'s default openai branch. The machinery is the shared
httpx chunk normalizer (``openai_compat.httpx_chunk.make_parse_event`` — ONE
mechanism for xai, cometapi, and the wave-2b dict-path providers;
critic-wave2a M2), composed with the xai policy. The xai deltas vs the
openai parser (all verified in-process at HEAD):

- the chunk_parser rebuild keeps ONLY id/created/model/choices/usage:
  ``system_fingerprint`` and every top-level extra (``citations``) are
  DROPPED — the opposite of the SDK path's verbatim extras passthrough
  (the factory's shared behavior).
- a ``choices: []`` chunk carrying ``usage`` gets a dummy choice injected in
  v1 purely so the wrapper machinery swallows it and re-synthesizes the
  final usage chunk; v2 keeps the openai passthrough shape (``choices: []``
  + the FOLDED usage) so the seam's synthesized-final-usage contract from
  the openai port applies unchanged.
- every usage-bearing chunk gets the reasoning fold + total normalize
  (the dict variants of the response post-steps — the policy's
  ``fold_usage`` hook), but the folded usage is ATTACHED only to the
  ``choices: []`` tail: v1's wrapper strips usage from every emitted
  content/finish chunk and only the synthesized final chunk carries it
  (verifier-grok F2).
- ``delta.reasoning`` is renamed to ``reasoning_content`` (the base
  handler's rename — the policy's ``reasoning="rename"`` mode) and native
  ``reasoning_content`` deltas are admitted — real Grok reasoning traffic,
  not an unreachable shape.
- a tool_call entry WITHOUT a ``type`` key gains ``type: "function"``
  (litellm's ``ChatCompletionDeltaToolCall`` default fires on the dict
  path; the SDK path yields None there, so the openai parser must not).
"""

from __future__ import annotations

from ...errors import TranslationError
from ...ir import PlainJson
from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line
from .response import fold_reasoning_tokens, normalize_usage_totals


def _folded_usage(raw_usage: PlainJson) -> PlainJson | TranslationError:
    """v1 folds + normalizes every usage-bearing chunk inside chunk_parser;
    an uncoercible token value raises out of the iterator there, so the
    coercion errors from the shared arithmetic stay loud here too."""
    if not isinstance(raw_usage, dict):
        return None
    folded = fold_reasoning_tokens(raw_usage)
    if isinstance(folded, TranslationError):
        return folded
    return normalize_usage_totals(folded)


_POLICY = HttpxChunkPolicy(reasoning="rename", fold_usage=_folded_usage)

parse_event = make_parse_event(_POLICY)
parse_line = make_parse_line(parse_event)
