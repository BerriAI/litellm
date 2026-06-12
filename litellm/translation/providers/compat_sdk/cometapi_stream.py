"""cometapi SSE ``chat.completion.chunk`` payloads -> IR stream events.

The v1 decode is the httpx dict path: ``BaseModelResponseIterator`` strips
``data:`` lines, ``CometAPIChatCompletionStreamingHandler.chunk_parser``
(llms/cometapi/chat/transformation.py:164-206) rebuilds each chunk, and the
result flows through ``CustomStreamWrapper``'s default openai branch. The
folds use the ``xai`` chunk dialect (the shared httpx-wrapper dialect:
reasoning-only deltas count as non-empty, the model is never re-read from
chunks).

The machinery is the shared httpx chunk normalizer
(``openai_compat.httpx_chunk.make_parse_event`` — ONE mechanism for xai,
cometapi, and the wave-2b dict-path providers; critic-wave2a M2), composed
with the cometapi policy. The cometapi facts, all verified in-process at
HEAD against v1 replays:

- ``reasoning="copy_both"``: ``delta.reasoning`` is COPIED into
  ``reasoning_content`` when the key is present and the original key
  SURVIVES (v1 assigns without popping — both keys reach the emitted Delta
  dump); native ``reasoning_content`` deltas pass through verbatim.
- strict envelope: v1 KeyErrors (-> CometAPIException -> BadRequestError) on
  chunks missing ``id``/``created``/``model``/``choices``, and raises on any
  chunk CARRYING an ``error`` key (presence, not value: chunk_parser checks
  ``"error" in chunk``) — both stay loud error values here, never served.
- the rebuild keeps ONLY id/created/usage/model/choices (extras and
  ``system_fingerprint`` dropped), usage rides only the ``choices: []``
  tail, and a type-less tool_call entry gains ``type: "function"`` — the
  factory's shared dict-path behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line


def _missing_keys_reason(missing: Sequence[str]) -> str:
    return (
        f"stream chunk missing {list(missing)!r}; v1's chunk_parser raises "
        "KeyError (CometAPIException) on it"
    )


_POLICY = HttpxChunkPolicy(
    reasoning="copy_both",
    error_on_key_presence=True,
    required_keys=("id", "created", "model", "choices"),
    missing_keys_reason=_missing_keys_reason,
)

parse_event = make_parse_event(_POLICY)
parse_line = make_parse_line(parse_event)
