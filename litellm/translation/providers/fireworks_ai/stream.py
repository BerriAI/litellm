"""fireworks_ai SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1 decodes fireworks streams through the BASE
``OpenAIChatCompletionStreamingHandler`` (no iterator override;
canary-pinned — ``transform_response``'s ``fireworks_ai/`` prefix and the
content repair never run on chunks, so stream chunk models are the BARE
wire model) into ``CustomStreamWrapper``'s default openai branch — the
compat_httpx FAMILY policy over the ONE shared httpx chunk normalizer:
``reasoning="rename"``, value-checked error chunks, no required envelope
keys, wire usage verbatim on the ``choices: []`` tail only. Provider error
chunks are the family's LOUD fail-closed PINNED DIVERGENCE (v1's base
handler silently swallows them) — fireworks_ai joins the report's single
named row. Folds use the ``"xai"`` ChunkDialect.
"""

from __future__ import annotations

from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line

parse_event = make_parse_event(HttpxChunkPolicy(reasoning="rename"))
parse_line = make_parse_line(parse_event)
