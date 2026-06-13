"""snowflake Cortex SSE chunk payloads -> IR stream events.

v1 decodes snowflake streams through the BASE
``OpenAIChatCompletionStreamingHandler`` (no iterator override;
canary-pinned — neither the content_list rewrite nor the ``snowflake/``
prefix ever runs on chunks) into ``CustomStreamWrapper``'s default openai
branch — the compat_httpx FAMILY policy over the ONE shared httpx chunk
normalizer. Whether real Cortex deltas ever carry ``content_list`` is
UNPINNED upstream (researcher-4 §9): the differential is two-sided against
v1's decode, so a content_list delta would be an unknown-delta-key typed
error in v2 where v1 serves a chunk that silently lost it — get real
fixtures before flag-on streaming. Provider error chunks are the family's
LOUD fail-closed PINNED DIVERGENCE — snowflake joins the report's single
named row. Folds use the ``"xai"`` ChunkDialect.
"""

from __future__ import annotations

from ..openai_compat.httpx_chunk import BASE_HANDLER_POLICY, make_parse_event
from ..openai_compat.stream import make_parse_line

parse_event = make_parse_event(BASE_HANDLER_POLICY)
parse_line = make_parse_line(parse_event)
