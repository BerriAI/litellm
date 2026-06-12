"""deepseek SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1 decodes deepseek streams through the BASE
``OpenAIChatCompletionStreamingHandler`` (``DeepSeekChatConfig`` has no
iterator override; verified in-process at HEAD) into ``CustomStreamWrapper``'s
default openai branch — exactly the compat_httpx FAMILY policy over the ONE
shared httpx chunk normalizer: ``reasoning="rename"`` (the base handler's
unconditional pop-rename), value-checked error chunks, no required envelope
keys, wire usage verbatim on the ``choices: []`` tail only. Provider error
chunks are a LOUD v2 boundary error where v1's base handler silently
swallows them — the SAME deliberate fail-closed divergence the compat_httpx
family pinned (two-sided rows in test_differential_deepseek_stream.py; the
report's single PINNED DIVERGENCE row names every base-handler consumer).
Folds use the ``"xai"`` ChunkDialect (the generic httpx dict path).
"""

from __future__ import annotations

from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line

parse_event = make_parse_event(HttpxChunkPolicy(reasoning="rename"))
parse_line = make_parse_line(parse_event)
