"""groq SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1's decode is ``GroqChatCompletionStreamingHandler(
OpenAIChatCompletionStreamingHandler)`` over standard ``data:``/[DONE] SSE:
a TRUTHY ``error`` value raises OpenAIError (the xai-style VALUE check, not
cometapi's key-presence check), ``delta.reasoning`` is POPPED into
``reasoning_content``, then the BASE rebuild runs. That is EXACTLY the
shared httpx_chunk factory with ``reasoning="rename"`` — the longtail
guidance's prediction ("groq's pop semantics == the existing rename mode"),
verified against the v1 source: the pre-step pop followed by the base
handler's own rename is output-identical to one rename. NO new
ReasoningMode arm was needed (the no-consumer-no-arm rule holds).
"""

from __future__ import annotations

from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line

_POLICY = HttpxChunkPolicy(reasoning="rename")

parse_event = make_parse_event(_POLICY)
parse_line = make_parse_line(parse_event)
