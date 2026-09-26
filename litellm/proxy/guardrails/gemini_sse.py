"""Gemini SSE <-> ModelResponse conversion for guardrail streaming hooks.

The Google ``:streamGenerateContent`` route relays the upstream ``data:`` frames as raw bytes, so a
guardrail's ``async_post_call_streaming_iterator_hook`` cannot read them as chunk objects. These
helpers assemble such a stream into a ModelResponse and re-emit it once the guardrail rewrote it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm.proxy.guardrails.anthropic_sse import joined_sse_stream, parsed_sse_events
from litellm.types.utils import ModelResponse

_GEMINI_RESPONSE_KEYS: Final = frozenset({"candidates", "usageMetadata", "promptFeedback"})


@dataclass(frozen=True, slots=True)
class _StreamParserLogging:
    """The one attribute the Gemini stream parser reads off its logging object."""

    optional_params: Mapping[str, object]


def is_gemini_sse_stream(all_chunks: Sequence[object]) -> bool:
    sse_stream: Final = joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    return any(not _GEMINI_RESPONSE_KEYS.isdisjoint(event) for event in parsed_sse_events(sse_stream))


def assemble_gemini_sse_stream(all_chunks: Sequence[object]) -> ModelResponse | None:
    """Assemble raw Gemini SSE frames into a ModelResponse.

    A frame the Gemini parser rejects (a mid-stream ``error`` payload, say) raises the parser's own
    error so the caller reports the upstream failure rather than a guardrail one.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator
    from litellm.main import stream_chunk_builder

    sse_stream: Final = joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    parser: Final = ModelResponseIterator(
        streaming_response=None,
        sync_stream=False,
        logging_obj=_StreamParserLogging(optional_params={}),  # pyright: ignore[reportArgumentType]  # the parser reads only optional_params, and a native Gemini request carries no legacy `functions`
    )
    chunks: Final = tuple(
        parsed for event in parsed_sse_events(sse_stream) if (parsed := parser.chunk_parser(dict(event))) is not None
    )
    if not chunks:
        return None
    assembled: Final = stream_chunk_builder(chunks=list(chunks))
    return assembled if isinstance(assembled, ModelResponse) else None


def gemini_sse_chunks_from_response(assembled: ModelResponse) -> tuple[bytes, ...]:
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

    body: Final = GoogleGenAIAdapter().translate_completion_to_generate_content(response=assembled)
    return (f"data: {json.dumps(body)}\n\n".encode(),)
