"""Wire format of a buffered streaming response, for guardrails that scan whole streams.

A `post_call` guardrail receives a stream as whatever its route produces: chat-completions chunk
objects, raw Anthropic SSE bytes, Responses API event models, or opaque SSE from a route that
frames its own. Each has to be read, refused and rewritten in its own shape, so a guardrail that
branches on a single "is this raw SSE" flag silently treats three of those four as the fourth.

Classification lives here rather than in each guardrail so the surfaces stay one list.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto
from typing import Final

from litellm.proxy.guardrails.anthropic_sse import (
    is_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)

_RESPONSES_EVENT_TYPE_PREFIX: Final = "response."


class StreamSurface(Enum):
    """Which endpoint shape a buffered stream belongs to."""

    CHAT_COMPLETIONS = auto()
    ANTHROPIC_MESSAGES = auto()
    RESPONSES = auto()
    OPAQUE_SSE = auto()


def _stream_item_type(item: object) -> str | None:
    event_type: Final = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    return event_type if isinstance(event_type, str) else None


def classify_stream(all_chunks: Sequence[object]) -> StreamSurface:
    """Wire format the buffered chunks belong to.

    Raw bytes are not enough to name the surface: ``/v1/messages`` frames its own SSE, but so does
    the Google ``:streamGenerateContent`` route, and reading the latter as Anthropic would refuse it
    in a format its client cannot parse.
    """
    if is_raw_sse_stream(all_chunks):
        return StreamSurface.ANTHROPIC_MESSAGES if is_anthropic_sse_stream(all_chunks) else StreamSurface.OPAQUE_SSE
    if any(
        (event_type := _stream_item_type(chunk)) is not None
        and event_type.startswith(_RESPONSES_EVENT_TYPE_PREFIX)
        for chunk in all_chunks
    ):
        return StreamSurface.RESPONSES
    return StreamSurface.CHAT_COMPLETIONS


def is_terminal_error_stream(all_chunks: Sequence[object]) -> bool:
    """Whether the buffered stream is only the refusal an earlier guardrail in the chain emitted.

    `post_call` guardrails are composed, so a hook can be handed the terminal error items a
    preceding one produced. Those carry no content to scan, and replacing them would hide the
    refusal the client is owed.
    """
    if all_chunks and all(_stream_item_type(chunk) == "error" for chunk in all_chunks):
        return True
    return is_sse_error_stream(all_chunks)
