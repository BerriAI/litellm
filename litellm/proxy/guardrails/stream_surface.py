"""Wire format of a buffered streaming response, for guardrails that scan whole streams.

A `post_call` guardrail receives a stream as whatever its route produces: chat-completions chunk
objects, raw Anthropic SSE bytes, Responses API event models, or opaque SSE from a route that
frames its own. Each has to be read, refused and rewritten in its own shape, so a guardrail that
branches on a single "is this raw SSE" flag silently treats three of those four as the fourth.

Classification lives here rather than in each guardrail so the surfaces stay one list.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import Enum, auto
from typing import Final

from litellm.proxy.guardrails.anthropic_sse import (
    is_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamEvents

_RESPONSES_EVENT_TYPE_PREFIX: Final = "response."

# Only these carry the finished output; response.created also carries a body, but an empty one
_RESPONSES_TERMINAL_EVENT_TYPES: Final = frozenset(
    {
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value,
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value,
        ResponsesAPIStreamEvents.RESPONSE_FAILED.value,
    }
)

# Every event whose ``delta`` is model output already on its way to the client. Read off the event
# enum rather than listed, so an event added there cannot quietly fall out of the comparison
_RESPONSES_DELTA_EVENT_TYPES: Final = frozenset(
    event.value for event in ResponsesAPIStreamEvents if event.value.endswith(".delta")
)

# What makes two delta events part of the same field of the turn, rather than two fields that
# merely streamed next to each other
_RESPONSES_DELTA_FIELD_ATTRS: Final = ("type", "item_id", "output_index", "content_index", "summary_index")


class StreamSurface(Enum):
    """Which endpoint shape a buffered stream belongs to."""

    CHAT_COMPLETIONS = auto()
    ANTHROPIC_MESSAGES = auto()
    RESPONSES = auto()
    OPAQUE_SSE = auto()


def _stream_item_field(item: object, field: str) -> object | None:
    return item.get(field) if isinstance(item, dict) else getattr(item, field, None)


def _stream_item_type(item: object) -> str | None:
    event_type: Final = _stream_item_field(item, "type")
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
        (event_type := _stream_item_type(chunk)) is not None and event_type.startswith(_RESPONSES_EVENT_TYPE_PREFIX)
        for chunk in all_chunks
    ):
        return StreamSurface.RESPONSES
    return StreamSurface.CHAT_COMPLETIONS


def final_responses_api_response(all_chunks: Sequence[object]) -> ResponsesAPIResponse | None:
    """The finished body a terminal ``/v1/responses`` event carries.

    Read with ``getattr`` rather than ``model_dump``: litellm's chat-to-Responses bridge stamps
    ``sequence_number`` straight onto ``__dict__``, which pydantic's dump path drops.

    A stream cut short before it completes has to read as unassembled rather than as a clean empty
    response, since ``response.created`` carries a body too and scanning that would release every
    buffered delta unscanned.
    """
    return next(
        (
            body
            for chunk in reversed(all_chunks)
            if _stream_item_field(chunk, "type") in _RESPONSES_TERMINAL_EVENT_TYPES
            and isinstance(body := _stream_item_field(chunk, "response"), ResponsesAPIResponse)
        ),
        None,
    )


def _responses_delta_field(chunk: object) -> tuple[str, ...]:
    return tuple(str(_stream_item_field(chunk, attr)) for attr in _RESPONSES_DELTA_FIELD_ATTRS)


def responses_delta_field_texts(all_chunks: Sequence[object]) -> tuple[str, ...]:
    """Text each field of a ``/v1/responses`` turn spelled out in its delta events.

    One field's deltas join as they streamed, since a finding can be split across them, and
    separate fields stay apart, so a reasoning summary running into the visible answer cannot
    spell out text that neither of them carries on its own.
    """
    deltas: Final = tuple(
        (_responses_delta_field(chunk), delta)
        for chunk in all_chunks
        if _stream_item_field(chunk, "type") in _RESPONSES_DELTA_EVENT_TYPES
        and isinstance(delta := _stream_item_field(chunk, "delta"), str)
    )
    return tuple(
        "".join(delta for field, delta in deltas if field == streamed_field)
        for streamed_field in dict.fromkeys(field for field, _ in deltas)
    )


def responses_deltas_absent_from_body(all_chunks: Sequence[object], body: ResponsesAPIResponse) -> tuple[str, ...]:
    """Streamed delta text the terminal body does not account for.

    Reasoning summaries and tool-call arguments reach the client through delta events that some
    providers never repeat in the finished body, so a scan of that body alone would not have seen
    them. Comparison runs over the JSON-escaped body so text carrying quotes or newlines is not
    reported missing purely because of escaping.
    """
    encoded_body: Final = json.dumps(body.model_dump(mode="json"), default=str)
    return tuple(
        text for text in responses_delta_field_texts(all_chunks) if text and json.dumps(text)[1:-1] not in encoded_body
    )


def is_terminal_error_stream(all_chunks: Sequence[object]) -> bool:
    """Whether the buffered stream is only the refusal an earlier guardrail in the chain emitted.

    `post_call` guardrails are composed, so a hook can be handed the terminal error items a
    preceding one produced. Those carry no content to scan, and replacing them would hide the
    refusal the client is owed.
    """
    if all_chunks and all(_stream_item_type(chunk) == "error" for chunk in all_chunks):
        return True
    return is_sse_error_stream(all_chunks)
