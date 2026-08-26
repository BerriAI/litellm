"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

from litellm.types.utils import Choices, ModelResponse


def is_raw_sse_stream(all_chunks: Sequence[object]) -> bool:
    return any(isinstance(chunk, (str, bytes)) for chunk in all_chunks)


def _joined_sse_stream(all_chunks: Sequence[object]) -> str | None:
    raw: Final = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        for chunk in all_chunks
        if isinstance(chunk, (str, bytes))
    )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _anthropic_message_start(sse_stream: str) -> Mapping[str, object] | None:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    return next(
        (
            message
            for event in AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(sse_stream)  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses
            if (event_data := AnthropicPassthroughLoggingHandler._extract_sse_data(event)) is not None  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses; a private import beats forking SSE parsing
            and event_data.get("type") == "message_start"
            and isinstance(message := event_data.get("message"), dict)
        ),
        None,
    )


def assemble_anthropic_sse_stream(
    all_chunks: Sequence[object], *, restore_identity: bool = False
) -> ModelResponse | None:
    """Assemble raw Anthropic SSE frames into a ModelResponse.

    ``restore_identity`` stamps the upstream message id and model onto the result, which the
    assembler does not carry through. It is off by default so callers that re-emit the assembled
    response keep the wire shape they had before this helper was shared. The writes land on a
    freshly built object that is unreachable from caller state until returned.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    message_start: Final = _anthropic_message_start(sse_stream)
    if message_start is None:
        return None
    model: Final = message_start.get("model") if restore_identity else None
    try:
        assembled: Final = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(  # pyright: ignore[reportPrivateUsage]  # the only SSE-to-ModelResponse assembler; reimplementing it here would fork the parser
            all_chunks=(sse_stream,),
            litellm_logging_obj=None,  # pyright: ignore[reportArgumentType]  # only forwarded to stream_chunk_builder, which accepts None
            model=model if isinstance(model, str) else "",
        )
    except Exception:  # noqa: BLE001  # stream_chunk_builder re-raises every assembly failure as litellm.APIError
        return None
    if not isinstance(assembled, ModelResponse):
        return None
    if not restore_identity:
        return assembled
    message_id: Final = message_start.get("id")
    if isinstance(message_id, str):
        assembled.id = message_id
    if isinstance(model, str) and model:
        assembled.model = model
    return assembled


def model_response_text(response: ModelResponse) -> str:
    """Assistant text of a response, used to detect whether a guardrail rewrote it."""
    return "".join(
        choice.message.content
        for choice in response.choices
        if isinstance(choice, Choices)  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime choices can be StreamingChoices
        and isinstance(choice.message.content, str)
    )


def anthropic_sse_error_frames(message: str) -> tuple[bytes, ...]:
    """Anthropic error event, for a failure discovered after the response headers were flushed.

    Once a keepalive ping has been sent a raise cannot reach the client, so the failure has to
    travel as a frame.
    """
    body: Final = json.dumps(message)
    return (
        f'event: error\ndata: {{"type": "error", "error": {{"type": "guardrail_error", '
        f'"message": {body}}}}}\n\n'.encode(),
    )


def _is_text_delta_event(event: str) -> bool:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    event_data: Final = AnthropicPassthroughLoggingHandler._extract_sse_data(event)  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses
    if event_data is None or event_data.get("type") != "content_block_delta":
        return False
    delta: Final = event_data.get("delta")
    return isinstance(delta, dict) and delta.get("type") == "text_delta"


def _text_delta_frame(text: str, index: object) -> bytes:
    payload: Final = {  # mutable-ok: json.dumps serializes a dict
        "type": "content_block_delta",
        "index": index if isinstance(index, int) else 0,
        "delta": {"type": "text_delta", "text": text},  # mutable-ok: json.dumps serializes a dict
    }
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


def _content_block_index(event: str) -> object:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    event_data: Final = AnthropicPassthroughLoggingHandler._extract_sse_data(event)  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses
    return None if event_data is None else event_data.get("index")


def rewrite_anthropic_sse_text(all_chunks: Sequence[object], replacement: str) -> tuple[bytes, ...] | None:
    """Re-emit the upstream frames with the assistant text replaced by ``replacement``.

    Rebuilding the stream from an assembled ``ModelResponse`` loses everything the assembler does
    not model, such as the cache, server-tool-use and service-tier usage fields Anthropic sends in
    ``message_start``. Rewriting the frames in place keeps them. Returns None when the stream holds
    no text delta to replace.

    Model Armor sanitizes the whole assistant text as one string, so a response split over several
    text blocks gets all of its sanitized text in the first block and the later text blocks come out
    empty. Block order and every start/stop pair still hold, and concatenating the text yields
    exactly what Model Armor returned.
    """
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    events: Final = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(sse_stream)  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses
    text_positions: Final = tuple(position for position, event in enumerate(events) if _is_text_delta_event(event))
    if not text_positions:
        return None
    dropped: Final = frozenset(text_positions[1:])
    return tuple(
        _text_delta_frame(replacement, _content_block_index(event))
        if position == text_positions[0]
        else f"{event}\n\n".encode()
        for position, event in enumerate(events)
        if position not in dropped
    )


def anthropic_sse_chunks_from_response(assembled: ModelResponse) -> tuple[bytes, ...]:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    anthropic_response: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=assembled
    )
    return tuple(FakeAnthropicMessagesStreamIterator(response=anthropic_response).chunks)
