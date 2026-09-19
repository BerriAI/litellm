"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    cast,  # noqa: TID251  # the Messages body is a total=False TypedDict; validating through it drops provider fields
)

from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})

_TOOL_BLOCK_TYPES: Final = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})

_ANTHROPIC_EVENT_TYPES: Final = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    }
)


def is_raw_sse_stream(all_chunks: Sequence[object]) -> bool:
    return any(isinstance(chunk, (str, bytes)) for chunk in all_chunks)


def sse_stream_text(all_chunks: Sequence[object]) -> str | None:
    """The raw SSE frames joined as text, for a caller that forwards the stream instead of folding it."""
    return _joined_sse_stream(all_chunks)


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


def _parsed_sse_events(sse_stream: str) -> tuple[Mapping[str, object], ...]:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )

    return tuple(
        event_data
        for event in AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(sse_stream)  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses
        if (event_data := AnthropicPassthroughLoggingHandler._extract_sse_data(event)) is not None  # pyright: ignore[reportPrivateUsage]  # same parser the assembler uses; a private import beats forking SSE parsing
    )


def _anthropic_message_start(sse_stream: str) -> Mapping[str, object] | None:
    return next(
        (
            message
            for event_data in _parsed_sse_events(sse_stream)
            if event_data.get("type") == "message_start" and isinstance(message := event_data.get("message"), dict)
        ),
        None,
    )


def is_anthropic_sse_stream(all_chunks: Sequence[object]) -> bool:
    """Whether raw SSE frames are Anthropic Messages events.

    ``is_raw_sse_stream`` only says the chunks are unparsed bytes, and ``/v1/messages`` is not the
    only endpoint that streams those: the Google ``:streamGenerateContent`` route marks its own
    stream raw too. Reading its frames as Anthropic ones would refuse the response in a wire format
    its client cannot parse, so the surface is decided on the event types actually present.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    return any(event.get("type") in _ANTHROPIC_EVENT_TYPES for event in _parsed_sse_events(sse_stream))


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


def is_sse_error_stream(all_chunks: Sequence[object]) -> bool:
    """Whether the buffered stream carries nothing but error frames.

    post_call guardrails run in a chain, so a hook can be handed the terminal error frames an
    earlier guardrail emitted when it blocked. Those carry no message to assemble, and replacing
    them would hide the refusal the client is owed. Covers both wire forms a guardrail emits: the
    Anthropic ``error`` event and the chat-completions ``{"error": ...}`` payload.
    """
    if not all(isinstance(chunk, (str, bytes)) for chunk in all_chunks):
        # A stream mixing typed chunks with an error frame still carries content to scan, and the
        # frames-only join below would drop exactly the part that has to be scanned
        return False
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    events: Final = _parsed_sse_events(sse_stream)
    return len(events) > 0 and all(
        event.get("type") == "error" or isinstance(event.get("error"), Mapping) for event in events
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


def _str_field(source: Mapping[str, object], key: str) -> str:
    return value if isinstance(value := source.get(key), str) else ""


def _accumulated_tool_input(partial_json: Sequence[str]) -> object:
    """Assemble a tool block's streamed arguments.

    A stream cut mid-block leaves unparseable JSON. Returning an empty input there would
    drop whatever the model had already emitted, and a caller can force exactly that by
    prompting for restricted content inside a tool argument and letting max_tokens truncate
    it: the text would reach the client in the replayed frames having never been scanned.
    So the raw fragment is carried under ``_raw`` instead, keeping the block a well-formed
    object while leaving the text visible to whatever inspects the assembled body.
    """
    joined: Final = "".join(partial_json)
    if not joined:
        return {}
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"_raw": joined}


@dataclass(frozen=True, slots=True)
class _ContentBlock:
    """A ``content_block_start`` and every delta that has landed on it since."""

    start: Mapping[str, object]
    text: tuple[str, ...] = ()
    thinking: tuple[str, ...] = ()
    signature: tuple[str, ...] = ()
    partial_json: tuple[str, ...] = ()
    citations: tuple[object, ...] = ()

    def with_delta(self, delta: Mapping[str, object]) -> _ContentBlock:
        match delta.get("type"):
            case "text_delta":
                return replace(self, text=(*self.text, _str_field(delta, "text")))
            case "thinking_delta":
                return replace(self, thinking=(*self.thinking, _str_field(delta, "thinking")))
            case "signature_delta":
                return replace(self, signature=(*self.signature, _str_field(delta, "signature")))
            case "input_json_delta":
                return replace(self, partial_json=(*self.partial_json, _str_field(delta, "partial_json")))
            case "citations_delta":
                citation: Final = delta.get("citation")
                return self if citation is None else replace(self, citations=(*self.citations, citation))
            case _:
                return self

    def _accumulated_fields(self) -> Mapping[str, object]:
        block_type: Final = self.start.get("type")
        if block_type == "text":
            return {"text": "".join(self.text)}
        if block_type == "thinking":
            return {"thinking": "".join(self.thinking), "signature": "".join(self.signature)}
        if block_type in _TOOL_BLOCK_TYPES:
            return {"input": _accumulated_tool_input(self.partial_json)}
        return _EMPTY_MAP

    def rendered(self) -> dict[str, object]:
        cited: Final[Mapping[str, object]] = (
            MappingProxyType({"citations": tuple(self.citations)}) if self.citations else _EMPTY_MAP
        )
        return {**self.start, **self._accumulated_fields(), **cited}


_EMPTY_BLOCKS: Final[Mapping[int, _ContentBlock]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _AnthropicMessage:
    envelope: Mapping[str, object] | None = None
    blocks: Mapping[int, _ContentBlock] = _EMPTY_BLOCKS
    stop_reason: object = None
    stop_sequence: object = None
    delta_usage: Mapping[str, object] = _EMPTY_MAP


def _with_block(state: _AnthropicMessage, index: int, block: _ContentBlock) -> _AnthropicMessage:
    return replace(state, blocks=MappingProxyType({**state.blocks, index: block}))


def _with_message_delta(state: _AnthropicMessage, event: Mapping[str, object]) -> _AnthropicMessage:
    delta: Final = event.get("delta")
    usage: Final = event.get("usage")
    stops: Final = delta if isinstance(delta, Mapping) else _EMPTY_MAP
    return replace(
        state,
        stop_reason=stops.get("stop_reason", state.stop_reason),
        stop_sequence=stops.get("stop_sequence", state.stop_sequence),
        delta_usage=(
            MappingProxyType({**state.delta_usage, **usage}) if isinstance(usage, Mapping) else state.delta_usage
        ),
    )


def _with_event(state: _AnthropicMessage, event: Mapping[str, object]) -> _AnthropicMessage:
    match event.get("type"):
        case "message_start":
            message: Final = event.get("message")
            return replace(state, envelope=message) if isinstance(message, Mapping) else state
        case "content_block_start":
            index: Final = event.get("index")
            block: Final = event.get("content_block")
            if not isinstance(index, int) or not isinstance(block, Mapping):
                return state
            return _with_block(state, index, _ContentBlock(start=block))
        case "content_block_delta":
            delta_index: Final = event.get("index")
            delta: Final = event.get("delta")
            if not isinstance(delta_index, int) or not isinstance(delta, Mapping):
                return state
            started: Final = state.blocks.get(delta_index)
            return state if started is None else _with_block(state, delta_index, started.with_delta(delta))
        case "message_delta":
            return _with_message_delta(state, event)
        case _:
            return state


def assemble_anthropic_sse_body(all_chunks: Sequence[object]) -> Mapping[str, object] | None:
    """Fold raw Anthropic SSE frames back into the non-streaming Messages body.

    ``assemble_anthropic_sse_stream`` folds them into a ``ModelResponse``, which is all a guardrail
    scanning text needs. A guardrail that posts the provider body itself needs the shape the
    non-streaming route already sends, because the chat-completions one has no field for thinking
    blocks, citations, ``stop_sequence`` or the cache token split and drops them silently.
    """
    sse_stream: Final = _joined_sse_stream(all_chunks)
    if sse_stream is None:
        return None
    state: Final = reduce(_with_event, _parsed_sse_events(sse_stream), _AnthropicMessage())
    envelope: Final = state.envelope
    if envelope is None:
        return None
    started_usage: Final = envelope.get("usage")
    return {
        **envelope,
        "content": tuple(block.rendered() for _, block in sorted(state.blocks.items())),
        "stop_reason": state.stop_reason if state.stop_reason is not None else envelope.get("stop_reason"),
        "stop_sequence": state.stop_sequence if state.stop_sequence is not None else envelope.get("stop_sequence"),
        "usage": {
            **(started_usage if isinstance(started_usage, Mapping) else _EMPTY_MAP),
            **state.delta_usage,
        },
    }


def anthropic_sse_chunks_from_body(body: Mapping[str, object]) -> tuple[bytes, ...]:
    """Re-emit a native Anthropic Messages body as the SSE frames its client expects."""
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    typed: Final = cast("AnthropicMessagesResponse", dict(body))
    return tuple(FakeAnthropicMessagesStreamIterator(response=typed).chunks)
