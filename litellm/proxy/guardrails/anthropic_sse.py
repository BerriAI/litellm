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
    Final,
    cast,  # noqa: TID251  # the Messages body is a total=False TypedDict; validating through it drops provider fields
)

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.messages.native.utils import filtered_safeguard_results, redacted_stop_details
from litellm.llms.anthropic.messages.stream_assembly import (
    _joined_sse_stream,  # pyright: ignore[reportPrivateUsage]  # moved with the assembler; a copy splits trailing utf-8
    _parsed_sse_events,  # pyright: ignore[reportPrivateUsage]  # same parser assemble_anthropic_sse_stream uses
    anthropic_message_delta_fields,
    anthropic_sse_error_events,
    anthropic_sse_event_type,
    anthropic_sse_event_types,
    anthropic_sse_started_blocks,
    anthropic_sse_stopped_block_index,
    anthropic_sse_unfinished_blocks,
)
from litellm.llms.anthropic.messages.stream_assembly import (
    assemble_anthropic_sse_stream as assemble_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.messages.stream_assembly import (
    has_unreadable_json_event as has_unreadable_json_event,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.messages.stream_assembly import (
    is_anthropic_sse_stream as is_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.messages.stream_assembly import (
    is_raw_sse_stream as is_raw_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.utils import Choices, ModelResponse

_EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})

_TOOL_BLOCK_TYPES: Final = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})

_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def sse_stream_text(all_chunks: Sequence[object]) -> str | None:
    """The raw SSE frames joined as text, for a caller that forwards the stream instead of folding it."""
    return _joined_sse_stream(all_chunks)


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


from litellm.llms.anthropic.messages.stream_assembly import (  # noqa: E402  # re-export after the definitions above
    is_sse_error_stream as is_sse_error_stream,  # noqa: PLC0414  # public re-export
)


def _tool_use_inputs(content: Sequence[object]) -> Mapping[str, object]:
    blocks: Final = (block for raw in content if (block := as_str_mapping(raw)) is not None)
    return MappingProxyType(
        {
            tool_use_id: block.get("input")
            for block in blocks
            if block.get("type") == "tool_use" and isinstance(tool_use_id := block.get("id"), str)
        }
    )


def _unchanged_tool_use_ids(source_content: Sequence[object], delivered_content: Sequence[object]) -> frozenset[str]:
    source_inputs: Final = _tool_use_inputs(source_content)
    return frozenset(
        tool_use_id
        for tool_use_id, tool_input in _tool_use_inputs(delivered_content).items()
        if tool_use_id in source_inputs and source_inputs[tool_use_id] == tool_input
    )


def _rewritten_message_delta_fields(
    source_chunks: Sequence[object], unchanged_tool_use_ids: frozenset[str], keep_explanations: bool
) -> Mapping[str, object]:
    source_fields: Final = anthropic_message_delta_fields(source_chunks)
    stop_details: Final = source_fields.get("stop_details")
    safeguard_results: Final = source_fields.get("safeguard_results")
    return MappingProxyType(
        {
            **(
                {"stop_details": redacted_stop_details(stop_details, keep_explanations)}
                if stop_details is not None
                else {}
            ),
            **(
                {
                    "safeguard_results": filtered_safeguard_results(
                        safeguard_results, unchanged_tool_use_ids, keep_explanations
                    )
                }
                if safeguard_results is not None
                else {}
            ),
        }
    )


_MESSAGE_END_EVENT_TYPES: Final = frozenset({"message_delta", "message_stop"})


def _text_block_left_open(rewritten_chunks: Sequence[bytes], source_chunks: Sequence[object]) -> int | None:
    """Index of the rewritten text block to leave open when the failed upstream never stopped its text.

    Assembly keeps a cut-off text block and drops every other cut-off block, and the round trip
    merges all text into one block, so that block carries text the upstream left open. It stays
    open only when it is the last block, since a client must see each block stop before the next
    one starts.
    """
    upstream_left_text_open: Final = any(
        block.get("type") == "text" for block in anthropic_sse_unfinished_blocks(source_chunks).values()
    )
    rewritten_blocks: Final = anthropic_sse_started_blocks(rewritten_chunks)
    if not upstream_left_text_open or not rewritten_blocks:
        return None
    last_index: Final = max(rewritten_blocks)
    return last_index if rewritten_blocks[last_index].get("type") == "text" else None


def _is_empty_text(block: object) -> bool:
    fields: Final = as_str_mapping(block)
    return fields is not None and fields.get("type") == "text" and not fields.get("text")


def _without_empty_text(response: AnthropicMessagesResponse) -> AnthropicMessagesResponse:
    return {**response, "content": [block for block in response.get("content") or () if not _is_empty_text(block)]}


def _sent_by_the_failed_upstream(chunk: bytes, block_left_open: int | None, end_events_sent: frozenset[str]) -> bool:
    event: Final = chunk.decode()
    event_type: Final = anthropic_sse_event_type(event)
    if event_type in _MESSAGE_END_EVENT_TYPES:
        return event_type in end_events_sent
    return block_left_open is None or anthropic_sse_stopped_block_index(event) != block_left_open


def _sse_chunks_of_events(events: Sequence[str]) -> tuple[bytes, ...]:
    return tuple(f"{event}\n\n".encode() for event in events)


def anthropic_sse_chunks_from_response(
    assembled: ModelResponse, source_chunks: Sequence[object] = (), *, keep_explanations: bool = False
) -> tuple[bytes, ...]:
    """Re-emit a rewritten response as the SSE events of the ``source_chunks`` stream it replaces.

    ``message_delta`` carries the source stream's ``stop_details`` and ``safeguard_results`` and
    nothing the OpenAI round trip made up, with a verdict kept only for a tool call delivered with
    the input the classifier judged. Their free-text ``explanation`` was written about the text the
    guardrail rewrote, so it is dropped unless the caller rewrites no text (``keep_explanations``).
    A source stream that ended in an upstream ``error`` event ends with that event again, never
    with a ``message_delta`` or ``message_stop`` the upstream did not send, with its cut-off text
    block left open unless another block follows it, and with no finished text block when it read
    no text, since Claude Code does not retry a stream that shows a finished block. A response that
    cannot be re-emitted, such as one with a tool input that cannot be parsed, leaves a failed stream
    its error alone.
    """
    error_events: Final = anthropic_sse_error_events(source_chunks)
    try:
        return _rebuilt_sse_chunks(assembled, source_chunks, error_events, keep_explanations)
    except Exception:
        if not error_events:
            raise
        verbose_proxy_logger.warning("Failed Anthropic stream could not be rebuilt, sending its upstream error alone")
        return _sse_chunks_of_events(error_events)


def _rebuilt_sse_chunks(
    assembled: ModelResponse, source_chunks: Sequence[object], error_events: Sequence[str], keep_explanations: bool
) -> tuple[bytes, ...]:
    from litellm.llms.anthropic.messages.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.messages.native.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    adapter: Final = LiteLLMAnthropicMessagesAdapter()
    anthropic_response: Final = adapter.translate_openai_response_to_anthropic(response=assembled)
    source_response: Final = assemble_anthropic_sse_stream(source_chunks, through_upstream_error=True)
    source_content: Final = (
        adapter.translate_openai_response_to_anthropic(response=source_response).get("content")
        if source_response is not None
        else None
    )
    unchanged_tool_use_ids: Final = _unchanged_tool_use_ids(
        source_content or (), anthropic_response.get("content") or ()
    )
    rewritten_chunks: Final = FakeAnthropicMessagesStreamIterator(
        response=_without_empty_text(anthropic_response) if error_events else anthropic_response,
        message_delta_fields=_rewritten_message_delta_fields(source_chunks, unchanged_tool_use_ids, keep_explanations),
    ).chunks
    return _ended_as_the_upstream_ended(rewritten_chunks, source_chunks, error_events)


def _ended_as_the_upstream_ended(
    rewritten_chunks: Sequence[bytes], source_chunks: Sequence[object], error_events: Sequence[str]
) -> tuple[bytes, ...]:
    if not error_events:
        return tuple(rewritten_chunks)
    block_left_open: Final = _text_block_left_open(rewritten_chunks, source_chunks)
    end_events_sent: Final = _MESSAGE_END_EVENT_TYPES & anthropic_sse_event_types(source_chunks)
    return (
        *(chunk for chunk in rewritten_chunks if _sent_by_the_failed_upstream(chunk, block_left_open, end_events_sent)),
        *_sse_chunks_of_events(error_events),
    )


def _str_field(source: Mapping[str, object], key: str) -> str:
    return value if isinstance(value := source.get(key), str) else ""


def _object_field(source: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value: Final = source.get(key)
    return _JSON_OBJECT_ADAPTER.validate_python(value) if isinstance(value, Mapping) else None


def _accumulated_tool_input(joined: str) -> object:
    """Assemble a tool block's streamed arguments.

    A stream cut mid-block leaves unparseable JSON. Returning an empty input there would
    drop whatever the model had already emitted, and a caller can force exactly that by
    prompting for restricted content inside a tool argument and letting max_tokens truncate
    it: the text would reach the client in the replayed frames having never been scanned.
    So the raw fragment is carried under ``_raw`` instead, keeping the block a well-formed
    object while leaving the text visible to whatever inspects the assembled body.
    """
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
            joined: Final = "".join(self.partial_json)
            return {"input": _accumulated_tool_input(joined)} if joined else _EMPTY_MAP
        return _EMPTY_MAP

    def rendered(self) -> Mapping[str, object]:
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
    stops: Final = _object_field(event, "delta") or _EMPTY_MAP
    usage: Final = _object_field(event, "usage")
    return replace(
        state,
        stop_reason=stops.get("stop_reason", state.stop_reason),
        stop_sequence=stops.get("stop_sequence", state.stop_sequence),
        delta_usage=MappingProxyType({**state.delta_usage, **usage}) if usage is not None else state.delta_usage,
    )


def _with_event(state: _AnthropicMessage, event: Mapping[str, object]) -> _AnthropicMessage:
    match event.get("type"):
        case "message_start":
            message: Final = _object_field(event, "message")
            return state if message is None else replace(state, envelope=message)
        case "content_block_start":
            index: Final = event.get("index")
            block: Final = _object_field(event, "content_block")
            if not isinstance(index, int) or block is None:
                return state
            return _with_block(state, index, _ContentBlock(start=block))
        case "content_block_delta":
            delta_index: Final = event.get("index")
            delta: Final = _object_field(event, "delta")
            if not isinstance(delta_index, int) or delta is None:
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


def anthropic_sse_chunks_from_body(
    body: Mapping[str, object], source_chunks: Sequence[object] = ()
) -> tuple[bytes, ...]:
    """Re-emit a native Anthropic Messages body as the SSE frames of the ``source_chunks`` stream it replaces.

    A source stream that ended in an upstream ``error`` event ends with that event again, the same
    way ``anthropic_sse_chunks_from_response`` re-emits one.
    """
    from litellm.llms.anthropic.messages.native.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    error_events: Final = anthropic_sse_error_events(source_chunks)
    typed: Final = cast(AnthropicMessagesResponse, dict(body))  # cast-ok: validating drops provider-only fields
    rewritten_chunks: Final = FakeAnthropicMessagesStreamIterator(
        response=_without_empty_text(typed) if error_events else typed
    ).chunks
    return _ended_as_the_upstream_ended(rewritten_chunks, source_chunks, error_events)
