"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.messages.native.utils import filtered_safeguard_results, redacted_stop_details
from litellm.llms.anthropic.messages.stream_assembly import (
    AnthropicSseFrame,
    anthropic_message_delta_fields,
    anthropic_sse_error_event_of,
    anthropic_sse_error_events,
    anthropic_sse_event_type,
    anthropic_sse_event_types,
    anthropic_sse_frames,
    anthropic_sse_reads_alike,
    anthropic_sse_started_blocks,
    anthropic_sse_stopped_block_index,
    anthropic_sse_unfinished_blocks,
)
from litellm.llms.anthropic.messages.stream_assembly import (
    anthropic_sse_assembles_every_tool_input as anthropic_sse_assembles_every_tool_input,  # noqa: PLC0414  # public re-export
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
    source_fields: Mapping[str, object], unchanged_tool_use_ids: frozenset[str], keep_explanations: bool
) -> Mapping[str, object]:
    stop_details: Final = source_fields.get("stop_details")
    safeguard_results: Final = source_fields.get("safeguard_results")
    container: Final = source_fields.get("container")
    return MappingProxyType(
        {
            **({"container": container} if container is not None else {}),
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


def anthropic_sse_chunks_of_events(events: Sequence[str]) -> tuple[bytes, ...]:
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
    its error alone. No empty text block is re-emitted, since the OpenAI round trip adds one to a response
    with only tool calls.
    """
    error_events: Final = anthropic_sse_error_events(source_chunks)
    try:
        return _rebuilt_sse_chunks(assembled, source_chunks, error_events, keep_explanations)
    except Exception:
        if not error_events:
            raise
        verbose_proxy_logger.warning("Failed Anthropic stream could not be rebuilt, sending its upstream error alone")
        return anthropic_sse_chunks_of_events(error_events)


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
        response=_without_empty_text(anthropic_response),
        message_delta_fields=_rewritten_message_delta_fields(
            anthropic_message_delta_fields(source_chunks), unchanged_tool_use_ids, keep_explanations
        ),
    ).chunks
    if not error_events:
        return tuple(rewritten_chunks)
    block_left_open: Final = _text_block_left_open(rewritten_chunks, source_chunks)
    end_events_sent: Final = _MESSAGE_END_EVENT_TYPES & anthropic_sse_event_types(source_chunks)
    return (
        *(chunk for chunk in rewritten_chunks if _sent_by_the_failed_upstream(chunk, block_left_open, end_events_sent)),
        *anthropic_sse_chunks_of_events(error_events),
    )


_BLOCK_CONTENT_KEYS: Final = MappingProxyType({"content_block_start": "content_block", "content_block_delta": "delta"})


@dataclass(frozen=True, slots=True)
class AnthropicSseBlock:
    """The text and the tool input JSON a client reads in a content block, from its start and its deltas."""

    index: int
    text: str
    tool_input: str
    replaces_start_input: bool


def _is_block_content(event: Mapping[str, object]) -> bool:
    return event.get("type") in _BLOCK_CONTENT_KEYS


def _block_content(event: Mapping[str, object]) -> Mapping[str, object] | None:
    return as_str_mapping(event.get(_BLOCK_CONTENT_KEYS[str(event.get("type"))]))


def _index_of(event: Mapping[str, object]) -> int:
    index: Final = event.get("index")
    return index if isinstance(index, int) else -1


def _holds_only_strings(content: Mapping[str, object] | None) -> bool:
    return content is not None and all(isinstance(content.get(key) or "", str) for key in ("text", "partial_json"))


def _joined_strings(contents: Sequence[Mapping[str, object]], key: str) -> str:
    return "".join(value for content in contents if isinstance(value := content.get(key), str))


def _sse_block(index: int, contents: Sequence[Mapping[str, object]]) -> AnthropicSseBlock:
    start_input: Final = contents[0].get("input")
    streamed_input: Final = _joined_strings(contents[1:], "partial_json")
    return AnthropicSseBlock(
        index=index,
        text=_joined_strings(contents, "text"),
        tool_input=streamed_input or (json.dumps(start_input) if start_input else ""),
        replaces_start_input=bool(streamed_input and start_input),
    )


def anthropic_sse_blocks(source_chunks: Sequence[object]) -> tuple[AnthropicSseBlock, ...] | None:
    """The content blocks of the stream, or None when some client reads the stream another way."""
    if not anthropic_sse_reads_alike(source_chunks):
        return None
    frames: Final = tuple(frame for frame in anthropic_sse_frames(source_chunks) if _is_block_content(frame.event))
    if not all(_holds_only_strings(_block_content(frame.event)) for frame in frames):
        return None
    by_index: Final = groupby(sorted((frame.event for frame in frames), key=_index_of), key=_index_of)
    contents: Final = (tuple(_block_content(event) or {} for event in events) for _, events in by_index)
    return tuple(_sse_block(index, block_contents) for index, block_contents in enumerate(contents))


@dataclass(frozen=True, slots=True)
class _SseRewrite:
    blocks: Mapping[int, AnthropicSseBlock]
    drops_citations: bool
    unchanged_tool_use_ids: frozenset[str]
    error_data: Mapping[str, str]


def _sse_event(event: Mapping[str, object]) -> str:
    return f"event: {event.get('type')}\ndata: {json.dumps(event)}"


def _replaced_event(frame: AnthropicSseFrame, event: Mapping[str, object]) -> str:
    return _sse_event(event) if frame.error_data is None else anthropic_sse_error_event_of(json.dumps(event))


def _block_delta(index: int, delta: Mapping[str, object]) -> str:
    return _sse_event({"type": "content_block_delta", "index": index, "delta": delta})


def _block_deltas(block: AnthropicSseBlock) -> Iterator[str]:
    if block.text:
        yield _block_delta(block.index, {"type": "text_delta", "text": block.text})
    if block.tool_input:
        yield _block_delta(block.index, {"type": "input_json_delta", "partial_json": block.tool_input})


def _rewritten_start(frame: AnthropicSseFrame, block: AnthropicSseBlock | None, rewrite: _SseRewrite) -> Iterator[str]:
    content: Final = _block_content(frame.event) or {}
    emptied: Final[Mapping[str, object]] = {
        **({"text": "", "input": {}} if block is not None else {}),
        **({"citations": []} if rewrite.drops_citations and content.get("citations") else {}),
    }
    kept: Final = {key: emptied.get(key, value) for key, value in content.items()}
    yield frame.raw if kept == content else _replaced_event(frame, {**frame.event, "content_block": kept})
    yield from () if block is None else _block_deltas(block)


def _rewritten_delta(frame: AnthropicSseFrame, block: AnthropicSseBlock | None, rewrite: _SseRewrite) -> Iterator[str]:
    delta: Final = _block_content(frame.event) or {}
    dropped_keys: Final = frozenset(
        (*(("text", "partial_json") if block is not None else ()), *(("citation",) if rewrite.drops_citations else ()))
    )
    if dropped_keys.isdisjoint(delta):
        yield frame.raw
    elif frame.error_data is not None:
        emptied: Final = {key: "" if key in dropped_keys else value for key, value in delta.items()}
        yield _replaced_event(frame, {**frame.event, "delta": emptied})


def _rewritten_message_delta(frame: AnthropicSseFrame, rewrite: _SseRewrite) -> str:
    delta: Final = as_str_mapping(frame.event.get("delta")) or {}
    fields: Final = _rewritten_message_delta_fields(delta, rewrite.unchanged_tool_use_ids, keep_explanations=False)
    kept: Final = {key: fields.get(key, value) for key, value in delta.items() if fields.get(key, "") is not None}
    return frame.raw if kept == delta else _replaced_event(frame, {**frame.event, "delta": kept})


def _with_rewritten_error(frame: AnthropicSseFrame, rewrite: _SseRewrite) -> tuple[AnthropicSseFrame, ...]:
    if frame.error_data is None or rewrite.error_data.get(frame.error_data, frame.error_data) == frame.error_data:
        return (frame,)
    error_event: Final = anthropic_sse_error_event_of(rewrite.error_data[frame.error_data])
    return anthropic_sse_frames(anthropic_sse_chunks_of_events((error_event,)))


def _rewritten_frame(frame: AnthropicSseFrame, rewrite: _SseRewrite) -> Iterator[str]:
    index: Final = frame.event.get("index")
    block: Final = rewrite.blocks.get(index) if isinstance(index, int) else None
    match frame.event.get("type"):
        case "content_block_start":
            yield from _rewritten_start(frame, block, rewrite)
        case "content_block_delta":
            yield from _rewritten_delta(frame, block, rewrite)
        case "message_delta" if rewrite.blocks:
            yield _rewritten_message_delta(frame, rewrite)
        case _:
            yield frame.raw


def anthropic_sse_chunks_rewritten(
    source_chunks: Sequence[object],
    source_blocks: Sequence[AnthropicSseBlock],
    blocks: Sequence[AnthropicSseBlock],
    error_data: Mapping[str, str],
) -> tuple[object, ...]:
    """The stream with ``blocks`` in place of its ``source_blocks`` and each ``error`` event's data mapped.

    Every other frame stays as the upstream sent it and where it sent it, so thinking keeps its signature and a cut
    stream stays cut. A rewritten block sends its whole text or input in one delta after its start. The text blocks
    lose their citations when one of them changed, since a citation quotes text the guardrail never read, and
    ``message_delta`` loses the verdicts and explanations written about what changed.
    """
    rewritten: Final = MappingProxyType(
        {
            block.index: block
            for source, block in zip(source_blocks, blocks, strict=True)
            if block != source or source.replaces_start_input
        }
    )
    if not rewritten and all(data == rewritten_data for data, rewritten_data in error_data.items()):
        return tuple(source_chunks)
    changed_inputs: Final = frozenset(
        source.index
        for source, block in zip(source_blocks, blocks, strict=True)
        if source.tool_input != block.tool_input
    )
    started: Final = anthropic_sse_started_blocks(source_chunks)
    rewrite: Final = _SseRewrite(
        blocks=rewritten,
        drops_citations=any(source.text != block.text for source, block in zip(source_blocks, blocks, strict=True)),
        unchanged_tool_use_ids=frozenset(
            block_id
            for index, block in started.items()
            if index not in changed_inputs and isinstance(block_id := block.get("id"), str)
        ),
        error_data=error_data,
    )
    frames: Final = chain.from_iterable(
        _with_rewritten_error(frame, rewrite) for frame in anthropic_sse_frames(source_chunks)
    )
    return anthropic_sse_chunks_of_events(
        tuple(chain.from_iterable(_rewritten_frame(frame, rewrite) for frame in frames))
    )
