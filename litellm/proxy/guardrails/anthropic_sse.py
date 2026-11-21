"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from typing_extensions import assert_never

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.pass_through.stream_assembly import anthropic_sse_events, is_sse_error_event
from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream as assemble_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_anthropic_sse_stream as is_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_raw_sse_stream as is_raw_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter


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


from litellm.llms.anthropic.pass_through.stream_assembly import (  # noqa: E402  # re-export after the definitions above
    is_sse_error_stream as is_sse_error_stream,  # noqa: PLC0414  # public re-export
)

_BlockKey: TypeAlias = tuple[str, str]
_Content: TypeAlias = tuple[Mapping[str, object], ...]

_UPSTREAM_VERDICT_KEYS: Final = frozenset({"stop_details", "safeguard_results"})
_MESSAGE_EVENT_TYPES: Final = frozenset({"message_start", "message_delta", "message_stop", "ping"})
_BLOCK_EVENT_TYPES: Final = frozenset({"content_block_start", "content_block_delta", "content_block_stop"})


def _block_key(block: Mapping[str, object]) -> _BlockKey | None:
    """A block's identity across the source stream and the rewrite: its own ``id``, or the call it answers."""
    block_id: Final = block.get("id")
    if isinstance(block_id, str):
        return ("id", block_id)
    tool_use_id: Final = block.get("tool_use_id")
    if isinstance(tool_use_id, str):
        return (str(block.get("type")), tool_use_id)
    return None


def _keyed_blocks(content: _Content) -> Mapping[_BlockKey, Mapping[str, object]]:
    return MappingProxyType({key: block for block in content if (key := _block_key(block)) is not None})


def _unkeyed_group(content: _Content, block_type: object) -> _Content:
    return tuple(block for block in content if _block_key(block) is None and block.get("type") == block_type)


def _is_empty_text(block: Mapping[str, object]) -> bool:
    return block.get("type") == "text" and not block.get("text")


@dataclass(frozen=True, slots=True)
class _ContentDiff:
    """The source stream's content against the rewritten response's, both read through the same OpenAI round trip,
    so a block counts as changed only when the guardrail changed it."""

    source_content: _Content
    rewritten_content: _Content
    source_keyed: Mapping[_BlockKey, Mapping[str, object]]
    rewritten_keyed: Mapping[_BlockKey, Mapping[str, object]]

    @classmethod
    def of(cls, source_content: _Content, rewritten_content: _Content) -> _ContentDiff:
        return cls(source_content, rewritten_content, _keyed_blocks(source_content), _keyed_blocks(rewritten_content))

    @property
    def dropped_ids(self) -> frozenset[str]:
        return frozenset(key[1] for key in self.source_keyed if key[0] == "id" and key not in self.rewritten_keyed)

    def changed_group(self, block_type: object) -> _Content | None:
        """The rewritten blocks of an id-less type, or None when the rewrite left that type as it was."""
        rewritten_group: Final = _unkeyed_group(self.rewritten_content, block_type)
        return None if rewritten_group == _unkeyed_group(self.source_content, block_type) else rewritten_group


@dataclass(frozen=True, slots=True)
class _MessageEvent:
    event: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Block:
    content_block: Mapping[str, object]
    events: _Content


_Segment: TypeAlias = _MessageEvent | _Block


def _event_index(event: Mapping[str, object]) -> int | None:
    index: Final = event.get("index")
    return index if isinstance(index, int) else None


def _source_block(index: int, events: _Content) -> _Block | None:
    block_events: Final = tuple(
        event for event in events if _event_index(event) == index and event.get("type") in _BLOCK_EVENT_TYPES
    )
    content_block: Final = next(
        (
            block
            for event in block_events
            if event.get("type") == "content_block_start"
            and (block := as_str_mapping(event.get("content_block"))) is not None
        ),
        None,
    )
    return None if content_block is None else _Block(content_block, block_events)


def _is_message_event(event: Mapping[str, object]) -> bool:
    return event.get("type") in _MESSAGE_EVENT_TYPES and _event_index(event) is None


def _source_segments(events: _Content) -> tuple[_Segment, ...]:
    message_events: Final = (
        (position, _MessageEvent(event)) for position, event in enumerate(events) if _is_message_event(event)
    )
    block_starts: Final = (
        (position, index)
        for position, event in enumerate(events)
        if event.get("type") == "content_block_start" and (index := _event_index(event)) is not None
    )
    blocks: Final = (
        (position, block) for position, index in block_starts if (block := _source_block(index, events)) is not None
    )
    return tuple(segment for _, segment in sorted((*message_events, *blocks), key=lambda pair: pair[0]))


def _content_block_body_events(block: Mapping[str, object]) -> _Content:
    match block.get("type"):
        case "text":
            return (
                {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": block.get("text", "")}},
            )
        case "thinking":
            thinking: Final = block.get("thinking", "")
            return (
                {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                *(
                    ({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": thinking}},)
                    if thinking
                    else ()
                ),
                {
                    "type": "content_block_delta",
                    "delta": {"type": "signature_delta", "signature": block.get("signature") or ""},
                },
            )
        case "tool_use" | "server_tool_use":
            return (
                {"type": "content_block_start", "content_block": {**block, "input": {}}},
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
                },
            )
        case _:
            return ({"type": "content_block_start", "content_block": block},)


def _synthesized_block(content_block: Mapping[str, object]) -> _Block:
    return _Block(content_block, (*_content_block_body_events(content_block), {"type": "content_block_stop"}))


def _synthesized_group(group: _Content) -> tuple[_Block, ...]:
    return tuple(_synthesized_block(block) for block in group if not _is_empty_text(block))


def _blocks_of_type(blocks: Sequence[_Block], block_type: object) -> tuple[_Block, ...]:
    return tuple(block for block in blocks if block.content_block.get("type") == block_type)


def _spliced_unkeyed_block(block: _Block, source_blocks: Sequence[_Block], diff: _ContentDiff) -> tuple[_Block, ...]:
    """An id-less block (text, thinking) is matched by type, so a changed type is re-emitted whole where it began."""
    block_type: Final = block.content_block.get("type")
    rewritten_group: Final = diff.changed_group(block_type)
    if rewritten_group is None:
        return (block,)
    if _blocks_of_type(source_blocks, block_type)[0] is not block:
        return ()
    return _synthesized_group(rewritten_group)


def _changed_block(
    content_block: Mapping[str, object], source_block: Mapping[str, object], rewritten_block: Mapping[str, object]
) -> Mapping[str, object]:
    """The upstream block with the fields the guardrail changed, so a round trip artifact (``server_tool_use`` read
    back as ``tool_use``) never reaches the client."""
    changed: Final = {key: value for key, value in rewritten_block.items() if source_block.get(key) != value}
    return {**content_block, **changed}


def _spliced_source_block(block: _Block, source_blocks: Sequence[_Block], diff: _ContentDiff) -> tuple[_Block, ...]:
    """A source block as the client gets it, kept as the upstream sent it unless the guardrail changed it.

    A block the round trip cannot represent, such as ``mcp_tool_use``, is one no guardrail read or
    changed, so it is kept unless it answers a call the guardrail dropped.
    """
    key: Final = _block_key(block.content_block)
    if key is None:
        return _spliced_unkeyed_block(block, source_blocks, diff)
    if block.content_block.get("tool_use_id") in diff.dropped_ids:
        return ()
    source_block: Final = diff.source_keyed.get(key)
    if source_block is None:
        return (block,)
    rewritten_block: Final = diff.rewritten_keyed.get(key)
    if rewritten_block is None:
        return ()
    if source_block == rewritten_block:
        return (block,)
    return (_synthesized_block(_changed_block(block.content_block, source_block, rewritten_block)),)


def _is_new_block(block: Mapping[str, object], source_block_types: frozenset[object], diff: _ContentDiff) -> bool:
    key: Final = _block_key(block)
    if key is not None:
        return key not in diff.source_keyed
    block_type: Final = block.get("type")
    return block_type not in source_block_types and diff.changed_group(block_type) is not None


def _new_blocks(source_blocks: Sequence[_Block], diff: _ContentDiff) -> tuple[_Block, ...]:
    """Blocks the guardrail wrote that no source block stands for, such as the text replacing a denied tool call."""
    source_block_types: Final = frozenset(block.content_block.get("type") for block in source_blocks)
    return _synthesized_group(
        tuple(block for block in diff.rewritten_content if _is_new_block(block, source_block_types, diff))
    )


def _spliced_segment(segment: _Segment, source_blocks: Sequence[_Block], diff: _ContentDiff) -> tuple[_Segment, ...]:
    match segment:
        case _MessageEvent():
            return (segment,)
        case _Block():
            return _spliced_source_block(segment, source_blocks, diff)
        case _:
            assert_never(segment)


def _precedes_new_blocks(segment: _Segment) -> bool:
    return isinstance(segment, _Block) or segment.event.get("type") == "message_start"


def _spliced_segments(segments: Sequence[_Segment], diff: _ContentDiff) -> tuple[_Segment, ...]:
    source_blocks: Final = tuple(segment for segment in segments if isinstance(segment, _Block))
    new_blocks: Final = _new_blocks(source_blocks, diff)
    anchor: Final = max(
        (position for position, segment in enumerate(segments) if _precedes_new_blocks(segment)), default=-1
    )
    return (
        *(new_blocks if anchor == -1 else ()),
        *chain.from_iterable(
            (*_spliced_segment(segment, source_blocks, diff), *(new_blocks if position == anchor else ()))
            for position, segment in enumerate(segments)
        ),
    )


def _without_upstream_verdicts(fields: Mapping[str, object]) -> Mapping[str, object]:
    """``stop_details`` and ``safeguard_results`` can quote the text the guardrail rewrote, so only a null one stays."""
    return {key: value for key, value in fields.items() if key not in _UPSTREAM_VERDICT_KEYS or value is None}


def _delivered_message_event(
    event: Mapping[str, object], stop_reason_patch: Mapping[str, object]
) -> Mapping[str, object]:
    match event.get("type"):
        case "message_start":
            message: Final = as_str_mapping(event.get("message"))
            return (
                event
                if message is None
                else {**event, "message": {**_without_upstream_verdicts(message), "content": []}}
            )
        case "message_delta":
            delta: Final = as_str_mapping(event.get("delta")) or {}
            return {**event, "delta": {**_without_upstream_verdicts(delta), **stop_reason_patch}}
        case _:
            return event


def _segment_events(
    segment: _Segment, position: int, block_positions: Sequence[int], stop_reason_patch: Mapping[str, object]
) -> _Content:
    match segment:
        case _MessageEvent(event=event):
            return (_delivered_message_event(event, stop_reason_patch),)
        case _Block(events=events):
            index: Final = block_positions.index(position)
            return tuple({**event, "index": index} for event in events)
        case _:
            assert_never(segment)


def _sse_chunks_of_segments(segments: Sequence[_Segment], stop_reason_patch: Mapping[str, object]) -> tuple[bytes, ...]:
    """The segments as SSE events, each block numbered by its place in the delivered stream."""
    block_positions: Final = tuple(position for position, segment in enumerate(segments) if isinstance(segment, _Block))
    events: Final = chain.from_iterable(
        _segment_events(segment, position, block_positions, stop_reason_patch)
        for position, segment in enumerate(segments)
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _content_blocks(response: AnthropicMessagesResponse) -> _Content:
    return tuple(block for raw in response.get("content") or () if (block := as_str_mapping(raw)) is not None)


def _healthy_source(
    source_chunks: Sequence[object], source_events: _Content, adapter: LiteLLMAnthropicMessagesAdapter
) -> AnthropicMessagesResponse | None:
    """The source stream as the OpenAI round trip reads it, or None for one that failed or cannot be read."""
    if any(is_sse_error_event(event) for event in source_events):
        return None
    source_response: Final = assemble_anthropic_sse_stream(source_chunks)
    if source_response is None:
        return None
    try:
        return adapter.translate_openai_response_to_anthropic(response=source_response)
    except ValueError:
        return None


def anthropic_sse_chunks_from_response(assembled: ModelResponse, source_chunks: Sequence[object]) -> tuple[bytes, ...]:
    """Re-emit a rewritten response as the ``source_chunks`` stream it replaces, changed only where the guardrail
    changed it.

    Every event and block the guardrail left alone reaches the client as the upstream sent it, so a field
    the OpenAI-shaped round trip cannot carry (an ``mcp_tool_use`` block, a tool's ``caller``, the
    ``message_start`` usage and null keys, the upstream message id) still does. A block the guardrail
    changed is re-emitted in its place from the rewrite, a dropped tool call leaves with every result
    that answers it, and text the guardrail wrote where the upstream had none follows the last block.
    ``message_start`` always goes out with empty ``content``, an event of a type the Messages stream
    does not define is left out, and ``message_delta`` takes the rewrite's ``stop_reason`` when the
    rewrite changed it.

    A source stream that carries an upstream error, or that cannot be read, is rebuilt whole from the
    rewritten response, so no frame the guardrail did not read reaches the client.
    """
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    adapter: Final = LiteLLMAnthropicMessagesAdapter()
    anthropic_response: Final = adapter.translate_openai_response_to_anthropic(response=assembled)
    source_events: Final = anthropic_sse_events(source_chunks)
    source: Final = _healthy_source(source_chunks, source_events, adapter)
    if source is None:
        return tuple(FakeAnthropicMessagesStreamIterator(response=anthropic_response).chunks)
    stop_reason: Final = anthropic_response.get("stop_reason")
    stop_reason_patch: Final = {"stop_reason": stop_reason} if stop_reason != source.get("stop_reason") else {}
    diff: Final = _ContentDiff.of(_content_blocks(source), _content_blocks(anthropic_response))
    segments: Final = _spliced_segments(_source_segments(source_events), diff)
    return _sse_chunks_of_segments(segments, stop_reason_patch)
