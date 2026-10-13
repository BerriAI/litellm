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

from litellm.llms.anthropic.pass_through.stream_assembly import (
    _joined_sse_stream,  # pyright: ignore[reportPrivateUsage]  # moved with the assembler; a copy splits trailing utf-8
    _parsed_sse_events,  # pyright: ignore[reportPrivateUsage]  # same parser assemble_anthropic_sse_stream uses
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream as assemble_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_anthropic_sse_stream as is_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_raw_sse_stream as is_raw_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_sse_error_stream as is_sse_error_stream,  # noqa: PLC0414  # public re-export
)
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})

_TOOL_BLOCK_TYPES: Final = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})


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


def anthropic_sse_chunks_from_response(assembled: ModelResponse) -> tuple[bytes, ...]:
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
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

def anthropic_sse_chunks_from_body(body: Mapping[str, object]) -> tuple[bytes, ...]:
    """Re-emit a native Anthropic Messages body as the SSE frames its client expects."""
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    typed: Final = cast("AnthropicMessagesResponse", dict(body))
    return tuple(FakeAnthropicMessagesStreamIterator(response=typed).chunks)
