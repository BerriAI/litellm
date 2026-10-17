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

from pydantic import TypeAdapter

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
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

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


from litellm.llms.anthropic.pass_through.stream_assembly import (  # noqa: E402  # re-export after the definitions above
    is_sse_error_stream as is_sse_error_stream,  # noqa: PLC0414  # public re-export
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


def anthropic_sse_chunks_from_body(body: Mapping[str, object]) -> tuple[bytes, ...]:
    """Re-emit a native Anthropic Messages body as the SSE frames its client expects."""
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    typed: Final = cast("AnthropicMessagesResponse", dict(body))  # cast-ok: validating drops provider-only fields
    return tuple(FakeAnthropicMessagesStreamIterator(response=typed).chunks)
