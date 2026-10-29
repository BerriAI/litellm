"""Anthropic SSE <-> ModelResponse conversion for guardrail streaming hooks.

`/v1/messages` streams reach a guardrail's `async_post_call_streaming_iterator_hook` as raw SSE
frames rather than chunk objects, which `stream_chunk_builder` cannot assemble. These helpers let a
hook scan such a stream, and re-emit it when the guardrail rewrote the response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.messages.native.utils import filtered_safeguard_results, redacted_stop_details
from litellm.llms.anthropic.messages.stream_assembly import (
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
    is_anthropic_sse_stream as is_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.messages.stream_assembly import (
    is_raw_sse_stream as is_raw_sse_stream,  # noqa: PLC0414  # public re-export
)
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


def has_cut_off_tool_use(source_chunks: Sequence[object]) -> bool:
    """Whether a failed upstream cut off a tool call, which assembly leaves out of what it checks."""
    return any(block.get("type") == "tool_use" for block in anthropic_sse_unfinished_blocks(source_chunks).values())


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


def _sent_by_the_failed_upstream(chunk: bytes, block_left_open: int | None, end_events_sent: frozenset[object]) -> bool:
    event: Final = chunk.decode()
    event_type: Final = anthropic_sse_event_type(event)
    if event_type in _MESSAGE_END_EVENT_TYPES:
        return event_type in end_events_sent
    return block_left_open is None or anthropic_sse_stopped_block_index(event) != block_left_open


def anthropic_sse_chunks_from_response(
    assembled: ModelResponse, source_chunks: Sequence[object] = (), *, keep_explanations: bool = False
) -> tuple[bytes, ...]:
    """Re-emit a rewritten response as the SSE events of the ``source_chunks`` stream it replaces.

    ``message_delta`` carries the source stream's ``stop_details`` and ``safeguard_results`` and
    nothing the OpenAI round trip made up, with a verdict kept only for a tool call delivered with
    the input the classifier judged. Their free-text ``explanation`` was written about the text the
    guardrail rewrote, so it is dropped unless the caller rewrites no text (``keep_explanations``).
    A source stream that ended in an upstream ``error`` event ends with that event again, never
    with a ``message_delta`` or ``message_stop`` the upstream did not send, and with its cut-off
    text block left open unless another block follows it.
    """
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
        response=anthropic_response,
        message_delta_fields=_rewritten_message_delta_fields(source_chunks, unchanged_tool_use_ids, keep_explanations),
    ).chunks
    error_events: Final = anthropic_sse_error_events(source_chunks)
    if not error_events:
        return tuple(rewritten_chunks)
    block_left_open: Final = _text_block_left_open(rewritten_chunks, source_chunks)
    end_events_sent: Final = _MESSAGE_END_EVENT_TYPES & anthropic_sse_event_types(source_chunks)
    return (
        *(chunk for chunk in rewritten_chunks if _sent_by_the_failed_upstream(chunk, block_left_open, end_events_sent)),
        *(f"{event}\n\n".encode() for event in error_events),
    )
