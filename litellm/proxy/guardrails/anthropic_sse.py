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

from pydantic import TypeAdapter, ValidationError

from litellm.llms.anthropic.pass_through.stream_assembly import anthropic_message_delta_fields
from litellm.llms.anthropic.pass_through.stream_assembly import (
    assemble_anthropic_sse_stream as assemble_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_anthropic_sse_stream as is_anthropic_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.llms.anthropic.pass_through.stream_assembly import (
    is_raw_sse_stream as is_raw_sse_stream,  # noqa: PLC0414  # public re-export
)
from litellm.types.llms.anthropic import (
    AnthropicStopDetails,
    SafeguardResult,
    SafeguardStatusAvailable,
    SafeguardStatusUnsupported,
    SafeguardToolUseEvaluated,
    SafeguardToolUseVerdict,
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


from litellm.llms.anthropic.pass_through.stream_assembly import (  # noqa: E402  # re-export after the definitions above
    is_sse_error_stream as is_sse_error_stream,  # noqa: PLC0414  # public re-export
)

_STOP_DETAILS: Final = TypeAdapter(AnthropicStopDetails)
_SAFEGUARD_RESULTS: Final = TypeAdapter(tuple[SafeguardResult, ...])


def _stop_details_without_explanation(raw: object) -> AnthropicStopDetails | None:
    try:
        stop_details: Final = _STOP_DETAILS.validate_python(raw)
    except ValidationError:
        return None
    return {**stop_details, "explanation": None}


def _verdict_without_explanation(verdict: SafeguardToolUseVerdict) -> SafeguardToolUseVerdict:
    match verdict:
        case {"type": "evaluated"}:
            return SafeguardToolUseEvaluated(type="evaluated", outcome=verdict["outcome"])
        case _:
            return verdict


def _status_for_delivered_tool_uses(
    status: SafeguardStatusAvailable | SafeguardStatusUnsupported, delivered_tool_use_ids: frozenset[str]
) -> SafeguardStatusAvailable | SafeguardStatusUnsupported:
    match status:
        case {"type": "available"}:
            return SafeguardStatusAvailable(
                type="available",
                tool_uses={
                    tool_use_id: _verdict_without_explanation(verdict)
                    for tool_use_id, verdict in status["tool_uses"].items()
                    if tool_use_id in delivered_tool_use_ids
                },
            )
        case _:
            return status


def _safeguard_results_for_delivered_tool_uses(
    raw: object, delivered_tool_use_ids: frozenset[str]
) -> tuple[SafeguardResult, ...] | None:
    try:
        results: Final = _SAFEGUARD_RESULTS.validate_python(raw)
    except ValidationError:
        return None
    return tuple(
        SafeguardResult(
            type=result["type"], status=_status_for_delivered_tool_uses(result["status"], delivered_tool_use_ids)
        )
        for result in results
    )


def _tool_use_id(block: object) -> str | None:
    match block:
        case {"type": "tool_use", "id": str() as tool_use_id}:
            return tool_use_id
        case _:
            return None


def _rewritten_message_delta_fields(
    source_chunks: Sequence[object], delivered_content: Sequence[object]
) -> Mapping[str, object]:
    source_fields: Final = anthropic_message_delta_fields(source_chunks)
    delivered_tool_use_ids: Final = frozenset(
        tool_use_id for block in delivered_content if (tool_use_id := _tool_use_id(block)) is not None
    )
    stop_details: Final = _stop_details_without_explanation(source_fields.get("stop_details"))
    safeguard_results: Final = _safeguard_results_for_delivered_tool_uses(
        source_fields.get("safeguard_results"), delivered_tool_use_ids
    )
    return MappingProxyType(
        {
            **({"stop_details": stop_details} if stop_details is not None else {}),
            **({"safeguard_results": safeguard_results} if safeguard_results is not None else {}),
        }
    )


def anthropic_sse_chunks_from_response(assembled: ModelResponse, source_chunks: Sequence[object]) -> tuple[bytes, ...]:
    """Re-emit a rewritten response as the SSE events of the ``source_chunks`` stream it replaces."""
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    anthropic_response: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=assembled
    )
    return tuple(
        FakeAnthropicMessagesStreamIterator(
            response=anthropic_response,
            message_delta_fields=_rewritten_message_delta_fields(
                source_chunks, anthropic_response.get("content") or ()
            ),
        ).chunks
    )
