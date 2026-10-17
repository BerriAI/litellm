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

from litellm.litellm_core_utils.core_helpers import as_str_mapping
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

_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _stop_details(raw: object, keep_explanations: bool) -> object:
    stop_details: Final = as_str_mapping(raw)
    if keep_explanations or stop_details is None or "explanation" not in stop_details:
        return raw
    return {**stop_details, "explanation": None}


def _verdict(raw: object, keep_explanations: bool) -> object:
    verdict: Final = as_str_mapping(raw)
    if keep_explanations or verdict is None:
        return raw
    return {key: value for key, value in verdict.items() if key != "explanation"}


def _status(raw: object, unchanged_tool_use_ids: frozenset[str], keep_explanations: bool) -> object:
    status: Final = as_str_mapping(raw)
    tool_uses: Final = as_str_mapping(status.get("tool_uses")) if status is not None else None
    if status is None or tool_uses is None:
        return raw
    return {
        **status,
        "tool_uses": {
            tool_use_id: _verdict(verdict, keep_explanations)
            for tool_use_id, verdict in tool_uses.items()
            if tool_use_id in unchanged_tool_use_ids
        },
    }


def _safeguard_result(raw: object, unchanged_tool_use_ids: frozenset[str], keep_explanations: bool) -> object:
    result: Final = as_str_mapping(raw)
    if result is None or "status" not in result:
        return raw
    return {**result, "status": _status(result["status"], unchanged_tool_use_ids, keep_explanations)}


def _safeguard_results(raw: object, unchanged_tool_use_ids: frozenset[str], keep_explanations: bool) -> object:
    try:
        results: Final = _ITEMS.validate_python(raw)
    except ValidationError:
        return raw
    return tuple(_safeguard_result(result, unchanged_tool_use_ids, keep_explanations) for result in results)


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
            **({"stop_details": _stop_details(stop_details, keep_explanations)} if stop_details is not None else {}),
            **(
                {"safeguard_results": _safeguard_results(safeguard_results, unchanged_tool_use_ids, keep_explanations)}
                if safeguard_results is not None
                else {}
            ),
        }
    )


def anthropic_sse_chunks_from_response(
    assembled: ModelResponse, source_chunks: Sequence[object], *, keep_explanations: bool = False
) -> tuple[bytes, ...]:
    """Re-emit a rewritten response as the SSE events of the ``source_chunks`` stream it replaces.

    ``message_delta`` carries the source stream's ``stop_details`` and ``safeguard_results`` and
    nothing the OpenAI round trip made up, with a verdict kept only for a tool call delivered with
    the input the classifier judged. Their free-text ``explanation`` was written about the text the
    guardrail rewrote, so it is dropped unless the caller rewrites no text (``keep_explanations``).
    """
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    adapter: Final = LiteLLMAnthropicMessagesAdapter()
    anthropic_response: Final = adapter.translate_openai_response_to_anthropic(response=assembled)
    source_response: Final = assemble_anthropic_sse_stream(source_chunks)
    source_content: Final = (
        adapter.translate_openai_response_to_anthropic(response=source_response).get("content")
        if source_response is not None
        else None
    )
    unchanged_tool_use_ids: Final = _unchanged_tool_use_ids(
        source_content or (), anthropic_response.get("content") or ()
    )
    return tuple(
        FakeAnthropicMessagesStreamIterator(
            response=anthropic_response,
            message_delta_fields=_rewritten_message_delta_fields(
                source_chunks, unchanged_tool_use_ids, keep_explanations
            ),
        ).chunks
    )
