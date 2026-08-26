import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.common_utils import ANTHROPIC_ERROR_STATUS_CODE_MAP
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ: Final = PassThroughEndpointLogging()

INCOMPLETE_STREAM_ERROR_MESSAGE: Final = (
    "Provider stream ended before emitting a message_stop event; "
    "the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated."
)


def _is_message_stop_chunk(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return chunk.get("type") == "message_stop"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: message_stop" for line in chunk.splitlines())
    return False


def is_anthropic_ping_chunk(chunk: object) -> bool:
    """
    Whether a chunk is a pure ``ping`` keepalive frame. It carries no content
    and can recur indefinitely on a slow-starting or idle connection, so a
    mid-stream fallback wrapper drops it outright while still deciding
    whether to commit to the primary stream, rather than buffering it.

    A physical transport chunk that coalesces a ping with any other SSE
    event (``message_start``, ``content_block_delta``, ``event: error``, ...)
    is NOT a pure ping - dropping it whole would discard those events - so
    only a chunk whose every ``event:`` line is ``event: ping`` qualifies.
    """
    if isinstance(chunk, dict):
        return chunk.get("type") == "ping"
    if isinstance(chunk, (bytes, bytearray)):
        event_lines: Final = tuple(line for line in chunk.splitlines() if line.startswith(b"event:"))
        return bool(event_lines) and all(line == b"event: ping" for line in event_lines)
    return False


def is_anthropic_content_delta_chunk(chunk: object) -> bool:
    """
    Whether a chunk carries actual assistant-generated output (a
    ``content_block_delta`` frame), as opposed to a lifecycle/bookkeeping
    frame (``message_start``, ``content_block_start``/``stop``,
    ``message_delta``, ``message_stop``, ``ping``) that carries nothing
    worth preserving before an invisible mid-stream fallback retry.
    """
    if isinstance(chunk, dict):
        return chunk.get("type") == "content_block_delta"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: content_block_delta" for line in chunk.splitlines())
    return False


def _decoded_sse_data_line(line: bytes) -> object | None:
    if not line.startswith(b"data:"):
        return None
    try:
        return json.loads(line[len(b"data:") :].strip())
    except (ValueError, TypeError):
        return None


def _anthropic_error_event_payload(chunk: object) -> Mapping[str, object] | None:
    if isinstance(chunk, dict):
        return chunk if chunk.get("type") == "error" else None
    if isinstance(chunk, (bytes, bytearray)):
        decoded_lines: Final = (_decoded_sse_data_line(line) for line in chunk.splitlines())
        return next(
            (
                candidate
                for candidate in decoded_lines
                if isinstance(candidate, dict) and candidate.get("type") == "error"
            ),
            None,
        )
    return None


def _anthropic_error_body(chunk: object) -> Mapping[str, object] | None:
    """Return the ``error`` object of an Anthropic SSE ``event: error`` chunk, or None."""
    payload: Final = _anthropic_error_event_payload(chunk)
    error_body: Final = payload.get("error") if payload is not None else None
    return error_body if isinstance(error_body, dict) else None


def _is_provider_error_chunk(chunk: object) -> bool:
    return _anthropic_error_body(chunk) is not None


def parse_anthropic_error_event(chunk: object) -> tuple[str, str, int] | None:
    """
    Extract ``(error_type, message, http_status_code)`` from an Anthropic SSE
    ``event: error`` chunk (raw bytes or an already-decoded dict), or None if
    ``chunk`` is not an error event.

    The status code is looked up via ANTHROPIC_ERROR_STATUS_CODE_MAP,
    defaulting to 500 for an error ``type`` Anthropic hasn't documented yet.
    """
    error_body: Final = _anthropic_error_body(chunk)
    if error_body is None:
        return None
    error_type: Final = error_body.get("type")
    if not isinstance(error_type, str):
        return None
    message: Final = error_body.get("message")
    return (
        error_type,
        message if isinstance(message, str) else error_type,
        ANTHROPIC_ERROR_STATUS_CODE_MAP.get(error_type, 500),
    )


def _is_terminal_stream_chunk(chunk: object) -> bool:
    return _is_message_stop_chunk(chunk) or _is_provider_error_chunk(chunk)


def _sse_event(event_type: str, payload: Mapping[str, object]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _incomplete_stream_error_sse_event() -> bytes:
    return _sse_event(  # mutable-ok: one-shot JSON payload, never mutated after construction
        "error",
        {"type": "error", "error": {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE}},
    )


def _anthropic_content_block_start_and_deltas(
    block: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """
    ``(content_block_start.content_block, content_block_delta.delta events)``
    for one Anthropic response content block. A thinking block emits both a
    thinking_delta and a trailing signature_delta - a real Anthropic stream
    does the same, and dropping the signature makes any replay of that
    assistant message (a follow-up turn, a tool-use continuation) fail
    Anthropic's thinking-signature verification. redacted_thinking has no
    delta at all - it is sent complete in content_block_start.
    """
    match block.get("type"):
        case "tool_use":
            return (
                {  # mutable-ok: one-shot payload
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},  # mutable-ok: one-shot payload
                    "type": "tool_use",
                },
                (
                    {  # mutable-ok: one-shot payload
                        "partial_json": json.dumps(block.get("input") or {}),  # mutable-ok: one-shot payload
                        "type": "input_json_delta",
                    },
                ),
            )
        case "thinking":
            signature: Final = block.get("signature")
            signature_deltas: Final = (
                ({"signature": signature, "type": "signature_delta"},)  # mutable-ok: one-shot payload
                if isinstance(signature, str) and signature
                else ()
            )
            return (
                {"thinking": "", "signature": "", "type": "thinking"},  # mutable-ok: one-shot payload
                (
                    {"thinking": block.get("thinking") or "", "type": "thinking_delta"},  # mutable-ok: one-shot payload
                    *signature_deltas,
                ),
            )
        case "redacted_thinking":
            return ({"type": "redacted_thinking", "data": block.get("data")}, ())  # mutable-ok: one-shot JSON payload
        case _:
            return (
                {"type": "text", "text": ""},  # mutable-ok: one-shot JSON payload
                ({"type": "text_delta", "text": block.get("text") or ""},),  # mutable-ok: one-shot JSON payload
            )


def anthropic_messages_response_as_sse_events(response: AnthropicMessagesResponse) -> tuple[bytes, ...]:
    """
    Render a complete (non-streaming) AnthropicMessagesResponse as the SSE
    event sequence a real streaming request would have produced.

    A mid-stream fallback can resolve to a non-streaming response even
    though the client asked to stream (e.g. an agentic tool-use loop that
    intercepts and returns a complete message) - yielding that dict directly
    into a `/v1/messages` SSE byte stream would produce a malformed
    response, so it's synthesized into the message_start/content_block_*/
    message_delta/message_stop lifecycle a real stream would have sent.
    """
    content_blocks: Final = response.get("content") or ()
    content_events: Final = (
        event for index, block in enumerate(content_blocks) for event in _anthropic_content_block_events(index, block)
    )
    # A real message_start always carries a null stop_reason/stop_sequence and
    # a zero output_tokens - those are only known once generation finishes, so
    # copying the completed response's final values here would let a client
    # treat the message as already finished, or double-count output tokens.
    message_start_usage: Final = {  # mutable-ok: one-shot JSON payload
        **(response.get("usage") or {}),
        "output_tokens": 0,
    }
    message_start_payload: Final = {  # mutable-ok: one-shot JSON payload, never mutated after construction
        "type": "message_start",
        "message": {  # mutable-ok: one-shot JSON payload
            **response,
            "content": [],  # mutable-ok: one-shot JSON payload
            "stop_reason": None,
            "stop_sequence": None,
            "usage": message_start_usage,
        },
    }
    message_delta_payload: Final = {  # mutable-ok: one-shot JSON payload, never mutated after construction
        "type": "message_delta",
        "delta": {  # mutable-ok: one-shot JSON payload
            "stop_reason": response.get("stop_reason"),
            "stop_sequence": response.get("stop_sequence"),
        },
        "usage": response.get("usage") or {},  # mutable-ok: one-shot JSON payload
    }
    return (
        _sse_event("message_start", message_start_payload),
        *content_events,
        _sse_event("message_delta", message_delta_payload),
        _sse_event("message_stop", {"type": "message_stop"}),  # mutable-ok: one-shot JSON payload
    )


def _anthropic_content_block_events(index: int, block: Mapping[str, object]) -> tuple[bytes, ...]:
    start_block, deltas = _anthropic_content_block_start_and_deltas(block)
    start_payload: Final = {  # mutable-ok: one-shot payload
        "type": "content_block_start",
        "index": index,
        "content_block": start_block,
    }
    stop_payload: Final = {  # mutable-ok: one-shot payload
        "type": "content_block_stop",
        "index": index,
    }
    delta_events: Final = tuple(
        _sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": delta},  # mutable-ok: one-shot payload
        )
        for delta in deltas
    )
    return (
        _sse_event("content_block_start", start_payload),
        *delta_events,
        _sse_event("content_block_stop", stop_payload),
    )


class AnthropicMessagesStreamHiddenParams(TypedDict):
    additional_headers: dict[str, str]


@runtime_checkable
class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


async def aclose_if_supported(stream: object) -> None:
    if isinstance(stream, SupportsAclose):
        await stream.aclose()


_RESPONSE_HEADERS_ADAPTER: Final[TypeAdapter[dict[str, str]]] = TypeAdapter(dict[str, str])


def anthropic_messages_stream_hidden_params(
    response_headers: httpx.Headers,
) -> AnthropicMessagesStreamHiddenParams:
    return AnthropicMessagesStreamHiddenParams(
        additional_headers=_RESPONSE_HEADERS_ADAPTER.validate_python(process_response_headers(response_headers))
    )


class AnthropicMessagesStreamingResponse:
    """
    Wraps the /v1/messages SSE byte stream so upstream provider response
    headers (e.g. Bedrock's x-amzn-requestid / x-amzn-trace-id) survive as
    ``_hidden_params["additional_headers"]``, which the proxy forwards to
    clients as ``llm_provider-*`` response headers. Bare async generators
    cannot carry attributes, so header context was previously dropped.
    """

    def __init__(
        self,
        completion_stream: AsyncIterator[bytes],
        hidden_params: AnthropicMessagesStreamHiddenParams,
    ) -> None:
        self.completion_stream = completion_stream
        self._hidden_params = hidden_params

    @property
    def has_buffered_provider_output(self) -> bool:
        return getattr(self.completion_stream, "has_buffered_provider_output", False) is True

    def __aiter__(self) -> "AnthropicMessagesStreamingResponse":
        return self

    async def __anext__(self) -> bytes:
        return await self.completion_stream.__anext__()

    async def aclose(self) -> None:
        await aclose_if_supported(self.completion_stream)


class BaseAnthropicMessagesStreamingIterator:
    """
    Base class for Anthropic Messages streaming iterators that provides common logic
    for streaming response handling and logging.
    """

    def __init__(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = datetime.now()
        self.completion_start_time: datetime | None = None

    async def _handle_streaming_logging(self, collected_chunks: list[bytes]):
        """Handle the logging after all chunks have been collected."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        end_time: Final = datetime.now()
        # Set completion_start_time so TTFT is calculated from the first
        # chunk rather than falling back to end_time in async_success_handler.
        if self.completion_start_time is not None:
            self.litellm_logging_obj.completion_start_time = self.completion_start_time
            self.litellm_logging_obj.model_call_details["completion_start_time"] = self.completion_start_time
        # Enqueue on the rooted logging worker rather than asyncio.create_task:
        # this also runs during generator teardown after a client disconnect,
        # where an unrooted task could be garbage-collected before it bills.
        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=self.litellm_logging_obj,
                passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
                url_route="/v1/messages",
                request_body=self.request_body or {},
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=self.start_time,
                raw_bytes=collected_chunks,
                end_time=end_time,
            )
        )

    def get_async_streaming_response_iterator(
        self,
        httpx_response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        # Use the existing streaming handler for Anthropic
        return PassThroughStreamingHandler.chunk_processor(
            response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=self.start_time,
            passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
            url_route="/v1/messages",
        )

    def _convert_chunk_to_sse_format(self, chunk: dict | Any) -> bytes:
        """
        Convert a chunk to Server-Sent Events format.

        This method should be overridden by subclasses if they need custom
        chunk formatting logic.
        """
        if isinstance(chunk, dict):
            event_type: Final[str] = str(chunk.get("type", "message"))
            payload: Final = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
            return payload.encode()
        else:
            # For non-dict chunks, return as is
            return chunk

    async def async_sse_wrapper(
        self,
        completion_stream: AsyncIterator[bytes | GenericStreamingChunk | ModelResponseStream | dict],
    ) -> AsyncIterator[bytes]:
        """
        Generic async SSE wrapper that converts streaming chunks to SSE format
        and handles logging.

        This method provides the common logic for both Anthropic and Bedrock implementations.
        """
        collected_chunks: Final = []
        saw_terminal_event = False

        try:
            async for chunk in completion_stream:
                if self.completion_start_time is None:
                    self.completion_start_time = datetime.now()
                saw_terminal_event = saw_terminal_event or _is_terminal_stream_chunk(chunk)
                encoded_chunk = self._convert_chunk_to_sse_format(chunk)
                collected_chunks.append(encoded_chunk)
                yield encoded_chunk
        except (GeneratorExit, asyncio.CancelledError):
            # A client disconnect tears the generator down at the yield, so the
            # post-loop logging below never runs and the tokens already streamed
            # (and billed by the provider) would never reach spend tracking. See LIT-5839.
            if collected_chunks:
                await self._handle_streaming_logging(collected_chunks)
            raise

        if not saw_terminal_event:
            yield _incomplete_stream_error_sse_event()

        # Handle logging after all chunks are processed
        await self._handle_streaming_logging(collected_chunks)
