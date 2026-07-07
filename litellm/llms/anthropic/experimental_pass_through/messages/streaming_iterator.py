import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator, List, Protocol, Union, runtime_checkable

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ = PassThroughEndpointLogging()

INCOMPLETE_STREAM_ERROR_MESSAGE = (
    "Provider stream ended before emitting a message_stop event; "
    "the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated."
)


def _is_message_stop_chunk(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return chunk.get("type") == "message_stop"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: message_stop" for line in chunk.splitlines())
    return False


def _is_provider_error_chunk(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return chunk.get("type") == "error"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: error" for line in chunk.splitlines())
    return False


def _is_terminal_stream_chunk(chunk: object) -> bool:
    return _is_message_stop_chunk(chunk) or _is_provider_error_chunk(chunk)


def _incomplete_stream_error_sse_event() -> bytes:
    payload = json.dumps(
        {
            "type": "error",
            "error": {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE},
        }
    )
    return f"event: error\ndata: {payload}\n\n".encode()


class AnthropicMessagesStreamHiddenParams(TypedDict):
    additional_headers: dict[str, str]


@runtime_checkable
class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


async def aclose_if_supported(stream: object) -> None:
    if isinstance(stream, SupportsAclose):
        await stream.aclose()


_RESPONSE_HEADERS_ADAPTER: TypeAdapter[dict[str, str]] = TypeAdapter(dict[str, str])


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

    async def _handle_streaming_logging(self, collected_chunks: List[bytes]):
        """Handle the logging after all chunks have been collected."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        end_time = datetime.now()
        # Set completion_start_time so TTFT is calculated from the first
        # chunk rather than falling back to end_time in async_success_handler.
        if self.completion_start_time is not None:
            self.litellm_logging_obj.completion_start_time = self.completion_start_time
            self.litellm_logging_obj.model_call_details["completion_start_time"] = self.completion_start_time
        asyncio.create_task(
            PassThroughStreamingHandler._route_streaming_logging_to_handler(
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

    def _convert_chunk_to_sse_format(self, chunk: Union[dict, Any]) -> bytes:
        """
        Convert a chunk to Server-Sent Events format.

        This method should be overridden by subclasses if they need custom
        chunk formatting logic.
        """
        if isinstance(chunk, dict):
            event_type: str = str(chunk.get("type", "message"))
            payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
            return payload.encode()
        else:
            # For non-dict chunks, return as is
            return chunk

    async def async_sse_wrapper(
        self,
        completion_stream: AsyncIterator[Union[bytes, GenericStreamingChunk, ModelResponseStream, dict]],
    ) -> AsyncIterator[bytes]:
        """
        Generic async SSE wrapper that converts streaming chunks to SSE format
        and handles logging.

        This method provides the common logic for both Anthropic and Bedrock implementations.
        """
        collected_chunks = []
        saw_terminal_event = False

        async for chunk in completion_stream:
            if self.completion_start_time is None:
                self.completion_start_time = datetime.now()
            saw_terminal_event = saw_terminal_event or _is_terminal_stream_chunk(chunk)
            encoded_chunk = self._convert_chunk_to_sse_format(chunk)
            collected_chunks.append(encoded_chunk)
            yield encoded_chunk

        if not saw_terminal_event:
            yield _incomplete_stream_error_sse_event()

        # Handle logging after all chunks are processed
        await self._handle_streaming_logging(collected_chunks)
