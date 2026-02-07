import asyncio
import json
from datetime import datetime
from typing import List, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import StandardPassThroughResponseObject

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .success_handler import PassThroughEndpointLogging


class AnthropicStreamValidator:
    """
    Validates Anthropic SSE streams to detect errors and incomplete streams.

    Anthropic streaming format:
    - message_start: Beginning of response
    - content_block_start/delta/stop: Content chunks
    - message_delta: Final usage/stop_reason
    - message_stop: Proper termination
    - error: Error event (should raise to client)
    """

    def __init__(self):
        self.received_message_start = False
        self.received_message_stop = False
        self.received_error = False
        self.error_message: Optional[str] = None
        self.error_type: Optional[str] = None

    def process_chunk(self, chunk: bytes) -> Tuple[bool, Optional[str]]:
        """
        Process a chunk and check for errors or termination events.

        Args:
            chunk: Raw bytes from the stream

        Returns:
            Tuple of (is_error, error_message):
            - (True, error_message) if an error event was detected
            - (False, None) otherwise
        """
        try:
            chunk_str = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return (False, None)

        # Parse SSE events in the chunk
        # SSE format: "event: <type>\ndata: <json>\n\n"
        events = self._parse_sse_events(chunk_str)

        for event_type, event_data in events:
            if event_type == "message_start":
                self.received_message_start = True
            elif event_type == "message_stop":
                self.received_message_stop = True
            elif event_type == "error":
                self.received_error = True
                # Parse error details from the data
                if event_data:
                    try:
                        data_json = json.loads(event_data)
                        error_info = data_json.get("error", {})
                        self.error_type = error_info.get("type", "unknown_error")
                        self.error_message = error_info.get(
                            "message", "Unknown error from Anthropic"
                        )
                    except json.JSONDecodeError:
                        self.error_message = "Error parsing Anthropic error response"
                        self.error_type = "parse_error"
                else:
                    self.error_message = "Unknown error from Anthropic"
                    self.error_type = "unknown_error"

                return (True, self.error_message)

        return (False, None)

    def _parse_sse_events(
        self, chunk_str: str
    ) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Parse SSE events from a chunk string.

        Returns:
            List of (event_type, data) tuples
        """
        events = []
        current_event_type: Optional[str] = None
        current_data: Optional[str] = None

        for line in chunk_str.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                current_event_type = line[6:].strip()
            elif line.startswith("data:"):
                current_data = line[5:].strip()
            elif line == "" and (
                current_event_type is not None or current_data is not None
            ):
                # Empty line marks end of an event
                events.append((current_event_type, current_data))
                current_event_type = None
                current_data = None

        # Handle case where chunk doesn't end with double newline
        if current_event_type is not None or current_data is not None:
            events.append((current_event_type, current_data))

        return events

    def is_stream_complete(self) -> bool:
        """Check if the stream completed properly with message_stop."""
        return self.received_message_stop

    def get_incomplete_stream_error(self) -> bytes:
        """
        Generate an error event for an incomplete stream.

        Returns:
            SSE-formatted error event as bytes
        """
        error_data = {
            "type": "error",
            "error": {
                "type": "incomplete_stream_error",
                "message": "Stream ended unexpectedly without receiving message_stop event. "
                "The response may be incomplete.",
            },
        }
        return f"event: error\ndata: {json.dumps(error_data)}\n\n".encode("utf-8")


class PassThroughStreamingHandler:
    @staticmethod
    async def chunk_processor(
        response: httpx.Response,
        request_body: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        endpoint_type: EndpointType,
        start_time: datetime,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
    ):
        """
        - Yields chunks from the response
        - Collect non-empty chunks for post-processing (logging)
        - Inject cost into chunks if include_cost_in_streaming_usage is enabled
        - For Anthropic streams: detect error events and incomplete streams

        For Anthropic endpoint type:
        - Parses SSE events to detect 'error' events and yields them to the client
        - Tracks whether 'message_stop' was received for proper stream termination
        - If stream ends without 'message_stop', yields an error event to the client
        """
        try:
            raw_bytes: List[bytes] = []
            # Extract model name for cost injection
            model_name = PassThroughStreamingHandler._extract_model_for_cost_injection(
                request_body=request_body,
                url_route=url_route,
                endpoint_type=endpoint_type,
                litellm_logging_obj=litellm_logging_obj,
            )

            # Initialize stream validator for Anthropic endpoints
            # Also validate Vertex AI streams that use Anthropic format (streamRawPredict/rawPredict)
            should_validate_anthropic_stream = (
                endpoint_type == EndpointType.ANTHROPIC
                or (
                    endpoint_type == EndpointType.VERTEX_AI
                    and ("streamRawPredict" in url_route or "rawPredict" in url_route)
                )
            )
            stream_validator: Optional[AnthropicStreamValidator] = None
            if should_validate_anthropic_stream:
                stream_validator = AnthropicStreamValidator()

            async for chunk in response.aiter_bytes():
                raw_bytes.append(chunk)

                # Validate Anthropic stream chunks for errors
                if stream_validator is not None:
                    is_error, error_message = stream_validator.process_chunk(chunk)
                    if is_error:
                        verbose_proxy_logger.warning(
                            "Anthropic stream error detected: %s", error_message
                        )
                        # The error event is already in the chunk, so we yield it
                        # and let the client handle it

                if (
                    getattr(litellm, "include_cost_in_streaming_usage", False)
                    and model_name
                ):
                    if endpoint_type == EndpointType.VERTEX_AI:
                        # Only handle streamRawPredict (uses Anthropic format)
                        if "streamRawPredict" in url_route or "rawPredict" in url_route:
                            modified_chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                                chunk, model_name
                            )
                            if modified_chunk is not None:
                                chunk = modified_chunk

                yield chunk

            # Check if Anthropic stream ended without proper termination
            if (
                stream_validator is not None
                and stream_validator.received_message_start
                and not stream_validator.is_stream_complete()
                and not stream_validator.received_error
            ):
                # Stream started but didn't complete properly - yield error event
                verbose_proxy_logger.warning(
                    "Anthropic stream ended without message_stop event. "
                    "received_message_start=%s, received_message_stop=%s",
                    stream_validator.received_message_start,
                    stream_validator.received_message_stop,
                )
                error_chunk = stream_validator.get_incomplete_stream_error()
                raw_bytes.append(error_chunk)
                yield error_chunk

            # After all chunks are processed, handle post-processing
            end_time = datetime.now()

            asyncio.create_task(
                PassThroughStreamingHandler._route_streaming_logging_to_handler(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body or {},
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    raw_bytes=raw_bytes,
                    end_time=end_time,
                )
            )
        except Exception as e:
            verbose_proxy_logger.error(f"Error in chunk_processor: {str(e)}")
            raise

    @staticmethod
    async def _route_streaming_logging_to_handler(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        raw_bytes: List[bytes],
        end_time: datetime,
        model: Optional[str] = None,
    ):
        """
        Route the logging for the collected chunks to the appropriate handler

        Supported endpoint types:
        - Anthropic
        - Vertex AI
        - OpenAI

        Note: This method is called from a background task (asyncio.create_task).
        Errors are caught and logged to prevent silent failures in the logging pipeline.
        """
        try:
            all_chunks = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(
                raw_bytes
            )
            standard_logging_response_object: Optional[
                PassThroughEndpointLoggingResultValues
            ] = None
            kwargs: dict = {}
            if endpoint_type == EndpointType.ANTHROPIC:
                anthropic_passthrough_logging_handler_result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
                standard_logging_response_object = (
                    anthropic_passthrough_logging_handler_result["result"]
                )
                kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
            elif endpoint_type == EndpointType.VERTEX_AI:
                vertex_passthrough_logging_handler_result = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                    model=model,
                )
                standard_logging_response_object = (
                    vertex_passthrough_logging_handler_result["result"]
                )
                kwargs = vertex_passthrough_logging_handler_result["kwargs"]
            elif endpoint_type == EndpointType.OPENAI:
                openai_passthrough_logging_handler_result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
                standard_logging_response_object = (
                    openai_passthrough_logging_handler_result["result"]
                )
                kwargs = openai_passthrough_logging_handler_result["kwargs"]

            if standard_logging_response_object is None:
                standard_logging_response_object = StandardPassThroughResponseObject(
                    response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
                )
            await litellm_logging_obj.async_success_handler(
                result=standard_logging_response_object,
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )
            if (
                litellm_logging_obj._should_run_sync_callbacks_for_async_calls()
                is False
            ):
                return

            executor.submit(
                litellm_logging_obj.success_handler,
                result=standard_logging_response_object,
                end_time=end_time,
                cache_hit=False,
                start_time=start_time,
                **kwargs,
            )
        except Exception as e:
            # Log errors in the logging pipeline but don't propagate them
            # This method runs in a background task - unhandled exceptions would be silently dropped
            verbose_proxy_logger.exception(
                "Error in passthrough streaming logging handler for endpoint_type=%s, url_route=%s: %s",
                endpoint_type,
                url_route,
                str(e),
            )

    @staticmethod
    def _extract_model_for_cost_injection(
        request_body: Optional[dict],
        url_route: str,
        endpoint_type: EndpointType,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Optional[str]:
        """
        Extract model name for cost injection from various sources.
        """
        # Try to get model from request body
        if request_body:
            model = request_body.get("model")
            if model:
                return model

        # Try to get model from logging object
        if hasattr(litellm_logging_obj, "model_call_details"):
            model = litellm_logging_obj.model_call_details.get("model")
            if model:
                return model

        # For Vertex AI, try to extract from URL
        if endpoint_type == EndpointType.VERTEX_AI:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            if model and model != "unknown":
                return model

        return None

    @staticmethod
    def _convert_raw_bytes_to_str_lines(raw_bytes: List[bytes]) -> List[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        # Combine all bytes and decode to string
        combined_str = b"".join(raw_bytes).decode("utf-8")

        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines
