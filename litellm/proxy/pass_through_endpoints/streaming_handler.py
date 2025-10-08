import asyncio
from datetime import datetime
from typing import List, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
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
        """
        try:
            raw_bytes: List[bytes] = []
            async for chunk in response.aiter_bytes():
                raw_bytes.append(chunk)
                yield chunk

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
        """
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
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
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
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif endpoint_type == EndpointType.OPENAI:
            openai_passthrough_logging_handler_result = (
                OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                    litellm_logging_obj=litellm_logging_obj,
                    passthrough_success_handler_obj=passthrough_success_handler_obj,
                    url_route=url_route,
                    request_body=request_body,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    all_chunks=all_chunks,
                    end_time=end_time,
                )
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
        if litellm_logging_obj._should_run_sync_callbacks_for_async_calls() is False:
            return

        executor.submit(
            litellm_logging_obj.success_handler,
            result=standard_logging_response_object,
            end_time=end_time,
            cache_hit=False,
            start_time=start_time,
            **kwargs,
        )

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
