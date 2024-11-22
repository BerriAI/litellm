import asyncio
import json
import threading
from datetime import datetime
from enum import Enum
from typing import AsyncIterable, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicIterator,
)
from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexAIIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.types.utils import (
    GenericStreamingChunk,
    ModelResponse,
    StandardPassThroughResponseObject,
)

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .success_handler import PassThroughEndpointLogging
from .types import EndpointType


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
    collected_chunks: List[str] = []  # List to store all chunks
    try:
        async for chunk in response.aiter_lines():
            verbose_proxy_logger.debug(f"Processing chunk: {chunk}")
            if not chunk:
                continue

            # Handle SSE format - pass through the raw SSE format
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")

            # Store the chunk for post-processing
            if chunk.strip():  # Only store non-empty chunks
                collected_chunks.append(chunk)
                yield f"{chunk}\n"

        # After all chunks are processed, handle post-processing
        end_time = datetime.now()

        await _route_streaming_logging_to_handler(
            litellm_logging_obj=litellm_logging_obj,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route=url_route,
            request_body=request_body or {},
            endpoint_type=endpoint_type,
            start_time=start_time,
            all_chunks=collected_chunks,
            end_time=end_time,
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error in chunk_processor: {str(e)}")
        raise


async def _route_streaming_logging_to_handler(
    litellm_logging_obj: LiteLLMLoggingObj,
    passthrough_success_handler_obj: PassThroughEndpointLogging,
    url_route: str,
    request_body: dict,
    endpoint_type: EndpointType,
    start_time: datetime,
    all_chunks: List[str],
    end_time: datetime,
):
    """
    Route the logging for the collected chunks to the appropriate handler

    Supported endpoint types:
    - Anthropic
    - Vertex AI
    """
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
        standard_logging_response_object = anthropic_passthrough_logging_handler_result[
            "result"
        ]
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
            )
        )
        standard_logging_response_object = vertex_passthrough_logging_handler_result[
            "result"
        ]
        kwargs = vertex_passthrough_logging_handler_result["kwargs"]

    if standard_logging_response_object is None:
        standard_logging_response_object = StandardPassThroughResponseObject(
            response=f"cannot parse chunks to standard response object. Chunks={all_chunks}"
        )
    threading.Thread(
        target=litellm_logging_obj.success_handler,
        args=(
            standard_logging_response_object,
            start_time,
            end_time,
            False,
        ),
    ).start()
    await litellm_logging_obj.async_success_handler(
        result=standard_logging_response_object,
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        **kwargs,
    )
