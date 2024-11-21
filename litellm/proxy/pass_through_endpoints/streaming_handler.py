import asyncio
import json
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
from litellm.types.utils import GenericStreamingChunk

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .success_handler import PassThroughEndpointLogging
from .types import EndpointType


def get_litellm_chunk(
    model_iterator: VertexAIIterator,
    custom_stream_wrapper: litellm.utils.CustomStreamWrapper,
    chunk_dict: Dict,
) -> Optional[Dict]:

    generic_chunk: GenericStreamingChunk = model_iterator.chunk_parser(chunk_dict)
    if generic_chunk:
        return custom_stream_wrapper.chunk_creator(chunk=generic_chunk)
    return None


def get_iterator_class_from_endpoint_type(
    endpoint_type: EndpointType,
) -> Optional[type]:
    if endpoint_type == EndpointType.VERTEX_AI:
        return VertexAIIterator
    return None


async def chunk_processor(
    response: httpx.Response,
    request_body: Optional[dict],
    litellm_logging_obj: LiteLLMLoggingObj,
    endpoint_type: EndpointType,
    start_time: datetime,
    passthrough_success_handler_obj: PassThroughEndpointLogging,
    url_route: str,
) -> AsyncIterable[Union[str, bytes]]:
    request_body = request_body or {}
    iteratorClass = get_iterator_class_from_endpoint_type(endpoint_type)
    aiter_bytes = response.aiter_bytes()
    aiter_lines = response.aiter_lines()
    all_chunks = []
    if iteratorClass is None:
        # Generic endpoint - litellm does not do any tracking / logging for this
        async for chunk in aiter_lines:
            yield chunk
    elif endpoint_type == EndpointType.ANTHROPIC:
        anthropic_iterator = AnthropicIterator(
            sync_stream=False,
            streaming_response=aiter_lines,
            json_mode=False,
        )
        custom_stream_wrapper = litellm.utils.CustomStreamWrapper(
            completion_stream=aiter_bytes,
            model=None,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="anthropic",
        )
        async for chunk in aiter_lines:
            try:
                generic_chunk = anthropic_iterator.convert_str_chunk_to_generic_chunk(
                    chunk
                )
                litellm_chunk = custom_stream_wrapper.chunk_creator(chunk=generic_chunk)
                if litellm_chunk:
                    all_chunks.append(litellm_chunk)
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error parsing chunk: {e},\nReceived chunk: {chunk}"
                )
            finally:
                yield chunk
    else:
        # known streaming endpoint - litellm will do tracking / logging for this
        model_iterator = iteratorClass(
            sync_stream=False, streaming_response=aiter_bytes
        )
        custom_stream_wrapper = litellm.utils.CustomStreamWrapper(
            completion_stream=aiter_bytes, model=None, logging_obj=litellm_logging_obj
        )
        buffer = b""
        async for chunk in aiter_bytes:
            buffer += chunk
            try:
                _decoded_chunk = chunk.decode("utf-8")
                _chunk_dict = json.loads(_decoded_chunk)
                litellm_chunk = get_litellm_chunk(
                    model_iterator, custom_stream_wrapper, _chunk_dict
                )
                if litellm_chunk:
                    all_chunks.append(litellm_chunk)
            except json.JSONDecodeError:
                pass
            finally:
                yield chunk  # Yield the original bytes

        # Process any remaining data in the buffer
        if buffer:
            try:
                _chunk_dict = json.loads(buffer.decode("utf-8"))

                if isinstance(_chunk_dict, list):
                    for _chunk in _chunk_dict:
                        litellm_chunk = get_litellm_chunk(
                            model_iterator, custom_stream_wrapper, _chunk
                        )
                        if litellm_chunk:
                            all_chunks.append(litellm_chunk)
                elif isinstance(_chunk_dict, dict):
                    litellm_chunk = get_litellm_chunk(
                        model_iterator, custom_stream_wrapper, _chunk_dict
                    )
                    if litellm_chunk:
                        all_chunks.append(litellm_chunk)
            except json.JSONDecodeError:
                pass

    await _handle_logging_collected_chunks(
        litellm_logging_obj=litellm_logging_obj,
        passthrough_success_handler_obj=passthrough_success_handler_obj,
        url_route=url_route,
        request_body=request_body,
        endpoint_type=endpoint_type,
        start_time=start_time,
        end_time=datetime.now(),
        all_chunks=all_chunks,
    )


async def _handle_logging_collected_chunks(
    litellm_logging_obj: LiteLLMLoggingObj,
    passthrough_success_handler_obj: PassThroughEndpointLogging,
    url_route: str,
    request_body: dict,
    endpoint_type: EndpointType,
    start_time: datetime,
    all_chunks: List[Dict],
    end_time: datetime,
):
    """
    Build the complete response and handle the logging

    This gets triggered once all the chunks are collected
    """
    try:
        complete_streaming_response: Optional[
            Union[litellm.ModelResponse, litellm.TextCompletionResponse]
        ] = litellm.stream_chunk_builder(chunks=all_chunks)
        if complete_streaming_response is None:
            complete_streaming_response = litellm.ModelResponse()
        end_time = datetime.now()
        verbose_proxy_logger.debug(
            "complete_streaming_response %s", complete_streaming_response
        )
        kwargs = {}

        if passthrough_success_handler_obj.is_vertex_route(url_route):
            _model = passthrough_success_handler_obj.extract_model_from_url(url_route)
            complete_streaming_response.model = _model
            litellm_logging_obj.model = _model
            litellm_logging_obj.model_call_details["model"] = _model
        elif endpoint_type == EndpointType.ANTHROPIC:
            model = request_body.get("model", "")
            kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
                litellm_model_response=complete_streaming_response,
                model=model,
                kwargs=litellm_logging_obj.model_call_details,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
            )
            litellm_logging_obj.model = model
            complete_streaming_response.model = model
            litellm_logging_obj.model_call_details["model"] = model
        # Remove start_time and end_time from kwargs since they'll be passed explicitly
        kwargs.pop("start_time", None)
        kwargs.pop("end_time", None)
        litellm_logging_obj.model_call_details.update(kwargs)

        asyncio.create_task(
            litellm_logging_obj.async_success_handler(
                result=complete_streaming_response,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Error handling logging collected chunks: {e}")
