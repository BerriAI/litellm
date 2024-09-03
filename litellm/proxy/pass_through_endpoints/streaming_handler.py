import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import AsyncIterable, Dict, List, Optional, Union

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexAIIterator,
)
from litellm.types.utils import GenericStreamingChunk

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
    aiter_bytes: AsyncIterable[bytes],
    litellm_logging_obj: LiteLLMLoggingObj,
    endpoint_type: EndpointType,
    start_time: datetime,
    passthrough_success_handler_obj: PassThroughEndpointLogging,
    url_route: str,
) -> AsyncIterable[bytes]:

    iteratorClass = get_iterator_class_from_endpoint_type(endpoint_type)
    if iteratorClass is None:
        # Generic endpoint - litellm does not do any tracking / logging for this
        async for chunk in aiter_bytes:
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
        all_chunks = []
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

        complete_streaming_response: Optional[
            Union[litellm.ModelResponse, litellm.TextCompletionResponse]
        ] = litellm.stream_chunk_builder(chunks=all_chunks)
        if complete_streaming_response is None:
            complete_streaming_response = litellm.ModelResponse()
        end_time = datetime.now()

        if passthrough_success_handler_obj.is_vertex_route(url_route):
            _model = passthrough_success_handler_obj.extract_model_from_url(url_route)
            complete_streaming_response.model = _model
            litellm_logging_obj.model = _model
            litellm_logging_obj.model_call_details["model"] = _model

        asyncio.create_task(
            litellm_logging_obj.async_success_handler(
                result=complete_streaming_response,
                start_time=start_time,
                end_time=end_time,
            )
        )
