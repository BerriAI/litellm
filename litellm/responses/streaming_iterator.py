import json
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.utils import CustomStreamWrapper


class ResponsesAPIStreamingIterator:
    """
    Async iterator for processing streaming responses from the Responses API.

    This iterator handles the chunked streaming format returned by the Responses API
    and yields properly formatted ResponsesAPIStreamingResponse objects.
    """

    def __init__(
        self,
        response: httpx.Response,
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        logging_obj: LiteLLMLoggingObj,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.stream_iterator = response.aiter_lines()
        self.finished = False
        self.responses_api_provider_config = responses_api_provider_config

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        if self.finished:
            raise StopAsyncIteration

        try:
            # Get the next chunk from the stream
            try:
                chunk = await self.stream_iterator.__anext__()
            except StopAsyncIteration:
                self.finished = True
                raise StopAsyncIteration

            if not chunk:
                return await self.__anext__()

            # Handle SSE format (data: {...})
            chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if chunk is None:
                return await self.__anext__()

            # Handle "[DONE]" marker
            if chunk == "[DONE]":
                self.finished = True
                raise StopAsyncIteration

            try:
                # Parse the JSON chunk
                parsed_chunk = json.loads(chunk)

                # Format as ResponsesAPIStreamingResponse
                if isinstance(parsed_chunk, dict):
                    openai_responses_api_chunk: ResponsesAPIStreamingResponse = (
                        self.responses_api_provider_config.transform_streaming_response(
                            model=self.model,
                            parsed_chunk=parsed_chunk,
                            logging_obj=self.logging_obj,
                        )
                    )
                    return openai_responses_api_chunk

                return ResponsesAPIStreamingResponse(
                    type="response", response=parsed_chunk
                )

            except json.JSONDecodeError:
                # If we can't parse the chunk, continue to the next one
                return await self.__anext__()

        except httpx.HTTPError as e:
            # Handle HTTP errors
            self.finished = True
            raise e
