import json
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)


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
        logging_obj: Optional[LiteLLMLoggingObj] = None,
    ):
        self.response = response
        self.model = model
        self.logging_obj = logging_obj
        self.stream_iterator = response.aiter_lines()
        self.finished = False

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
            if chunk.startswith("data: "):
                chunk = chunk[6:]  # Remove "data: " prefix

            # Handle "[DONE]" marker
            if chunk == "[DONE]":
                self.finished = True
                raise StopAsyncIteration

            try:
                # Parse the JSON chunk
                parsed_chunk = json.loads(chunk)

                # Log the chunk if logging is enabled
                if self.logging_obj:
                    self.logging_obj.post_call(
                        input="",
                        api_key="",
                        original_response=parsed_chunk,
                        additional_args={
                            "complete_streaming_chunk": parsed_chunk,
                        },
                    )

                # Format as ResponsesAPIStreamingResponse
                if isinstance(parsed_chunk, dict):
                    # If the chunk already has a 'type' field, it's already in the right format
                    if "type" in parsed_chunk:
                        return ResponsesAPIStreamingResponse(**parsed_chunk)
                    # Otherwise, wrap it as a response
                    else:
                        return ResponsesAPIStreamingResponse(
                            type="response",
                            response=ResponsesAPIResponse(**parsed_chunk),
                        )

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
