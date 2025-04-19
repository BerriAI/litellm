from typing import List, Optional, Union

import litellm
from litellm.main import stream_chunk_builder
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)


class LiteLLMCompletionStreamingIterator(ResponsesAPIStreamingIterator):
    """
    Async iterator for processing streaming responses from the Responses API.
    """

    def __init__(
        self,
        litellm_custom_stream_wrapper: litellm.CustomStreamWrapper,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
    ):
        self.litellm_custom_stream_wrapper: litellm.CustomStreamWrapper = (
            litellm_custom_stream_wrapper
        )
        self.request_input: Union[str, ResponseInputParam] = request_input
        self.responses_api_request: ResponsesAPIOptionalRequestParams = (
            responses_api_request
        )
        self.collected_chunks: List[ModelResponseStream] = []
        self.finished: bool = False

    async def __anext__(
        self,
    ) -> Union[ResponsesAPIStreamingResponse, ResponseCompletedEvent]:
        try:
            while True:
                if self.finished is True:
                    raise StopAsyncIteration
                # Get the next chunk from the stream
                try:
                    chunk = await self.litellm_custom_stream_wrapper.__anext__()
                    self.collected_chunks.append(chunk)
                except StopAsyncIteration:
                    self.finished = True
                    response_completed_event = self._emit_response_completed_event()
                    if response_completed_event:
                        return response_completed_event
                    else:
                        raise StopAsyncIteration

        except Exception as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def __iter__(self):
        return self

    def __next__(
        self,
    ) -> Union[ResponsesAPIStreamingResponse, ResponseCompletedEvent]:
        try:
            while True:
                if self.finished is True:
                    raise StopAsyncIteration
                # Get the next chunk from the stream
                try:
                    chunk = self.litellm_custom_stream_wrapper.__next__()
                    self.collected_chunks.append(chunk)
                except StopAsyncIteration:
                    self.finished = True
                    response_completed_event = self._emit_response_completed_event()
                    if response_completed_event:
                        return response_completed_event
                    else:
                        raise StopAsyncIteration

        except Exception as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _emit_response_completed_event(self) -> Optional[ResponseCompletedEvent]:
        litellm_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = stream_chunk_builder(chunks=self.collected_chunks)
        if litellm_model_response and isinstance(litellm_model_response, ModelResponse):

            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    request_input=self.request_input,
                    chat_completion_response=litellm_model_response,
                    responses_api_request=self.responses_api_request,
                ),
            )
        else:
            return None
