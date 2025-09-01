from typing import List, Optional, Union

import litellm
from litellm.main import stream_chunk_builder
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ReasoningSummaryTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.utils import Delta as ChatCompletionDelta
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
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
        custom_llm_provider: Optional[str] = None,
        litellm_metadata: Optional[dict] = None,
    ):
        self.litellm_custom_stream_wrapper: litellm.CustomStreamWrapper = (
            litellm_custom_stream_wrapper
        )
        self.request_input: Union[str, ResponseInputParam] = request_input
        self.responses_api_request: ResponsesAPIOptionalRequestParams = (
            responses_api_request
        )
        self.custom_llm_provider: Optional[str] = custom_llm_provider
        self.litellm_metadata: Optional[dict] = litellm_metadata or {}
        self.collected_chat_completion_chunks: List[ModelResponseStream] = []
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
                    self.collected_chat_completion_chunks.append(chunk)
                    response_api_chunk = (
                        self._transform_chat_completion_chunk_to_response_api_chunk(
                            chunk
                        )
                    )
                    if response_api_chunk:
                        return response_api_chunk
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
                    raise StopIteration
                # Get the next chunk from the stream
                try:
                    chunk = self.litellm_custom_stream_wrapper.__next__()
                    self.collected_chat_completion_chunks.append(chunk)
                    response_api_chunk = (
                        self._transform_chat_completion_chunk_to_response_api_chunk(
                            chunk
                        )
                    )
                    if response_api_chunk:
                        return response_api_chunk
                except StopIteration:
                    self.finished = True
                    response_completed_event = self._emit_response_completed_event()
                    if response_completed_event:
                        return response_completed_event
                    else:
                        raise StopIteration

        except Exception as e:
            # Handle HTTP errors
            self.finished = True
            raise e

    def _transform_chat_completion_chunk_to_response_api_chunk(
        self, chunk: ModelResponseStream
    ) -> Optional[ResponsesAPIStreamingResponse]:
        """
        Transform a chat completion chunk to a response API chunk.

        This currently handles emitting the OutputTextDeltaEvent, which is used by other tools using the responses API
        and the ReasoningSummaryTextDeltaEvent, which is used by the responses API to emit reasoning content.
        """
        if (
            chunk.choices
            and hasattr(chunk.choices[0].delta, "reasoning_content")
            and chunk.choices[0].delta.reasoning_content
        ):
            reasoning_content = chunk.choices[0].delta.reasoning_content

            return ReasoningSummaryTextDeltaEvent(
                type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id=f"{chunk.id}_reasoning",
                output_index=0,
                delta=reasoning_content,
            )
        else:
            delta_content = self._get_delta_string_from_streaming_choices(chunk.choices)
            if delta_content:
                return OutputTextDeltaEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                    item_id=chunk.id,
                    output_index=0,
                    content_index=0,
                    delta=delta_content,
                )

        return None

    def _get_delta_string_from_streaming_choices(
        self, choices: List[StreamingChoices]
    ) -> str:
        """
        Get the delta string from the streaming choices

        For now this collected the first choice's delta string.

        It's unclear how users expect litellm to translate multiple-choices-per-chunk to the responses API output.
        """
        choice = choices[0]
        chat_completion_delta: ChatCompletionDelta = choice.delta
        return chat_completion_delta.content or ""

    def _emit_response_completed_event(self) -> Optional[ResponseCompletedEvent]:
        litellm_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = stream_chunk_builder(chunks=self.collected_chat_completion_chunks)
        if litellm_model_response and isinstance(litellm_model_response, ModelResponse):
            # Transform the response
            responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input=self.request_input,
                chat_completion_response=litellm_model_response,
                responses_api_request=self.responses_api_request,
            )
            
            # Encode the response ID to match non-streaming behavior
            encoded_response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=responses_api_response,
                custom_llm_provider=self.custom_llm_provider,
                litellm_metadata=self.litellm_metadata,
            )

            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=encoded_response,
            )
        else:
            return None
