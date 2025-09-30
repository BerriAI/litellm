from typing import List, Optional, Union
import uuid

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
        litellm_completion_request: Optional[dict] = None,
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
        self.litellm_completion_request: dict = litellm_completion_request or {}
        self.collected_chat_completion_chunks: List[ModelResponseStream] = []
        self.finished: bool = False
        self.response_completed_event: Optional[ResponseCompletedEvent] = None

    def _encode_chunk_id(self, chunk_id: str) -> str:
        """
        Encode chunk ID using the same format as non-streaming responses.
        """
        model_info = self.litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id")
        return ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider=self.custom_llm_provider,
            model_id=model_id,
            response_id=chunk_id,
        )

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
                        # PATCH: Store session in Redis for streaming responses
                        await self._store_session_in_redis(response_completed_event)
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

            encoded_chunk_id = self._encode_chunk_id(chunk.id)
            return ReasoningSummaryTextDeltaEvent(
                type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id=f"{encoded_chunk_id}_reasoning",
                output_index=0,
                delta=reasoning_content,
            )
        else:
            delta_content = self._get_delta_string_from_streaming_choices(chunk.choices)
            if delta_content:
                encoded_chunk_id = self._encode_chunk_id(chunk.id)
                return OutputTextDeltaEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                    item_id=encoded_chunk_id,
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

            responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                request_input=self.request_input,
                chat_completion_response=litellm_model_response,
                responses_api_request=self.responses_api_request,
            )
            
            # PATCH: Store session immediately in Redis for streaming responses
            # This ensures the session is available for subsequent requests
            if responses_api_response.id:
                # Store the completed event to be used in async context
                self.response_completed_event = ResponseCompletedEvent(
                    type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                    response=responses_api_response,
                )
                return self.response_completed_event
            
            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=responses_api_response,
            )
        else:
            return None
    
    async def _store_session_in_redis(self, response_completed_event: ResponseCompletedEvent):
        """
        PATCH: Store session in Redis for streaming responses
        This fixes the issue where Redis sessions weren't created for streaming requests
        """
        try:
            response = response_completed_event.response
            if response and response.id:
                # Get the session ID from metadata or from the completion request
                session_id = (self.litellm_completion_request.get("litellm_trace_id") or 
                             self.litellm_metadata.get("litellm_trace_id") or 
                             str(uuid.uuid4()))
                
                # Get the full messages from the completion request (includes history)
                messages = self.litellm_completion_request.get("messages", []).copy()
                
                # Add the assistant response to the messages
                if response.output and len(response.output) > 0:
                    output_item = response.output[0]
                    if output_item.content and len(output_item.content) > 0:
                        content_item = output_item.content[0]
                        if hasattr(content_item, "text"):
                            messages.append({"role": "assistant", "content": content_item.text})
                
                # Store session in Redis
                await LiteLLMCompletionResponsesConfig._patch_store_session_in_redis(
                    response_id=response.id,
                    session_id=session_id,
                    messages=messages
                )
        except Exception:
            # Silently fail - Redis storage is a patch for timing issues
            # and shouldn't break the streaming response
            pass
