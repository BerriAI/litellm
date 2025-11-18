import time
import uuid
from typing import List, Optional, Union, cast

import litellm
from litellm.main import stream_chunk_builder
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    PART_UNION_TYPES,
    BaseLiteLLMOpenAIResponseObject,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    ContentPartDonePartOutputText,
    ContentPartDonePartReasoningText,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningSummaryTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
    OutputTextAnnotationAddedEvent
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
        model: str,
        litellm_custom_stream_wrapper: litellm.CustomStreamWrapper,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        litellm_metadata: Optional[dict] = None,
        litellm_completion_request: Optional[dict] = None,
    ):
        self.model: str = model
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
        self.litellm_logging_obj = litellm_custom_stream_wrapper.logging_obj
        self.sent_response_created_event: bool = False
        self.sent_response_in_progress_event: bool = False
        self.sent_output_item_added_event: bool = False
        self.sent_content_part_added_event: bool = False
        self.sent_output_text_done_event: bool = False
        self.sent_output_content_part_done_event: bool = False
        self.sent_output_item_done_event: bool = False
        self.sent_annotation_events: bool = False
        self.litellm_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = None
        self.final_text: str = ""

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

    def _default_response_created_event_data(self) -> dict:
        response_created_event_data = {
            "id": f"resp_{str(uuid.uuid4())}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "instructions": self.responses_api_request.get("instructions", None),
            "max_output_tokens": None,
            "model": self.model,
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": True,
        }
        if "temperature" in self.responses_api_request:
            response_created_event_data["temperature"] = self.responses_api_request[
                "temperature"
            ]
        if "text" in self.responses_api_request:
            response_created_event_data["text"] = self.responses_api_request["text"]
        if "tool_choice" in self.responses_api_request:
            response_created_event_data["tool_choice"] = self.responses_api_request[
                "tool_choice"
            ]
        else:
            response_created_event_data["tool_choice"] = "auto"
        if "tools" in self.responses_api_request:
            response_created_event_data["tools"] = self.responses_api_request["tools"]
        else:
            response_created_event_data["tools"] = []
        if "top_p" in self.responses_api_request:
            response_created_event_data["top_p"] = self.responses_api_request["top_p"]
        else:
            response_created_event_data["top_p"] = 1.0
        if "truncation" in self.responses_api_request:
            response_created_event_data["truncation"] = self.responses_api_request[
                "truncation"
            ]
        if "user" in self.responses_api_request:
            response_created_event_data["user"] = self.responses_api_request["user"]
        if "metadata" in self.responses_api_request:
            response_created_event_data["metadata"] = self.responses_api_request[
                "metadata"
            ]
        return response_created_event_data

    def create_response_created_event(self) -> ResponseCreatedEvent:
        """
        data: {"type":"response.created","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

        """
        response_created_event_data = self._default_response_created_event_data()
        return ResponseCreatedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
            response=ResponsesAPIResponse(**response_created_event_data),
        )

    def create_response_in_progress_event(self) -> ResponseInProgressEvent:
        response_in_progress_event_data = self._default_response_created_event_data()
        response_in_progress_event_data["status"] = "in_progress"
        return ResponseInProgressEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
            response=ResponsesAPIResponse(**response_in_progress_event_data),
        )

    def create_output_item_added_event(self) -> OutputItemAddedEvent:
        return OutputItemAddedEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=BaseLiteLLMOpenAIResponseObject(
                **{
                    "id": f"msg_{str(uuid.uuid4())}",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
            ),
        )

    def create_content_part_added_event(self) -> ContentPartAddedEvent:
        return ContentPartAddedEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            item_id=f"msg_{str(uuid.uuid4())}",
            output_index=0,
            content_index=0,
            part=BaseLiteLLMOpenAIResponseObject(
                **{"type": "output_text", "text": "", "annotations": []}
            ),
        )

    def create_litellm_model_response(
        self,
    ) -> Optional[ModelResponse]:
        return cast(
            Optional[ModelResponse],
            stream_chunk_builder(
                chunks=self.collected_chat_completion_chunks,
                logging_obj=self.litellm_logging_obj,
            ),
        )

    def create_output_text_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> OutputTextDoneEvent:
        return OutputTextDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            item_id=f"msg_{str(uuid.uuid4())}",
            output_index=0,
            content_index=0,
            text=getattr(litellm_complete_object.choices[0].message, "content", "")  # type: ignore
            or "",
        )

    def create_output_content_part_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> ContentPartDoneEvent:

        text = getattr(litellm_complete_object.choices[0].message, "content", "") or ""  # type: ignore
        reasoning_content = getattr(litellm_complete_object.choices[0].message, "reasoning_content", "") or ""  # type: ignore
        annotations = getattr(litellm_complete_object.choices[0].message, "annotations", None)  # type: ignore

        part: Optional[PART_UNION_TYPES] = None
        if reasoning_content:
            part = ContentPartDonePartReasoningText(
                type="reasoning_text",
                reasoning=reasoning_content,
            )

        else:
            response_annotations = LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
                annotations=annotations
            )
            part = ContentPartDonePartOutputText(
                type="output_text",
                text=text,
                annotations=response_annotations,  # type: ignore
                logprobs=None,
            )

        return ContentPartDoneEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
            item_id=f"msg_{str(uuid.uuid4())}",
            output_index=0,
            content_index=0,
            part=part,
        )

    def create_output_item_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> OutputItemDoneEvent:
        text = self.litellm_model_response.choices[0].message.content or ""  # type: ignore
        annotations = getattr(self.litellm_model_response.choices[0].message, "annotations", None)  # type: ignore

        response_annotations = LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
            annotations=annotations
        )
        return OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            sequence_number=1,
            item=BaseLiteLLMOpenAIResponseObject(
                **{
                    "id": f"msg_{str(uuid.uuid4())}",
                    "status": "completed",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": response_annotations,
                        }
                    ],
                }
            ),
        )

    def return_default_done_events(
        self, litellm_complete_object: ModelResponse
    ) -> Optional[BaseLiteLLMOpenAIResponseObject]:
        if self.sent_output_text_done_event is False:
            self.sent_output_text_done_event = True
            return self.create_output_text_done_event(litellm_complete_object)
        if self.sent_output_content_part_done_event is False:
            self.sent_output_content_part_done_event = True
            return self.create_output_content_part_done_event(litellm_complete_object)
        if self.sent_output_item_done_event is False:
            self.sent_output_item_done_event = True
            return self.create_output_item_done_event(litellm_complete_object)
        return None

    def return_default_initial_events(
        self,
    ) -> Optional[BaseLiteLLMOpenAIResponseObject]:
        if self.sent_response_created_event is False:
            self.sent_response_created_event = True
            return self.create_response_created_event()
        elif self.sent_response_in_progress_event is False:
            self.sent_response_in_progress_event = True
            return self.create_response_in_progress_event()
        elif self.sent_output_item_added_event is False:
            self.sent_output_item_added_event = True
            return self.create_output_item_added_event()
        elif self.sent_content_part_added_event is False:
            self.sent_content_part_added_event = True
            return self.create_content_part_added_event()
        return None

    def is_stream_finished(self) -> bool:
        if (
            self.sent_output_text_done_event is True
            and self.sent_output_content_part_done_event is True
            and self.sent_output_item_done_event is True
        ):
            return True
        return False

    def common_done_event_logic(
        self, sync_mode: bool = True
    ) -> BaseLiteLLMOpenAIResponseObject:
        if not self.litellm_model_response or isinstance(
            self.litellm_model_response, TextCompletionResponse
        ):
            self.litellm_model_response = self.create_litellm_model_response()
        if self.litellm_model_response:
            done_event = self.return_default_done_events(self.litellm_model_response)
            if done_event:
                return done_event
        else:
            if sync_mode:
                raise StopIteration
            else:
                raise StopAsyncIteration

        self.finished = self.is_stream_finished()
        response_completed_event = self._emit_response_completed_event()
        if response_completed_event:
            return response_completed_event
        else:
            if sync_mode:
                raise StopIteration
            else:
                raise StopAsyncIteration

    async def __anext__(
        self,
    ) -> Union[
        ResponsesAPIStreamingResponse,
        ResponseCompletedEvent,
        BaseLiteLLMOpenAIResponseObject,
    ]:
        try:
            while True:
                if self.finished is True:
                    raise StopAsyncIteration

                result = self.return_default_initial_events()
                if result:
                    return result
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
    ) -> Union[
        ResponsesAPIStreamingResponse,
        ResponseCompletedEvent,
        BaseLiteLLMOpenAIResponseObject,
    ]:
        try:
            while True:
                if self.finished is True:
                    raise StopIteration
                # Get the next chunk from the stream

                result = self.return_default_initial_events()
                if result:
                    return result
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
                    return self.common_done_event_logic(sync_mode=True)
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
        It also handles emitting annotation.added events when annotations are detected in the chunk.
        """
        # Check if this chunk has annotations first (before processing text/reasoning)
        # This ensures we detect and queue annotation events from the annotation chunk
        if chunk.choices and hasattr(chunk.choices[0].delta, "annotations"):
            annotations = chunk.choices[0].delta.annotations
            if annotations and self.sent_annotation_events is False:
                self.sent_annotation_events = True
                # Store annotation events to emit them one by one
                if not hasattr(self, '_pending_annotation_events'):
                    
                    response_annotations = LiteLLMCompletionResponsesConfig._transform_chat_completion_annotations_to_response_output_annotations(
                        annotations=annotations
                    )                    
                    self._pending_annotation_events = []
                    for idx, annotation in enumerate(response_annotations):
                        annotation_dict = annotation.model_dump() if hasattr(annotation, 'model_dump') else dict(annotation)
                        event = OutputTextAnnotationAddedEvent(
                            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED,
                            item_id=chunk.id,
                            output_index=0,
                            content_index=0,
                            annotation_index=idx,
                            annotation=annotation_dict,
                        )
                        self._pending_annotation_events.append(event)        
        # Priority 1: Handle reasoning content (highest priority)
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

        # Priority 2: Handle text deltas
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

        # Priority 3: If we have pending annotation events, emit the next one
        # This happens when the current chunk has no text/reasoning content
        if hasattr(self, '_pending_annotation_events') and self._pending_annotation_events:
            event = self._pending_annotation_events.pop(0)
            return event

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
            # Add cost to usage object if include_cost_in_streaming_usage is True
            if (
                litellm.include_cost_in_streaming_usage
                and self.litellm_logging_obj is not None
            ):
                usage = getattr(litellm_model_response, "usage", None)
                if usage is not None:
                    setattr(
                        usage,
                        "cost",
                        self.litellm_logging_obj._response_cost_calculator(
                            result=litellm_model_response
                        ),
                    )

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
