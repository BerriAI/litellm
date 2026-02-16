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
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextAnnotationAddedEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningSummaryPartDoneEvent,
    ReasoningSummaryTextDeltaEvent,
    ReasoningSummaryTextDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
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
        model: str,
        litellm_custom_stream_wrapper: litellm.CustomStreamWrapper,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        litellm_metadata: Optional[dict] = None,
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
        self._cached_item_id: Optional[str] = None
        self._cached_response_id: Optional[str] = None
        self._pending_tool_events: List[BaseLiteLLMOpenAIResponseObject] = []
        self._tool_output_index_by_call_id: dict[str, int] = {}
        self._tool_args_by_call_id: dict[str, str] = {}
        self._tool_call_id_by_index: dict[int, str] = {}
        self._ambiguous_tool_call_indexes: set[int] = set()
        self._next_tool_output_index: int = 1  # output_index=0 reserved for the message item
        self._final_tool_events_queued: bool = False
        self._sequence_number: int = 0  
        self._cached_reasoning_item_id: Optional[str] = None
        self._sent_reasoning_summary_text_done_event: bool = False
        self._sent_reasoning_summary_part_done_event: bool = False
        self._reasoning_summary_text: str = ""
        # -- GENERIC RESPONSE-EVENTS PENDING QUEUE as required by fix --
        self._pending_response_events: List[BaseLiteLLMOpenAIResponseObject] = []
        self._reasoning_active = False
        self._reasoning_done_emitted = False
        self._reasoning_item_id: Optional[str] = None


    def _get_or_assign_tool_output_index(self, call_id: str) -> int:
        existing = self._tool_output_index_by_call_id.get(call_id)
        if existing is not None:
            return existing
        idx = self._next_tool_output_index
        self._next_tool_output_index += 1
        self._tool_output_index_by_call_id[call_id] = idx
        return idx

    def _normalize_tool_call_index(self, tool_call: object) -> Optional[int]:
        idx_raw = (
            tool_call.get("index")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "index", None)
        )
        if idx_raw is None:
            return None
        try:
            return int(idx_raw)
        except (TypeError, ValueError):
            return None


    def _is_reasoning_end(self, chunk):
        delta = chunk.choices[0].delta

        # if this indicates reasoning content, don't consider reasoning ended
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            return False
        if hasattr(delta, "thinking_blocks") and delta.thinking_blocks:
            return False

        return (
            delta.content
            or delta.function_call
            or delta.tool_calls
            or chunk.choices[0].finish_reason is not None
        )

    def _queue_tool_call_delta_events(self, tool_calls: object) -> None:
        """
        Convert chat-completions streaming `tool_calls` deltas into Responses API streaming events.

        We emit:
        - response.output_item.added (function_call)
        - response.function_call_arguments.delta (split into smaller chunks to match OpenAI behavior)
        
        Note: Some providers (like Bedrock) send tool call arguments in one large chunk.
        We split these into smaller deltas to match OpenAI's token-by-token streaming behavior.
        """
        if not isinstance(tool_calls, list):
            return

        for tc in tool_calls:
            tc_index = self._normalize_tool_call_index(tc)
            call_id_raw = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            call_id = ""

            if call_id_raw:
                call_id = str(call_id_raw)
                if tc_index is not None:
                    existing_call_id = self._tool_call_id_by_index.get(tc_index)
                    if existing_call_id is not None and existing_call_id != call_id:
                        # Reusing the same index for multiple call_ids is ambiguous for id-less deltas.
                        # Guard against silent misrouting by disabling index fallback for this index.
                        self._ambiguous_tool_call_indexes.add(tc_index)
                    self._tool_call_id_by_index[tc_index] = call_id
            elif tc_index is not None:
                if tc_index in self._ambiguous_tool_call_indexes:
                    continue
                mapped_call_id = self._tool_call_id_by_index.get(tc_index)
                if mapped_call_id:
                    call_id = mapped_call_id

            if not call_id:
                continue

            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            fn_name = ""
            fn_args_delta = ""
            if isinstance(fn, dict):
                fn_name = str(fn.get("name") or "")
                fn_args_delta = str(fn.get("arguments") or "")
            else:
                fn_name = str(getattr(fn, "name", "") or "")
                fn_args_delta = str(getattr(fn, "arguments", "") or "")

            output_index = self._get_or_assign_tool_output_index(call_id)

            if call_id not in self._tool_args_by_call_id:
                self._tool_args_by_call_id[call_id] = ""
                self._sequence_number += 1
                event = OutputItemAddedEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                    output_index=output_index,
                    item=BaseLiteLLMOpenAIResponseObject(
                        **{
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": "",
                            "status": "in_progress",
                        }
                    ),
                )
                event.__dict__['sequence_number'] = self._sequence_number
                self._pending_tool_events.append(event)

            if fn_args_delta:
                self._tool_args_by_call_id[call_id] += fn_args_delta
                
                # Split large argument deltas into smaller chunks to match OpenAI's streaming behavior
                # This is especially important for providers like Bedrock that send complete arguments at once
                chunk_size = 10  # Match typical OpenAI delta size
                for i in range(0, len(fn_args_delta), chunk_size):
                    delta_chunk = fn_args_delta[i:i + chunk_size]
                    self._sequence_number += 1
                    delta_event: BaseLiteLLMOpenAIResponseObject = FunctionCallArgumentsDeltaEvent(
                        type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                        item_id=call_id,
                        output_index=output_index,
                        delta=delta_chunk,
                    )
                    # Add sequence_number as extra field (BaseLiteLLMOpenAIResponseObject allows extra fields)
                    delta_event.__dict__['sequence_number'] = self._sequence_number
                    self._pending_tool_events.append(delta_event)

    def _queue_final_tool_call_done_events(self, litellm_complete_object: ModelResponse) -> None:
        """
        Ensure tool calls that were not streamed as deltas still get emitted before response.completed.
        """
        if self._final_tool_events_queued:
            return
        self._final_tool_events_queued = True

        try:
            message = litellm_complete_object.choices[0].message  # type: ignore
            tool_calls = getattr(message, "tool_calls", None)
        except Exception:
            tool_calls = None

        if not tool_calls or not isinstance(tool_calls, list):
            return

        for tc in tool_calls:
            call_id_raw = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if not call_id_raw:
                continue
            call_id = str(call_id_raw)
            output_index = self._get_or_assign_tool_output_index(call_id)

            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            fn_name = ""
            fn_args = ""
            if isinstance(fn, dict):
                fn_name = str(fn.get("name") or "")
                fn_args = str(fn.get("arguments") or "")
            else:
                fn_name = str(getattr(fn, "name", "") or "")
                fn_args = str(getattr(fn, "arguments", "") or "")

            # Track if this is a new tool call that wasn't streamed
            is_new_tool_call = call_id not in self._tool_args_by_call_id
            
            # If we never sent output_item.added for this call_id, emit it now.
            if is_new_tool_call:
                self._tool_args_by_call_id[call_id] = ""
                self._sequence_number += 1
                event = OutputItemAddedEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                    output_index=output_index,
                    item=BaseLiteLLMOpenAIResponseObject(
                        **{
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": fn_name,
                            "arguments": "",
                            "status": "in_progress",
                        }
                    ),
                )
                event.__dict__['sequence_number'] = self._sequence_number
                self._pending_tool_events.append(event)

            final_args = fn_args or self._tool_args_by_call_id.get(call_id, "")
            
            # Emit delta events for arguments that weren't streamed yet
            # This handles cases where Bedrock sends the complete tool call at the end
            already_streamed = self._tool_args_by_call_id.get(call_id, "")
            remaining_args = final_args[len(already_streamed):] if final_args else ""
            
            if remaining_args:
                # Split into smaller chunks to match OpenAI's streaming behavior
                chunk_size = 10  # Match typical OpenAI delta size
                for i in range(0, len(remaining_args), chunk_size):
                    delta_chunk = remaining_args[i:i + chunk_size]
                    self._sequence_number += 1
                    delta_event = FunctionCallArgumentsDeltaEvent(
                        type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                        item_id=call_id,
                        output_index=output_index,
                        delta=delta_chunk,
                    )
                    delta_event.__dict__['sequence_number'] = self._sequence_number
                    self._pending_tool_events.append(delta_event)
            
            self._sequence_number += 1
            done_event = FunctionCallArgumentsDoneEvent(
                type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
                item_id=call_id,
                output_index=output_index,
                arguments=final_args,
            )
            done_event.__dict__['sequence_number'] = self._sequence_number
            self._pending_tool_events.append(done_event)

            self._sequence_number += 1
            item_done_event = OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=output_index,
                sequence_number=self._sequence_number,
                item=BaseLiteLLMOpenAIResponseObject(
                    **{
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": fn_name,
                        "arguments": final_args,
                        "status": "completed",
                    }
                ),
            )
            self._pending_tool_events.append(item_done_event)

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
            # Transform tool_choice from dict format (e.g., {"type": "auto"}) to string format
            response_created_event_data["tool_choice"] = LiteLLMCompletionResponsesConfig._transform_tool_choice(
                self.responses_api_request["tool_choice"]
            ) or "auto"
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
        self._sequence_number += 1
        event = ResponseCreatedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
            response=ResponsesAPIResponse(**response_created_event_data),
        )
        event.__dict__['sequence_number'] = self._sequence_number
        return event

    def create_response_in_progress_event(self) -> ResponseInProgressEvent:
        response_in_progress_event_data = self._default_response_created_event_data()
        response_in_progress_event_data["status"] = "in_progress"
        self._sequence_number += 1
        event = ResponseInProgressEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
            response=ResponsesAPIResponse(**response_in_progress_event_data),
        )
        event.__dict__['sequence_number'] = self._sequence_number
        return event

    def create_output_item_added_event(self) -> OutputItemAddedEvent:
        if self._cached_item_id is None:
            self._cached_item_id = f"msg_{str(uuid.uuid4())}"
        
        self._sequence_number += 1
        event = OutputItemAddedEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=BaseLiteLLMOpenAIResponseObject(
                **{
                    "id": self._cached_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
            ),
        )
        event.__dict__['sequence_number'] = self._sequence_number
        return event

    def create_content_part_added_event(self) -> ContentPartAddedEvent:
        if self._cached_item_id is None:
            self._cached_item_id = f"msg_{str(uuid.uuid4())}"
        
        self._sequence_number += 1
        event = ContentPartAddedEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
            item_id=self._cached_item_id,
            output_index=0,
            content_index=0,
            part=BaseLiteLLMOpenAIResponseObject(
                **{"type": "output_text", "text": "", "annotations": []}
            ),
        )
        event.__dict__['sequence_number'] = self._sequence_number
        return event

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

    def create_reasoning_summary_text_done_event(
        self,
        reasoning_item_id: str,
        reasoning_content: str,
        sequence_number: int,
    ) -> ReasoningSummaryTextDoneEvent:
        """
        Create response.reasoning_summary_text.done event.
        
        Example:
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_0c5dae30e53172980069708ba2f59c8197b71ca9820edad07c",
            "output_index": 0,
            "sequence_number": 97,
            "summary_index": 0,
            "text": "**Clarifying the first humans**\n\nThe  I'm addressing the user's specific interest."
        }
        """
        return ReasoningSummaryTextDoneEvent(
            type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
            item_id=reasoning_item_id,
            output_index=0,
            sequence_number=sequence_number,
            summary_index=0,
            text=reasoning_content,
        )

    def create_reasoning_summary_part_done_event(
        self, 
        reasoning_item_id: str,
        reasoning_content: str,
        sequence_number: int,
    ) -> ReasoningSummaryPartDoneEvent:
        """
        Create response.reasoning_summary_part.done event.
        
        Example:
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_0c5dae30e53172980069708ba2f59c8197b71ca9820edad07c",
            "output_index": 0,
            "part": {
                "type": "summary_text",
                "text": "**Clarifying the first humans**\n\nThe  earlier hominins. It feels important to ensure I'm addressing the user's specific interest."
            },
            "sequence_number": 98,
            "summary_index": 0
        }
        """
        return ReasoningSummaryPartDoneEvent(
            type=ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE,
            item_id=reasoning_item_id,
            output_index=0,
            sequence_number=sequence_number,
            summary_index=0,
            part=BaseLiteLLMOpenAIResponseObject(
                **{
                    "type": "summary_text",
                    "text": reasoning_content,
                }
            ),
        )

    def create_output_text_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> OutputTextDoneEvent:
        if self._cached_item_id is None:
            self._cached_item_id = f"msg_{str(uuid.uuid4())}"
        
        return OutputTextDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            item_id=self._cached_item_id,
            output_index=0,
            content_index=0,
            text=getattr(litellm_complete_object.choices[0].message, "content", "")  # type: ignore
            or "",
        )

    def create_output_content_part_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> ContentPartDoneEvent:
        if self._cached_item_id is None:
            self._cached_item_id = f"msg_{str(uuid.uuid4())}"

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
            item_id=self._cached_item_id,
            output_index=0,
            content_index=0,
            part=part,
        )

    def create_output_item_done_event(
        self, litellm_complete_object: ModelResponse
    ) -> OutputItemDoneEvent:
        if self._cached_item_id is None:
            self._cached_item_id = f"msg_{str(uuid.uuid4())}"
        
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
                    "id": self._cached_item_id,
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

    def create_reasoning_output_item_done_event(
        self,
        reasoning_item_id: str,
        reasoning_content: str,
        sequence_number: int,
    ) -> OutputItemDoneEvent:
        """
        Create response.output_item.done event for reasoning items.
        
        Example:
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 99,
            "item": {
                "id": "rs_0c5dae30e53172980069708ba2f59c8197b71ca9820edad07c",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "**Clarifying the first humans**..."
                    }
                ]
            }
        }
        """
        return OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            sequence_number=sequence_number,
            item=BaseLiteLLMOpenAIResponseObject(
                **{
                    "id": reasoning_item_id,
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": reasoning_content,
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
            # If tool calls exist, emit tool events before finishing/response.completed.
            if isinstance(self.litellm_model_response, ModelResponse):
                self._queue_final_tool_call_done_events(self.litellm_model_response)
            if self._pending_tool_events:
                return self._pending_tool_events.pop(0)

            done_event = self.return_default_done_events(self.litellm_model_response)
            if done_event:
                return done_event
        else:
            if sync_mode:
                raise StopIteration
            else:
                raise StopAsyncIteration

        self.finished = self.is_stream_finished()
        response_completed_event = self._emit_response_completed_event(
            self.litellm_model_response
        )
        if response_completed_event:
            return response_completed_event
        else:
            if sync_mode:
                raise StopIteration
            else:
                raise StopAsyncIteration

    def _ensure_output_item_for_chunk(self, chunk: ModelResponseStream) -> None:
        # Change: Never return a value, just enqueue output item events
        if self.sent_output_item_added_event:
            return
        delta = chunk.choices[0].delta

        self._sequence_number += 1
        self.sent_output_item_added_event = True

        # Reasoning-first
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            self._reasoning_active = True
            if self._cached_reasoning_item_id is None:
                self._cached_reasoning_item_id = f"rs_{uuid.uuid4()}"
            self._reasoning_item_id = self._cached_reasoning_item_id

            event = OutputItemAddedEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=BaseLiteLLMOpenAIResponseObject(
                    **{
                        "id": self._cached_reasoning_item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": None,
                    }
                ),
            )
            event.__dict__['sequence_number'] = self._sequence_number
            self._pending_response_events.append(event)
            return

        # Tool-first
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            # Tool calls already handled via _queue_tool_call_delta_events
            # DO NOT create message item
            return

        # Default: message
        self._cached_item_id = self._cached_item_id or f"msg_{uuid.uuid4()}"
        event = OutputItemAddedEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=BaseLiteLLMOpenAIResponseObject(
                **{
                    "id": self._cached_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            ),
        )
        event.__dict__['sequence_number'] = self._sequence_number
        self._pending_response_events.append(event)
        return

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
                # Emit any pending output_item or other response events before reading a new chunk
                if self._pending_response_events:
                    return self._pending_response_events.pop(0)
                # Emit any pending tool events before reading a new chunk
                if self._pending_tool_events:
                    return self._pending_tool_events.pop(0)

                try:
                    chunk = await self.litellm_custom_stream_wrapper.__anext__()
                    if chunk is not None:
                        chunk = cast(ModelResponseStream, chunk)
                        self._ensure_output_item_for_chunk(chunk)
                        # Proceed to transformation
                        self.collected_chat_completion_chunks.append(chunk)
                        if self._reasoning_active and not self._reasoning_done_emitted:
                            # get raw ModelResponse
                            text_reasoning = self.create_litellm_model_response()
                            # reasoning_content only
                            if self._is_reasoning_end(chunk):
                                reasoning_content = ""
                                # best effort to obtain reasoning_content from chat model response
                                if text_reasoning and text_reasoning.choices:
                                    choice = text_reasoning.choices[0]
                                    # Check if it's a Choices object (has message) or StreamingChoices (has delta)
                                    if hasattr(choice, "message"):
                                        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""
                                
                                # Ensure we have a valid reasoning_item_id
                                reasoning_item_id = self._reasoning_item_id or self._cached_reasoning_item_id or f"rs_{uuid.uuid4()}"
                                
                                # Create text.done event first with its own sequence number
                                self._sequence_number += 1
                                text_done_event = self.create_reasoning_summary_text_done_event(
                                    reasoning_item_id=reasoning_item_id,
                                    reasoning_content=reasoning_content,
                                    sequence_number=self._sequence_number
                                )
                                
                                # Create part.done event second with its own sequence number
                                self._sequence_number += 1
                                part_done_event = self.create_reasoning_summary_part_done_event(
                                    reasoning_item_id=reasoning_item_id,
                                    reasoning_content=reasoning_content,
                                    sequence_number=self._sequence_number
                                )
                                
                                self._sequence_number += 1
                                reasoning_output_item_done_event = self.create_reasoning_output_item_done_event(
                                    reasoning_item_id=reasoning_item_id,
                                    reasoning_content=reasoning_content,
                                    sequence_number=self._sequence_number
                                )
                                self._pending_response_events.extend([
                                    text_done_event,
                                    part_done_event,
                                    reasoning_output_item_done_event,
                                ])
                                self._reasoning_done_emitted = True
                                self._reasoning_active = False

                        response_api_chunk = (
                            self._transform_chat_completion_chunk_to_response_api_chunk(
                                chunk
                            )
                        )
                        if response_api_chunk:
                            self._pending_response_events.append(response_api_chunk)
                
                    if self._pending_response_events:
                        return self._pending_response_events.pop(0)

                except StopAsyncIteration:
                    return self.common_done_event_logic(sync_mode=False)

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
                result = self.return_default_initial_events()
                if result:
                    return result
                # Emit any pending output_item or other response events before reading a new chunk
                if self._pending_response_events:
                    return self._pending_response_events.pop(0)
                # Emit any pending tool events before reading a new chunk
                if self._pending_tool_events:
                    return self._pending_tool_events.pop(0)
                try:
                    chunk = self.litellm_custom_stream_wrapper.__next__()
                    self._ensure_output_item_for_chunk(chunk)
                    # Emit any just-queued output_item event
                    if self._pending_response_events:
                        return self._pending_response_events.pop(0)
                    self.collected_chat_completion_chunks.append(chunk)
                    response_api_chunk = (
                        self._transform_chat_completion_chunk_to_response_api_chunk(
                            chunk
                        )
                    )
                    if response_api_chunk:
                        return response_api_chunk
                    # Otherwise, loop to next chunk
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
        if self._cached_item_id is None and chunk.id:
            self._cached_item_id = chunk.id
        item_id = self._cached_item_id or chunk.id
        
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
                            item_id=item_id,
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

            return ReasoningSummaryTextDeltaEvent(
                type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id=f"rs_{hash(str(reasoning_content))}",
                output_index=0,
                delta=reasoning_content,
            )
        
        # Priority 2: Handle text deltas
        delta_content = self._get_delta_string_from_streaming_choices(chunk.choices)
        if delta_content:
            self._sequence_number += 1
            text_delta_event = OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id=item_id,
                output_index=0,
                content_index=0,
                delta=delta_content,
            )
            text_delta_event.__dict__['sequence_number'] = self._sequence_number
            return text_delta_event

        # Priority 3: Handle tool call deltas (if any) -> queue events and emit them
        # For each tool call delta, we emit events one at a time to match OpenAI's streaming behavior
        if (
            chunk.choices
            and hasattr(chunk.choices[0].delta, "tool_calls")
            and chunk.choices[0].delta.tool_calls
        ):
            self._queue_tool_call_delta_events(chunk.choices[0].delta.tool_calls)
            # Return one pending tool event at a time
            if self._pending_tool_events:
                return self._pending_tool_events.pop(0)
        
        # Priority 4: If we have pending annotation events, emit the next one
        # This happens when the current chunk has no text/reasoning content
        if hasattr(self, '_pending_annotation_events') and self._pending_annotation_events:
            event = self._pending_annotation_events.pop(0)
            return event

        # Priority 5: If we have pending tool events (from earlier chunk), emit the next one
        if self._pending_tool_events:
            return self._pending_tool_events.pop(0)

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

    def _emit_response_completed_event(
        self, litellm_model_response: ModelResponse
    ) -> Optional[ResponseCompletedEvent]:

        if litellm_model_response:
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

            return ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=encoded_response,
            )
        else:
            return None
