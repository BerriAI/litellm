# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ..._utils import PropertyInfo
from .response_error_event import ResponseErrorEvent
from .response_failed_event import ResponseFailedEvent
from .response_queued_event import ResponseQueuedEvent
from .response_created_event import ResponseCreatedEvent
from .response_completed_event import ResponseCompletedEvent
from .response_text_done_event import ResponseTextDoneEvent
from .response_audio_done_event import ResponseAudioDoneEvent
from .response_incomplete_event import ResponseIncompleteEvent
from .response_text_delta_event import ResponseTextDeltaEvent
from .response_audio_delta_event import ResponseAudioDeltaEvent
from .response_in_progress_event import ResponseInProgressEvent
from .response_refusal_done_event import ResponseRefusalDoneEvent
from .response_refusal_delta_event import ResponseRefusalDeltaEvent
from .response_mcp_call_failed_event import ResponseMcpCallFailedEvent
from .response_output_item_done_event import ResponseOutputItemDoneEvent
from .response_content_part_done_event import ResponseContentPartDoneEvent
from .response_output_item_added_event import ResponseOutputItemAddedEvent
from .response_content_part_added_event import ResponseContentPartAddedEvent
from .response_mcp_call_completed_event import ResponseMcpCallCompletedEvent
from .response_reasoning_text_done_event import ResponseReasoningTextDoneEvent
from .response_mcp_call_in_progress_event import ResponseMcpCallInProgressEvent
from .response_reasoning_text_delta_event import ResponseReasoningTextDeltaEvent
from .response_audio_transcript_done_event import ResponseAudioTranscriptDoneEvent
from .response_mcp_list_tools_failed_event import ResponseMcpListToolsFailedEvent
from .response_audio_transcript_delta_event import ResponseAudioTranscriptDeltaEvent
from .response_mcp_call_arguments_done_event import ResponseMcpCallArgumentsDoneEvent
from .response_image_gen_call_completed_event import ResponseImageGenCallCompletedEvent
from .response_mcp_call_arguments_delta_event import ResponseMcpCallArgumentsDeltaEvent
from .response_mcp_list_tools_completed_event import ResponseMcpListToolsCompletedEvent
from .response_image_gen_call_generating_event import (
    ResponseImageGenCallGeneratingEvent,
)
from .response_web_search_call_completed_event import (
    ResponseWebSearchCallCompletedEvent,
)
from .response_web_search_call_searching_event import (
    ResponseWebSearchCallSearchingEvent,
)
from .response_file_search_call_completed_event import (
    ResponseFileSearchCallCompletedEvent,
)
from .response_file_search_call_searching_event import (
    ResponseFileSearchCallSearchingEvent,
)
from .response_image_gen_call_in_progress_event import (
    ResponseImageGenCallInProgressEvent,
)
from .response_mcp_list_tools_in_progress_event import (
    ResponseMcpListToolsInProgressEvent,
)
from .response_custom_tool_call_input_done_event import (
    ResponseCustomToolCallInputDoneEvent,
)
from .response_reasoning_summary_part_done_event import (
    ResponseReasoningSummaryPartDoneEvent,
)
from .response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent,
)
from .response_web_search_call_in_progress_event import (
    ResponseWebSearchCallInProgressEvent,
)
from .response_custom_tool_call_input_delta_event import (
    ResponseCustomToolCallInputDeltaEvent,
)
from .response_file_search_call_in_progress_event import (
    ResponseFileSearchCallInProgressEvent,
)
from .response_function_call_arguments_done_event import (
    ResponseFunctionCallArgumentsDoneEvent,
)
from .response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from .response_output_text_annotation_added_event import (
    ResponseOutputTextAnnotationAddedEvent,
)
from .response_reasoning_summary_part_added_event import (
    ResponseReasoningSummaryPartAddedEvent,
)
from .response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from .response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from .response_code_interpreter_call_code_done_event import (
    ResponseCodeInterpreterCallCodeDoneEvent,
)
from .response_code_interpreter_call_completed_event import (
    ResponseCodeInterpreterCallCompletedEvent,
)
from .response_code_interpreter_call_code_delta_event import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
)
from .response_code_interpreter_call_in_progress_event import (
    ResponseCodeInterpreterCallInProgressEvent,
)
from .response_code_interpreter_call_interpreting_event import (
    ResponseCodeInterpreterCallInterpretingEvent,
)

__all__ = ["ResponseStreamEvent"]

ResponseStreamEvent: TypeAlias = Annotated[
    Union[
        ResponseAudioDeltaEvent,
        ResponseAudioDoneEvent,
        ResponseAudioTranscriptDeltaEvent,
        ResponseAudioTranscriptDoneEvent,
        ResponseCodeInterpreterCallCodeDeltaEvent,
        ResponseCodeInterpreterCallCodeDoneEvent,
        ResponseCodeInterpreterCallCompletedEvent,
        ResponseCodeInterpreterCallInProgressEvent,
        ResponseCodeInterpreterCallInterpretingEvent,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseErrorEvent,
        ResponseFileSearchCallCompletedEvent,
        ResponseFileSearchCallInProgressEvent,
        ResponseFileSearchCallSearchingEvent,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseInProgressEvent,
        ResponseFailedEvent,
        ResponseIncompleteEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseReasoningSummaryPartAddedEvent,
        ResponseReasoningSummaryPartDoneEvent,
        ResponseReasoningSummaryTextDeltaEvent,
        ResponseReasoningSummaryTextDoneEvent,
        ResponseReasoningTextDeltaEvent,
        ResponseReasoningTextDoneEvent,
        ResponseRefusalDeltaEvent,
        ResponseRefusalDoneEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
        ResponseWebSearchCallCompletedEvent,
        ResponseWebSearchCallInProgressEvent,
        ResponseWebSearchCallSearchingEvent,
        ResponseImageGenCallCompletedEvent,
        ResponseImageGenCallGeneratingEvent,
        ResponseImageGenCallInProgressEvent,
        ResponseImageGenCallPartialImageEvent,
        ResponseMcpCallArgumentsDeltaEvent,
        ResponseMcpCallArgumentsDoneEvent,
        ResponseMcpCallCompletedEvent,
        ResponseMcpCallFailedEvent,
        ResponseMcpCallInProgressEvent,
        ResponseMcpListToolsCompletedEvent,
        ResponseMcpListToolsFailedEvent,
        ResponseMcpListToolsInProgressEvent,
        ResponseOutputTextAnnotationAddedEvent,
        ResponseQueuedEvent,
        ResponseCustomToolCallInputDeltaEvent,
        ResponseCustomToolCallInputDoneEvent,
    ],
    PropertyInfo(discriminator="type"),
]
