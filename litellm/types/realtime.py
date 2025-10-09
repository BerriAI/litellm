from typing import List, Literal, Optional, Union

from typing_extensions import TypedDict

from .llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
)

ALL_DELTA_TYPES = Literal["text", "audio"]


class RealtimeResponseTransformInput(TypedDict):
    session_configuration_request: Optional[str]
    current_output_item_id: Optional[
        str
    ]  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_response_id: Optional[
        str
    ]  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]]
    current_conversation_id: Optional[str]
    current_delta_type: Optional[ALL_DELTA_TYPES]


class RealtimeResponseTypedDict(TypedDict):
    response: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_conversation_id: Optional[str]
    current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]]
    current_delta_type: Optional[ALL_DELTA_TYPES]
    session_configuration_request: Optional[str]


class RealtimeModalityResponseTransformOutput(TypedDict):
    returned_message: List[OpenAIRealtimeEvents]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_conversation_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_delta_type: Optional[ALL_DELTA_TYPES]


class RealtimeQueryParams(TypedDict, total=False):
    model: str
    intent: Optional[str]
    # Add more fields as needed
