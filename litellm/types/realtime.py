from typing import List, Optional, TypedDict, Union

from .llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
)


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


class RealtimeResponseTypedDict(TypedDict):
    response: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_conversation_id: Optional[str]
    current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]]


class RealtimeModalityResponseTransformOutput(TypedDict):
    returned_message: List[OpenAIRealtimeEvents]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_conversation_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
