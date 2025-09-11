from typing import List, Union, Generic, Optional
from typing_extensions import Literal

from ._types import ParsedChatCompletionSnapshot
from ...._models import BaseModel, GenericModel
from ..._parsing import ResponseFormatT
from ....types.chat import ChatCompletionChunk, ChatCompletionTokenLogprob


class ChunkEvent(BaseModel):
    type: Literal["chunk"]

    chunk: ChatCompletionChunk

    snapshot: ParsedChatCompletionSnapshot


class ContentDeltaEvent(BaseModel):
    """This event is yielded for every chunk with `choice.delta.content` data."""

    type: Literal["content.delta"]

    delta: str

    snapshot: str

    parsed: Optional[object] = None


class ContentDoneEvent(GenericModel, Generic[ResponseFormatT]):
    type: Literal["content.done"]

    content: str

    parsed: Optional[ResponseFormatT] = None


class RefusalDeltaEvent(BaseModel):
    type: Literal["refusal.delta"]

    delta: str

    snapshot: str


class RefusalDoneEvent(BaseModel):
    type: Literal["refusal.done"]

    refusal: str


class FunctionToolCallArgumentsDeltaEvent(BaseModel):
    type: Literal["tool_calls.function.arguments.delta"]

    name: str

    index: int

    arguments: str
    """Accumulated raw JSON string"""

    parsed_arguments: object
    """The parsed arguments so far"""

    arguments_delta: str
    """The JSON string delta"""


class FunctionToolCallArgumentsDoneEvent(BaseModel):
    type: Literal["tool_calls.function.arguments.done"]

    name: str

    index: int

    arguments: str
    """Accumulated raw JSON string"""

    parsed_arguments: object
    """The parsed arguments"""


class LogprobsContentDeltaEvent(BaseModel):
    type: Literal["logprobs.content.delta"]

    content: List[ChatCompletionTokenLogprob]

    snapshot: List[ChatCompletionTokenLogprob]


class LogprobsContentDoneEvent(BaseModel):
    type: Literal["logprobs.content.done"]

    content: List[ChatCompletionTokenLogprob]


class LogprobsRefusalDeltaEvent(BaseModel):
    type: Literal["logprobs.refusal.delta"]

    refusal: List[ChatCompletionTokenLogprob]

    snapshot: List[ChatCompletionTokenLogprob]


class LogprobsRefusalDoneEvent(BaseModel):
    type: Literal["logprobs.refusal.done"]

    refusal: List[ChatCompletionTokenLogprob]


ChatCompletionStreamEvent = Union[
    ChunkEvent,
    ContentDeltaEvent,
    ContentDoneEvent[ResponseFormatT],
    RefusalDeltaEvent,
    RefusalDoneEvent,
    FunctionToolCallArgumentsDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
    LogprobsContentDeltaEvent,
    LogprobsContentDoneEvent,
    LogprobsRefusalDeltaEvent,
    LogprobsRefusalDoneEvent,
]
