from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, Union

from typing_extensions import Required, TypedDict

from ..llms.openai import ChatCompletionAudioDelta
from ..utils import (
    CompletionTokensDetails,
    CompletionTokensDetailsWrapper,
    Function,
    PromptTokensDetailsWrapper,
    ServerToolUse,
    Usage,
)


class StreamingUsageDict(TypedDict, total=False):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    prompt_tokens_details: Union[PromptTokensDetailsWrapper, dict[str, object], None]
    completion_tokens_details: Union[CompletionTokensDetailsWrapper, dict[str, object], None]
    server_tool_use: Union[ServerToolUse, dict[str, object], None]
    cost: Optional[float]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]


class StreamingToolCallFunctionDict(TypedDict, total=False):
    name: Optional[str]
    arguments: Optional[str]
    provider_specific_fields: Optional[Mapping[str, object]]


class StreamingToolCallFunctionLike(Protocol):
    name: Optional[str]
    arguments: Optional[str]
    provider_specific_fields: Optional[Mapping[str, object]]


class StreamingToolCallDict(TypedDict, total=False):
    id: Optional[str]
    type: Optional[str]
    index: int
    function: Union[StreamingToolCallFunctionDict, StreamingToolCallFunctionLike, None]
    provider_specific_fields: Optional[Mapping[str, object]]


class StreamingToolCallLike(Protocol):
    id: Optional[str]
    type: Optional[str]
    index: int
    function: Optional[StreamingToolCallFunctionLike]
    provider_specific_fields: Optional[Mapping[str, object]]


class StreamingThinkingBlockDict(TypedDict, total=False):
    type: Optional[str]
    thinking: Optional[str]
    data: Optional[str]
    signature: Optional[str]


class StreamingChunkDelta(TypedDict, total=False):
    role: Required[Optional[str]]
    content: Optional[str]
    reasoning_content: Optional[str]
    tool_calls: Sequence[Union[StreamingToolCallDict, StreamingToolCallLike]]
    thinking_blocks: Optional[Sequence[StreamingThinkingBlockDict]]
    audio: Optional[ChatCompletionAudioDelta]


class StreamingChunkChoice(TypedDict, total=False):
    index: int
    finish_reason: Optional[str]
    delta: Required[StreamingChunkDelta]


class StreamingChunkDict(TypedDict, total=False):
    _hidden_params: dict[str, object]
    id: Required[str]
    object: Required[str]
    created: Required[int]
    model: Required[str]
    system_fingerprint: Optional[str]
    choices: Required[list[StreamingChunkChoice]]
    usage: Union[Usage, StreamingUsageDict, None]


class UsageChunkCalculation(TypedDict):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: Optional[PromptTokensDetailsWrapper]
    cost: Optional[float]


class ToolCallAccumulator(TypedDict):
    id: Optional[str]
    name: Optional[str]
    type: Optional[str]
    arguments: list[str]
    provider_specific_fields: Optional[dict[str, object]]


class ToolCallParams(TypedDict, total=False):
    id: str
    function: Function
    type: str
    provider_specific_fields: dict[str, object]


class UsagePerChunk(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
    server_tool_use: Optional[ServerToolUse]
    web_search_requests: Optional[int]
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: Optional[PromptTokensDetailsWrapper]
    cost: Optional[float]
