from typing import Iterable, List, Optional, Union

from typing_extensions import Literal, Required, TypedDict


class CallObject(TypedDict):
    name: str
    parameters: dict


class ToolResultObject(TypedDict):
    call: CallObject
    outputs: List[dict]


class ChatHistoryToolResult(TypedDict, total=False):
    role: Required[Literal["TOOL"]]
    tool_results: List[ToolResultObject]


class ToolCallObject(TypedDict):
    name: str
    parameters: dict


class ChatHistoryUser(TypedDict, total=False):
    role: Required[Literal["USER"]]
    message: str
    tool_calls: List[ToolCallObject]


class ChatHistorySystem(TypedDict, total=False):
    role: Required[Literal["SYSTEM"]]
    message: str
    tool_calls: List[ToolCallObject]


class ChatHistoryChatBot(TypedDict, total=False):
    role: Required[Literal["CHATBOT"]]
    message: str
    tool_calls: List[ToolCallObject]


ChatHistory = List[
    Union[ChatHistorySystem, ChatHistoryChatBot, ChatHistoryUser, ChatHistoryToolResult]
]


class CohereV2ChatResponseMessageToolCallFunction(TypedDict, total=False):
    name: str
    parameters: dict


class CohereV2ChatResponseMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: CohereV2ChatResponseMessageToolCallFunction


class CohereV2ChatResponseMessageContent(TypedDict):
    id: str
    type: Literal["tool"]
    tool: str


class CohereV2ChatResponseMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    tool_calls: List[CohereV2ChatResponseMessageToolCall]
    tool_plan: str
    content: List[CohereV2ChatResponseMessageContent]
    citations: List[dict]


class CohereV2ChatResponseUsageBilledUnits(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    search_units: int
    classifications: int


class CohereV2ChatResponseUsageTokens(TypedDict, total=False):
    input_tokens: int
    output_tokens: int


class CohereV2ChatResponseUsage(TypedDict, total=False):
    billed_units: CohereV2ChatResponseUsageBilledUnits
    tokens: CohereV2ChatResponseUsageTokens


class CohereV2ChatResponseLogProbs(TypedDict, total=False):
    token_ids: Required[List[int]]
    text: str
    logprobs: List[float]


class CohereV2ChatResponse(TypedDict):
    id: str
    finish_reason: str
    message: CohereV2ChatResponseMessage
    usage: CohereV2ChatResponseUsage
    logprobs: CohereV2ChatResponseLogProbs
