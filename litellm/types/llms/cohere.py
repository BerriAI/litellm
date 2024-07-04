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
