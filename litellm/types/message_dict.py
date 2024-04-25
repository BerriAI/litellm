from typing import Union, TypedDict, NotRequired

from typing_extensions import deprecated


class SystemMessageDict(TypedDict):
    content: str
    role: str
    name: NotRequired[str]


class UserMessageContentTextDict(TypedDict):
    type: str
    text: str


class ImageUrlDict(TypedDict):
    url: str
    detail: NotRequired[str]


class UserMessageContentImageDict(TypedDict):
    type: str
    image_url: ImageUrlDict


class UserMessageDict(TypedDict):
    content: Union[
        str, list[Union[UserMessageContentTextDict, UserMessageContentImageDict]]
    ]
    role: str
    name: NotRequired[str]


@deprecated("Deprecated and replaced by tool_calls")
class AssistantFunctionCallDict(TypedDict):
    arguments: str
    name: str


class AssistantToolCallsDict(TypedDict):
    id: str
    type: str
    function: str


class AssistantMessageDictV1(TypedDict):
    content: Union[None, str]
    role: str
    name: NotRequired[str]


class AssistantMessageDictV2(TypedDict):
    # `content` is required if neither `tool_calls` or `function_call` is present
    content: NotRequired[Union[None, str]]
    role: str
    name: NotRequired[str]
    tool_calls: list[AssistantToolCallsDict]
    function_call: NotRequired[AssistantFunctionCallDict]


class AssistantMessageDictV3(TypedDict):
    content: NotRequired[Union[None, str]]
    role: str
    name: NotRequired[str]
    tool_calls: NotRequired[list[AssistantToolCallsDict]]
    function_call: AssistantFunctionCallDict


AssistantMessageDict = Union[
    # `content` is required if neither `tool_calls` or `function_call` is present
    AssistantMessageDictV1,
    AssistantMessageDictV2,
    AssistantMessageDictV3,
]


class ToolMessageDict(TypedDict):
    role: str
    content: str
    tool_call_id: str


@deprecated("Deprecated")
class FunctionMessageDict(TypedDict):
    role: str
    content: Union[None, str]
    name: str


class TextContentModel(TypedDict):
    type: str
    text: str


class ImageUrlModel(TypedDict):
    url: str
    detail: NotRequired[str]


class ImageContentModel(TypedDict):
    type: str
    image_url: ImageUrlModel


MessageDict = Union[
    SystemMessageDict,
    UserMessageDict,
    AssistantMessageDict,
    ToolMessageDict,
    FunctionMessageDict,
]
