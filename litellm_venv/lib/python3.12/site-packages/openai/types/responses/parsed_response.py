# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import TYPE_CHECKING, List, Union, Generic, TypeVar, Optional
from typing_extensions import Annotated, TypeAlias

from ..._utils import PropertyInfo
from .response import Response
from ..._models import GenericModel
from ..._utils._transform import PropertyInfo
from .response_output_item import (
    McpCall,
    McpListTools,
    LocalShellCall,
    McpApprovalRequest,
    ImageGenerationCall,
    LocalShellCallAction,
)
from .response_output_text import ResponseOutputText
from .response_output_message import ResponseOutputMessage
from .response_output_refusal import ResponseOutputRefusal
from .response_reasoning_item import ResponseReasoningItem
from .response_custom_tool_call import ResponseCustomToolCall
from .response_computer_tool_call import ResponseComputerToolCall
from .response_function_tool_call import ResponseFunctionToolCall
from .response_function_web_search import ResponseFunctionWebSearch
from .response_file_search_tool_call import ResponseFileSearchToolCall
from .response_code_interpreter_tool_call import ResponseCodeInterpreterToolCall

__all__ = ["ParsedResponse", "ParsedResponseOutputMessage", "ParsedResponseOutputText"]

ContentType = TypeVar("ContentType")

# we need to disable this check because we're overriding properties
# with subclasses of their types which is technically unsound as
# properties can be mutated.
# pyright: reportIncompatibleVariableOverride=false


class ParsedResponseOutputText(ResponseOutputText, GenericModel, Generic[ContentType]):
    parsed: Optional[ContentType] = None


ParsedContent: TypeAlias = Annotated[
    Union[ParsedResponseOutputText[ContentType], ResponseOutputRefusal],
    PropertyInfo(discriminator="type"),
]


class ParsedResponseOutputMessage(
    ResponseOutputMessage, GenericModel, Generic[ContentType]
):
    if TYPE_CHECKING:
        content: List[ParsedContent[ContentType]]  # type: ignore[assignment]
    else:
        content: List[ParsedContent]


class ParsedResponseFunctionToolCall(ResponseFunctionToolCall):
    parsed_arguments: object = None

    __api_exclude__ = {"parsed_arguments"}


ParsedResponseOutputItem: TypeAlias = Annotated[
    Union[
        ParsedResponseOutputMessage[ContentType],
        ParsedResponseFunctionToolCall,
        ResponseFileSearchToolCall,
        ResponseFunctionWebSearch,
        ResponseComputerToolCall,
        ResponseReasoningItem,
        McpCall,
        McpApprovalRequest,
        ImageGenerationCall,
        LocalShellCall,
        LocalShellCallAction,
        McpListTools,
        ResponseCodeInterpreterToolCall,
        ResponseCustomToolCall,
    ],
    PropertyInfo(discriminator="type"),
]


class ParsedResponse(Response, GenericModel, Generic[ContentType]):
    if TYPE_CHECKING:
        output: List[ParsedResponseOutputItem[ContentType]]  # type: ignore[assignment]
    else:
        output: List[ParsedResponseOutputItem]

    @property
    def output_parsed(self) -> Optional[ContentType]:
        for output in self.output:
            if output.type == "message":
                for content in output.content:
                    if content.type == "output_text" and content.parsed:
                        return content.parsed

        return None
