from typing import Literal

from typing_extensions import Any, List, Optional, TypedDict

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject


class GenericResponseOutputItemContentAnnotation(BaseLiteLLMOpenAIResponseObject):
    """Annotation for content in a message"""

    type: Optional[str]
    start_index: Optional[int]
    end_index: Optional[int]
    url: Optional[str]
    title: Optional[str]
    pass


class OutputText(BaseLiteLLMOpenAIResponseObject):
    """Text output content from an assistant message"""

    type: Optional[str]  # "output_text"
    text: Optional[str]
    annotations: Optional[List[GenericResponseOutputItemContentAnnotation]]


class OutputFunctionToolCall(BaseLiteLLMOpenAIResponseObject):
    """A tool call to run a function"""

    arguments: Optional[str]
    call_id: Optional[str]
    name: Optional[str]
    type: Optional[str]  # "function_call"
    id: Optional[str]
    status: Literal["in_progress", "completed", "incomplete"]


class GenericResponseOutputItem(BaseLiteLLMOpenAIResponseObject):
    """
    Generic response API output item

    """

    type: str  # "message"
    id: str
    status: str  # "completed", "in_progress", etc.
    role: str  # "assistant", "user", etc.
    content: List[OutputText]
