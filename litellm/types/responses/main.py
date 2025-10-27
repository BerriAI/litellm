from typing import List, Literal, Optional, Union

from pydantic import PrivateAttr
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


class DeleteResponseResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of a delete response request

    {
        "id": "resp_6786a1bec27481909a17d673315b29f6",
        "object": "response",
        "deleted": true
    }
    """

    id: Optional[str]
    object: Optional[str]
    deleted: Optional[bool]

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class DecodedResponseId(TypedDict, total=False):
    """Structure representing a decoded response ID"""

    custom_llm_provider: Optional[str]
    model_id: Optional[str]
    response_id: str
