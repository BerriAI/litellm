from typing import Final, Literal, Optional, Union

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import PrivateAttr
from typing_extensions import Any, TypedDict

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

Phase = Literal["commentary", "final_answer"] | None


class GenericResponseOutputItemContentAnnotation(BaseLiteLLMOpenAIResponseObject):
    """Annotation for content in a message"""

    type: str | None
    start_index: int | None
    end_index: int | None
    url: str | None
    title: str | None


class OutputText(BaseLiteLLMOpenAIResponseObject):
    """Text output content from an assistant message"""

    type: str | None  # "output_text"
    text: str | None
    annotations: list[GenericResponseOutputItemContentAnnotation] | None


class OutputFunctionToolCall(BaseLiteLLMOpenAIResponseObject):
    """A tool call to run a function"""

    arguments: str | None
    call_id: str | None
    name: str | None
    type: str | None  # "function_call"
    id: str | None
    status: Literal["in_progress", "completed", "incomplete"]
    phase: Phase = None


class OutputImageGenerationCall(BaseLiteLLMOpenAIResponseObject):
    """An image generation call output"""

    type: Literal["image_generation_call"]
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    result: str | None  # Base64 encoded image data (without data:image prefix)


class OutputCodeInterpreterCallLog(BaseLiteLLMOpenAIResponseObject):
    """Log output from a code interpreter call"""

    type: Literal["logs"]
    logs: str


class OutputCodeInterpreterCall(BaseLiteLLMOpenAIResponseObject):
    """A code interpreter / code execution call output"""

    type: Literal["code_interpreter_call"]
    id: str
    code: str | None
    container_id: str | None
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    outputs: list[OutputCodeInterpreterCallLog] | None


def build_code_interpreter_log_outputs(
    content: Any,
) -> list[OutputCodeInterpreterCallLog] | None:
    """Convert Anthropic bash_code_execution stdout/stderr to log outputs.

    Shared by streaming (handler.py) and non-streaming (transformation.py) paths.
    """
    if not isinstance(content, dict):
        return None
    parts: Final = []
    if content.get("stdout"):
        parts.append(content["stdout"])
    if content.get("stderr"):
        parts.append(f"STDERR: {content['stderr']}")
    logs: Final = "".join(parts)
    return [OutputCodeInterpreterCallLog(type="logs", logs=logs)] if logs else None


class CustomToolCallOutputItem(BaseLiteLLMOpenAIResponseObject):
    """A custom/freeform tool call output item (e.g. apply_patch).

    Mirrors the ``custom_tool_call`` variant of OpenAI's Responses API output.
    Unlike ``OutputFunctionToolCall`` which uses ``arguments`` (JSON string),
    this uses ``input`` (raw string) for the tool payload.
    """

    type: Literal["custom_tool_call"]
    call_id: str
    id: str | None = None
    name: str
    input: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class GenericResponseOutputItem(BaseLiteLLMOpenAIResponseObject):
    """
    Generic response API output item

    """

    type: str  # "message"
    id: str
    status: str  # "completed", "in_progress", etc.
    role: str  # "assistant", "user", etc.
    content: list[OutputText]
    phase: Phase = None


class DeleteResponseResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of a delete response request

    {
        "id": "resp_6786a1bec27481909a17d673315b29f6",
        "object": "response",
        "deleted": true
    }
    """

    id: str | None
    object: str | None
    deleted: bool | None

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class DecodedResponseId(TypedDict, total=False):
    """Structure representing a decoded response ID"""

    custom_llm_provider: str | None
    model_id: str | None
    response_id: str
