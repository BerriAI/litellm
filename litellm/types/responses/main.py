import builtins
import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Optional, Union

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    Action,
    ActionOpenPage,
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from pydantic import ConfigDict, PrivateAttr, TypeAdapter, ValidationError
from typing_extensions import TypedDict

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


HOSTED_WEB_TOOL_NAMES: Final = frozenset({"web_search", "web_fetch"})


def _web_search_action(tool_input: object, content: object) -> ActionSearch:
    query: Final = tool_input.get("query", "") if isinstance(tool_input, Mapping) else ""
    result_items: Final = content if isinstance(content, Sequence) and not isinstance(content, (str, bytes)) else ()
    sources: Final = [
        ActionSearchSource(type="url", url=url)
        for item in result_items
        if isinstance(item, Mapping)
        and item.get("type") == "web_search_result"
        and isinstance((url := item.get("url")), str)
    ]
    return ActionSearch(
        type="search",
        query=query if isinstance(query, str) else "",
        queries=[query] if isinstance(query, str) and query else [],
        sources=sources,
    )


def _web_fetch_action(tool_input: object) -> ActionOpenPage:
    match tool_input:
        case {"url": str() as url}:
            return ActionOpenPage(type="open_page", url=url)
        case _:
            return ActionOpenPage(type="open_page", url=None)


def _result_content(result: object) -> object:
    match result:
        case {"content": object() as content}:
            return content
        case _:
            return None


def _is_tool_result_error(content: object, tool_name: str) -> bool:
    match content:
        case {"type": str() as content_type}:
            return content_type == f"{tool_name}_tool_result_error"
        case _:
            return False


WEB_SEARCH_CALL_ID_PREFIX: Final = "ws_"


def build_web_search_call(
    tool_id: str,
    tool_input: object,
    result: object,
    status: Literal["in_progress", "searching", "completed", "failed"] | None = None,
    tool_name: str = "web_search",
) -> ResponseFunctionWebSearch:
    content: Final = _result_content(result)
    return ResponseFunctionWebSearch(
        id=f"{WEB_SEARCH_CALL_ID_PREFIX}{tool_id}",
        type="web_search_call",
        status=status or ("failed" if _is_tool_result_error(content, tool_name) else "completed"),
        action=_web_fetch_action(tool_input) if tool_name == "web_fetch" else _web_search_action(tool_input, content),
    )


def web_search_call_tool_id(call_id: str) -> str | None:
    tool_id: Final = call_id.removeprefix(WEB_SEARCH_CALL_ID_PREFIX)
    return None if tool_id == call_id else tool_id


_HOSTED_WEB_TOOL_ARGUMENTS: Final = TypeAdapter(dict[str, object])


def hosted_web_tool_call(action: object) -> tuple[str, str] | None:
    match action:
        case ActionSearch(query=str() as query):
            return "web_search", json.dumps({"query": query})
        case ActionOpenPage(url=str() as url):
            return "web_fetch", json.dumps({"url": url})
        case _:
            return None


def _decoded_hosted_web_tool_arguments(arguments: str) -> Mapping[str, object]:
    try:
        return _HOSTED_WEB_TOOL_ARGUMENTS.validate_json(arguments)
    except ValidationError:
        return {}


def with_hosted_web_tool_arguments(action: Action, arguments: str) -> Action:
    match action, _decoded_hosted_web_tool_arguments(arguments):
        case ActionSearch(), {"query": str() as query}:
            return action.model_copy(update={"query": query, "queries": [query] if query else []})
        case ActionSearch(), _:
            return action.model_copy(update={"query": "", "queries": []})
        case ActionOpenPage(), {"url": str() as url}:
            return action.model_copy(update={"url": url})
        case ActionOpenPage(), _:
            return action.model_copy(update={"url": None})
        case _:
            return action


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
    content: object,
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


class OutputToolSearchCall(BaseLiteLLMOpenAIResponseObject):
    model_config = ConfigDict(extra="allow", frozen=True, protected_namespaces=())

    id: str
    arguments: object
    call_id: str | None = None
    execution: Literal["server", "client"]
    status: Literal["in_progress", "completed", "incomplete"]
    type: Literal["tool_search_call"]


class GenericResponseOutputItem(BaseLiteLLMOpenAIResponseObject):
    """
    Generic response API output item

    """

    type: str  # "message"
    id: str
    status: str  # "completed", "in_progress", etc.
    role: str  # "assistant", "user", etc.
    content: list[OutputText | ResponseOutputRefusal]
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

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params


class DecodedResponseId(TypedDict, total=False):
    """Structure representing a decoded response ID"""

    custom_llm_provider: str | None
    model_id: str | None
    response_id: str
