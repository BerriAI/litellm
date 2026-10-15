from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.types.llms.anthropic import AnthropicMessagesTextParam
from litellm.types.utils import TokenCountResponse


@dataclass(frozen=True, slots=True)
class VertexGeminiCountContext:
    system_instruction: Mapping[str, object] | None
    tools: tuple[Mapping[str, object], ...] | None


@dataclass(frozen=True, slots=True)
class VertexGeminiCountContextError:
    error_message: str


_SYSTEM_BLOCKS: Final = TypeAdapter(tuple[AnthropicMessagesTextParam, ...])
_TOOLS: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])
_VERTEX_TOOL_KEYS: Final = MappingProxyType(
    {
        "function_declarations": "functionDeclarations",
        "url_context": "urlContext",
        "code_execution": "codeExecution",
    }
)


def _system_parts(
    system: object | None,
) -> tuple[Mapping[str, JsonValue], ...] | VertexGeminiCountContextError:
    if system is None:
        return ()
    if isinstance(system, str):
        return ({"text": system},) if system else ()
    try:
        blocks: Final = _SYSTEM_BLOCKS.validate_python(system)
    except ValidationError:
        return VertexGeminiCountContextError(error_message="Gemini system instructions only support text blocks.")
    return tuple({"text": block["text"]} for block in blocks)


_INVALID_TOOLS: Final = VertexGeminiCountContextError(
    error_message="tools must be a list of objects, and each function tool needs a JSON object schema."
)


def _has_object_schema(tool: Mapping[str, JsonValue]) -> bool:
    function: Final = tool.get("function")
    schema: Final = (
        function.get("parameters") if isinstance(function, dict) else tool.get("input_schema", tool.get("parameters"))
    )
    return schema is None or isinstance(schema, dict)


def _is_function_tool(tool: Mapping[str, JsonValue]) -> bool:
    return isinstance(tool.get("name"), str) or "input_schema" in tool or "function" in tool


def _is_responses_function_tool(tool: Mapping[str, JsonValue]) -> bool:
    return tool.get("type") == "function" and "function" not in tool and "input_schema" not in tool


def _chat_tool(tool: Mapping[str, JsonValue]) -> tuple[Mapping[str, object], ...]:
    from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    if _is_responses_function_tool(tool):
        chat_tools, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            [dict(tool)]  # pyright: ignore[reportArgumentType]  # validated JSON dict in the Responses function tool shape
        )
        return tuple(chat_tools)
    anthropic_tools, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai(
        [dict(tool)]  # pyright: ignore[reportArgumentType]  # validated JSON dict in an Anthropic or chat tool shape
    )
    return tuple(anthropic_tools)


def _map_function_tools(
    tools: object | None,
) -> tuple[Mapping[str, object], ...] | None | VertexGeminiCountContextError:
    try:
        tool_values: Final = _TOOLS.validate_python(tools or ())
    except ValidationError:
        return _INVALID_TOOLS
    function_tools: Final = tuple(tool for tool in tool_values if _is_function_tool(tool))
    if not function_tools:
        return None
    if not all(_has_object_schema(tool) for tool in function_tools):
        return _INVALID_TOOLS

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    chat_tools: Final = tuple(
        chat_tool for tool in function_tools for chat_tool in _chat_tool(tool)
    )  # comprehension-ok: flattens each tool's translation
    try:
        mapped_tools: Final = VertexGeminiConfig().map_function(
            value=[dict(tool) for tool in chat_tools],
            optional_params={},
        )
    except (AttributeError, TypeError, ValueError):
        return _INVALID_TOOLS
    return tuple(
        MappingProxyType({_VERTEX_TOOL_KEYS.get(key, key): value for key, value in tool.items()})
        for tool in mapped_tools
    )


def vertex_gemini_count_context(
    system: object | None,
    tools: object | None,
) -> VertexGeminiCountContext | VertexGeminiCountContextError:
    system_parts: Final = _system_parts(system)
    if isinstance(system_parts, VertexGeminiCountContextError):
        return system_parts
    function_tools: Final = _map_function_tools(tools)
    if isinstance(function_tools, VertexGeminiCountContextError):
        return function_tools
    return VertexGeminiCountContext(
        system_instruction=MappingProxyType({"parts": system_parts}) if system_parts else None,
        tools=function_tools,
    )


def count_context_error_response(
    error: VertexGeminiCountContextError, request_model: str, model_used: str
) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=0,
        request_model=request_model,
        model_used=model_used,
        tokenizer_type="gemini",
        error=True,
        error_message=error.error_message,
        status_code=400,
    )
