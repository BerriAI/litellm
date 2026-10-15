from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.types.llms.anthropic import AnthropicMessagesTextParam
from litellm.types.utils import TokenCountResponse


class _AnthropicFunctionTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    description: str | None = None
    input_schema: Mapping[str, JsonValue]


class _OpenAIFunction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    description: str | None = None
    parameters: Mapping[str, JsonValue] | None = None


class _OpenAIChatFunctionTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["function"]
    function: _OpenAIFunction


class _ResponsesFunctionTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class VertexGeminiCountContext:
    system_instruction: Mapping[str, object] | None
    tools: tuple[Mapping[str, object], ...] | None


@dataclass(frozen=True, slots=True)
class VertexGeminiCountContextError:
    error_message: str


_SYSTEM_BLOCKS: Final = TypeAdapter(tuple[AnthropicMessagesTextParam, ...])
_TOOL_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_TOOL_MAPPING: Final = TypeAdapter(dict[str, JsonValue])
_ANTHROPIC_FUNCTION_TOOL: Final = TypeAdapter(_AnthropicFunctionTool)
_OPENAI_CHAT_FUNCTION_TOOL: Final = TypeAdapter(_OpenAIChatFunctionTool)
_RESPONSES_FUNCTION_TOOL: Final = TypeAdapter(_ResponsesFunctionTool)
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


def _openai_function_tool(
    name: str,
    description: str | None,
    parameters: Mapping[str, JsonValue] | None,
) -> Mapping[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            **({"description": description} if description is not None else {}),
            **({"parameters": deepcopy(dict(parameters))} if parameters is not None else {}),
        },
    }


def _normalize_function_tool(tool: object) -> Mapping[str, object] | None:
    try:
        tool_mapping: Final = _TOOL_MAPPING.validate_python(tool)
    except ValidationError:
        return None

    if "function" in tool_mapping:
        try:
            chat_tool: Final = _OPENAI_CHAT_FUNCTION_TOOL.validate_python(tool_mapping)
        except ValidationError:
            return None
        return _openai_function_tool(
            name=chat_tool.function.name,
            description=chat_tool.function.description,
            parameters=chat_tool.function.parameters,
        )

    if "input_schema" in tool_mapping:
        try:
            anthropic_tool: Final = _ANTHROPIC_FUNCTION_TOOL.validate_python(tool_mapping)
        except ValidationError:
            return None
        from litellm.llms.anthropic.pass_through.adapters.transformation import truncate_tool_name

        return _openai_function_tool(
            name=truncate_tool_name(anthropic_tool.name),
            description=anthropic_tool.description,
            parameters=anthropic_tool.input_schema,
        )

    if tool_mapping.get("type") == "function" and "name" in tool_mapping:
        try:
            responses_tool: Final = _RESPONSES_FUNCTION_TOOL.validate_python(tool_mapping)
        except ValidationError:
            return None
        return _openai_function_tool(
            name=responses_tool.name,
            description=responses_tool.description,
            parameters=responses_tool.parameters,
        )

    return None


def _map_function_tools(tools: object | None) -> tuple[Mapping[str, object], ...] | None:
    if tools is None:
        return None
    try:
        tool_values: Final = _TOOL_SEQUENCE.validate_python(tools)
    except ValidationError:
        return None
    normalized_tools: Final = tuple(
        normalized_tool for tool in tool_values if (normalized_tool := _normalize_function_tool(tool)) is not None
    )
    if not normalized_tools:
        return None

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    mapped_tools: Final = VertexGeminiConfig().map_function(
        value=[dict(tool) for tool in normalized_tools],
        optional_params={},
    )
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
    return VertexGeminiCountContext(
        system_instruction=MappingProxyType({"parts": system_parts}) if system_parts else None,
        tools=_map_function_tools(tools),
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
