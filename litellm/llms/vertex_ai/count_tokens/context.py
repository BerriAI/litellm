from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.types.llms.anthropic import (
    AllAnthropicPassThroughMessageValues,
    AllAnthropicToolsValues,
    AnthropicMessagesTextParam,
)


@dataclass(frozen=True, slots=True)
class AnthropicGeminiCountRequest:
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] | None


_ANTHROPIC_MESSAGES_ADAPTER: Final = TypeAdapter[tuple[AllAnthropicPassThroughMessageValues, ...]](
    tuple[AllAnthropicPassThroughMessageValues, ...]
)
_ANTHROPIC_TOOLS_ADAPTER: Final = TypeAdapter[tuple[AllAnthropicToolsValues, ...]](tuple[AllAnthropicToolsValues, ...])
_OPENAI_MESSAGES_ADAPTER: Final = TypeAdapter[tuple[Mapping[str, object], ...]](tuple[Mapping[str, object], ...])
_OPENAI_TOOLS_ADAPTER: Final = TypeAdapter[tuple[Mapping[str, object], ...]](tuple[Mapping[str, object], ...])


def _materialize_anthropic_messages(
    messages: tuple[AllAnthropicPassThroughMessageValues, ...],
) -> tuple[AllAnthropicPassThroughMessageValues, ...]:
    return tuple(
        message
        if isinstance(message["content"], (str, list))
        else cast(  # cast-ok: validation established content item types before iterator materialization
            AllAnthropicPassThroughMessageValues,
            {**message, "content": list(message["content"])},
        )
        for message in messages
    )


def convert_anthropic_gemini_count_request(
    messages: Sequence[Mapping[str, object]] | None,
    tools: Sequence[Mapping[str, object]] | None,
    model: str,
    custom_llm_provider: str,
) -> AnthropicGeminiCountRequest | None:
    if messages is None:
        return None
    try:
        validated_messages: Final = _ANTHROPIC_MESSAGES_ADAPTER.validate_python(messages)
        anthropic_messages: Final = _materialize_anthropic_messages(validated_messages)
        anthropic_tools: Final = _ANTHROPIC_TOOLS_ADAPTER.validate_python(tools) if tools is not None else None
    except ValidationError:
        return None

    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    adapter: Final = LiteLLMAnthropicMessagesAdapter()
    openai_messages: Final = _OPENAI_MESSAGES_ADAPTER.validate_python(
        adapter.translate_anthropic_messages_to_openai(  # pyright: ignore[reportUnknownMemberType]  # output is validated immediately
            messages=list(anthropic_messages),
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
    )
    openai_tools: Final = (
        _OPENAI_TOOLS_ADAPTER.validate_python(
            adapter.translate_anthropic_tools_to_openai(list(anthropic_tools), model)[0]
        )
        if anthropic_tools is not None
        else None
    )
    return AnthropicGeminiCountRequest(messages=tuple(openai_messages), tools=openai_tools)


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
    kind: Literal["context"]
    system_instruction: Mapping[str, object] | None
    tools: tuple[Mapping[str, JsonValue], ...] | None


@dataclass(frozen=True, slots=True)
class VertexGeminiCountContextError:
    kind: Literal["error"]
    error_message: str


_SYSTEM_BLOCKS: Final = TypeAdapter(tuple[AnthropicMessagesTextParam, ...])
_TOOL_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_TOOL_MAPPING: Final = TypeAdapter(dict[str, JsonValue])
_ANTHROPIC_FUNCTION_TOOL: Final = TypeAdapter(_AnthropicFunctionTool)
_OPENAI_CHAT_FUNCTION_TOOL: Final = TypeAdapter(_OpenAIChatFunctionTool)
_RESPONSES_FUNCTION_TOOL: Final = TypeAdapter(_ResponsesFunctionTool)
_MAPPED_TOOLS: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])
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
        return VertexGeminiCountContextError(
            kind="error",
            error_message="Vertex AI Gemini system instructions only support text blocks.",
        )
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
        return _openai_function_tool(
            name=anthropic_tool.name,
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


def _map_function_tools(tools: object | None) -> tuple[Mapping[str, JsonValue], ...] | None:
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

    try:
        mapped_tools: Final = VertexGeminiConfig().map_function(
            value=[deepcopy(dict(tool)) for tool in normalized_tools],
            optional_params={},
        )
        validated_tools: Final = _MAPPED_TOOLS.validate_python(mapped_tools)
    except ValidationError:
        return None
    return tuple(
        MappingProxyType({_VERTEX_TOOL_KEYS.get(key, key): value for key, value in tool.items()})
        for tool in validated_tools
    )


def vertex_gemini_count_context(
    system: object | None,
    tools: object | None,
) -> VertexGeminiCountContext | VertexGeminiCountContextError:
    system_parts: Final = _system_parts(system)
    if isinstance(system_parts, VertexGeminiCountContextError):
        return system_parts
    return VertexGeminiCountContext(
        kind="context",
        system_instruction=MappingProxyType({"parts": system_parts}) if system_parts else None,
        tools=_map_function_tools(tools),
    )
