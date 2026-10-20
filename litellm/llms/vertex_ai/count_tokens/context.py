import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.litellm_core_utils.prompt_templates.factory import chat_completion_tools_for
from litellm.types.llms.anthropic import AnthropicMessagesTextParam
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import ContentType
from litellm.types.utils import TokenCountMessagesFormat, TokenCountResponse

if TYPE_CHECKING:
    from litellm.exceptions import APIError
    from litellm.llms.anthropic.pass_through.context_management.result import PolyfillResult


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
        chat_tool for tool in function_tools for chat_tool in chat_completion_tools_for(tool)
    )  # comprehension-ok: flattens each tool's translation
    try:
        mapped_tools: Final = VertexGeminiConfig()._map_function(  # pyright: ignore[reportPrivateUsage]  # the tool mapping inference uses
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


def unconvertible_messages_error(
    messages: Sequence[Mapping[str, object]] | None, messages_format: TokenCountMessagesFormat
) -> VertexGeminiCountContextError | None:
    from litellm.llms.anthropic.pass_through.convertible_blocks import unconvertible_blocks_error

    error_message: Final = unconvertible_blocks_error(messages, messages_format)
    return None if error_message is None else VertexGeminiCountContextError(error_message=error_message)


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
        rejected_before_provider_call=True,
    )


def provider_error_response(error: "APIError", request_model: str, model_used: str) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=0,
        request_model=request_model,
        model_used=model_used,
        tokenizer_type="gemini",
        error=True,
        error_message=error.message,
        status_code=error.status_code,
    )


@dataclass(frozen=True, slots=True)
class InferenceRequestView:
    messages: Sequence[Mapping[str, object]]
    system: object | None


def _client_compaction(messages: Sequence[Mapping[str, object]], system: object | None) -> "PolyfillResult | None":
    from litellm.llms.anthropic.pass_through.context_management.editors.compact import (
        apply_client_compaction_block_history,
    )

    chat_messages: Final = [dict(message) for message in messages]
    if system is None or isinstance(system, str):
        return apply_client_compaction_block_history(messages=chat_messages, system=system)
    try:
        blocks: Final = _SYSTEM_BLOCKS.validate_python(system)
    except ValidationError:
        return None
    return apply_client_compaction_block_history(messages=chat_messages, system=[dict(block) for block in blocks])


def inference_request_view(
    messages: Sequence[Mapping[str, object]], system: object | None, messages_format: TokenCountMessagesFormat
) -> InferenceRequestView:
    """Apply the history rewrites /v1/messages runs before it converts a body for Gemini."""
    from litellm.llms.anthropic.pass_through.messages.utils import normalize_native_messages

    match messages_format:
        case "openai":
            return InferenceRequestView(messages=messages, system=system)
        case "anthropic":
            normalized: Final = normalize_native_messages(messages)
            compacted: Final = _client_compaction(normalized, system)
            if compacted is None:
                return InferenceRequestView(messages=normalized, system=system)
            return InferenceRequestView(messages=compacted.messages, system=compacted.system)
        case _:
            assert_never(messages_format)


def _chat_messages(
    messages: Sequence[Mapping[str, object]],
    messages_format: TokenCountMessagesFormat,
    model: str,
    custom_llm_provider: Literal["vertex_ai", "gemini"],
) -> Sequence[AllMessageValues]:
    from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter

    match messages_format:
        case "anthropic":
            return LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
                messages=list(messages),  # pyright: ignore[reportArgumentType]  # raw Anthropic body, read the way /v1/messages inference reads it
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        case "openai":
            return copy.deepcopy(list(messages))  # pyright: ignore[reportReturnType]  # chat messages from the request, copied because the AI Studio conversion rewrites them in place
        case _:
            assert_never(messages_format)


async def gemini_count_contents(
    messages: Sequence[Mapping[str, object]],
    messages_format: TokenCountMessagesFormat,
    model: str,
    custom_llm_provider: Literal["vertex_ai", "gemini"],
    litellm_params: Mapping[str, object],
) -> Sequence[ContentType] | VertexGeminiCountContextError:
    from litellm.exceptions import BadRequestError
    from litellm.litellm_core_utils.asyncify import asyncify
    from litellm.litellm_core_utils.prompt_templates.image_handling import async_inline_remote_media
    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
    from litellm.llms.vertex_ai.gemini.transformation import should_inline_remote_media
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    config: Final = GoogleAIStudioGeminiConfig() if custom_llm_provider == "gemini" else VertexGeminiConfig()
    try:
        chat_messages: Final = await async_inline_remote_media(
            list(_chat_messages(messages, messages_format, model, custom_llm_provider)),
            should_inline=should_inline_remote_media(custom_llm_provider),
        )
        return await asyncify(config._transform_messages)(  # pyright: ignore[reportPrivateUsage]  # the message transform inference runs
            messages=chat_messages, model=model, litellm_params=dict(litellm_params)
        )
    except BadRequestError as error:
        return VertexGeminiCountContextError(error_message=error.message)
