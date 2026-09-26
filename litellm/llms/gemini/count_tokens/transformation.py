import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

from pydantic import TypeAdapter

from litellm.llms.anthropic.common_utils import sanitize_replayed_anthropic_messages
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import AnthropicAdapter
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.common_utils import get_supports_system_message
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,  # pyright: ignore[reportPrivateUsage]  # shared chat-path converter
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # shared chat-path system splitter
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.vertex_ai import ContentType, PartType, SystemInstructions, Tools
from litellm.types.utils import AllMessageValues


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: tuple[ContentType, ...]
    system_instruction: SystemInstructions | None
    tools: tuple[Tools, ...] | None


@dataclass(frozen=True, slots=True)
class NativeCountTokensBody:
    contents: list[dict[str, object]]  # mutable-ok: TokenCountRequest takes list fields
    tools: list[dict[str, object]] | None  # mutable-ok: TokenCountRequest takes list fields
    system_instruction: object | None


_JSON_OBJECT_LIST: Final = TypeAdapter(list[dict[str, object]])


def parse_native_count_tokens_body(body: Mapping[str, object]) -> NativeCountTokensBody:
    wrapped_body: Final = body.get("generateContentRequest")
    wrapped: Final[Mapping[str, object]] = wrapped_body if isinstance(wrapped_body, Mapping) else {}
    tools: Final = body.get("tools") or wrapped.get("tools")
    return NativeCountTokensBody(
        contents=_JSON_OBJECT_LIST.validate_python(body.get("contents") or wrapped.get("contents") or []),
        tools=None if tools is None else _JSON_OBJECT_LIST.validate_python(tools),
        system_instruction=body.get("systemInstruction") or wrapped.get("systemInstruction"),
    )


@dataclass(frozen=True, slots=True)
class InvalidCountTokensRequest:
    message: str


_ANTHROPIC_ADAPTER: Final = AnthropicAdapter()

_ANTHROPIC_PART_TYPES: Final = frozenset(
    {
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "container_upload",
    }
)

_ANTHROPIC_TOOL_TYPE_NAMES: Final = frozenset({"mcp_toolset"})
_ANTHROPIC_DATED_TOOL_TYPE_RE: Final = re.compile(
    r"^(web_search|web_fetch|code_execution|computer|text_editor|bash|mcp_toolset)_\d{8}$"
)

_GEMINI_TOOL_KEYS: Final = frozenset(
    {
        "function_declarations",
        "functionDeclarations",
        "googleSearch",
        "google_search",
        "urlContext",
        "url_context",
        "codeExecution",
        "code_execution",
        "enterpriseWebSearch",
        "googleSearchRetrieval",
        "retrieval",
        "computerUse",
        "computer_use",
    }
)


def _has_anthropic_shape(
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
    messages: Sequence[Mapping[str, object]] | None,
) -> bool:
    if isinstance(system, list):
        return True
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") in _ANTHROPIC_PART_TYPES:
                    return True
    for tool in tools or ():
        if isinstance(tool, Mapping):
            if "input_schema" in tool:
                return True
            tool_type = tool.get("type")
            if isinstance(tool_type, str) and (
                tool_type in _ANTHROPIC_TOOL_TYPE_NAMES or _ANTHROPIC_DATED_TOOL_TYPE_RE.match(tool_type)
            ):
                return True
    return False


_ANTHROPIC_HOSTED_TOOL_TO_GEMINI: Final = MappingProxyType(
    {
        "web_search": "googleSearch",
        "web_fetch": "urlContext",
        "code_execution": "codeExecution",
    }
)


def _is_gemini_tool_shape(tool: Mapping[str, object]) -> bool:
    return any(key in tool for key in _GEMINI_TOOL_KEYS)


def _native_hosted_tool(tool: Mapping[str, object]) -> Tools | None:
    tool_type: Final = tool.get("type")
    if not isinstance(tool_type, str):
        return None
    dated: Final = _ANTHROPIC_DATED_TOOL_TYPE_RE.match(tool_type)
    gemini_key: Final = _ANTHROPIC_HOSTED_TOOL_TO_GEMINI.get(dated.group(1) if dated else tool_type)
    return cast(Tools, {gemini_key: {}}) if gemini_key else None  # cast-ok: single-key Gemini hosted-tool shape


def _as_openai_tool(tool: Mapping[str, object]) -> Mapping[str, object]:
    if "input_schema" not in tool:
        return tool
    return MappingProxyType(
        {
            "type": "function",
            "function": MappingProxyType(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema"),
                }
            ),
        }
    )


def _tool_params(
    tools: Sequence[Mapping[str, object]],
    web_search_options: object | None,
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: value
            for key, value in (("tools", tuple(tools) or None), ("web_search_options", web_search_options))
            if value is not None
        }
    )


def _openai_tool_params(tools: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return _tool_params(tools=tuple(_as_openai_tool(tool) for tool in tools), web_search_options=None)


_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_ARRAY: Final = TypeAdapter(list[object])


def _json_object_copy(value: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: adapter pops keys
    return _JSON_OBJECT.validate_json(json.dumps(value, default=dict))


def _json_array_copy(value: Sequence[object]) -> list[object]:  # mutable-ok: the chat converters take plain lists
    return _JSON_ARRAY.validate_json(json.dumps(value, default=dict))


def _gemini_tools_like_chat(model: str, tool_params: Mapping[str, object]) -> tuple[Tools, ...] | None:
    if not tool_params:
        return None
    optional_params: Final = GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params=_json_object_copy(tool_params),
        optional_params={},  # mutable-ok: the chat mapper writes the Gemini tools into it
        model=model,
        drop_params=False,
    )
    tools: Final = cast(  # cast-ok: the chat mapper stores Gemini Tools under "tools"
        "list[Tools] | None", optional_params.get("tools")
    )
    return tuple(tools) if tools else None


def _apply_mixed_tool_drop_rule(merged: Sequence[Tools]) -> tuple[Tools, ...] | None:
    if not merged:
        return None
    optional_params: Final = {"tools": list(merged)}  # mutable-ok: the shared drop rule rewrites this dict in place
    VertexGeminiConfig._drop_search_tools_mixed_with_functions(optional_params)  # pyright: ignore[reportPrivateUsage]  # shared chat-path drop rule
    kept: Final = optional_params["tools"]
    return tuple(kept) if kept else None


def _native_tools(model: str, tools: Sequence[Mapping[str, object]] | None) -> tuple[Tools, ...] | None:
    if not tools:
        return None
    passthrough: Final = tuple(
        cast(Tools, tool)  # cast-ok: Gemini wire-shape tools pass through
        for tool in tools
        if _is_gemini_tool_shape(tool)
    )
    translated: Final = tuple(_native_hosted_tool(tool) for tool in tools)
    hosted: Final = tuple(hosted_tool for hosted_tool in translated if hosted_tool is not None)
    mapped: Final = _gemini_tools_like_chat(
        model=model,
        tool_params=_openai_tool_params(
            tuple(
                tool
                for tool, hosted_tool in zip(tools, translated)
                if not _is_gemini_tool_shape(tool) and hosted_tool is None
            )
        ),
    )
    return _apply_mixed_tool_drop_rule(passthrough + hosted + (mapped or ()))


def _payload_like_chat(
    model: str,
    messages: Sequence[object],
    tool_params: Mapping[str, object],
) -> GeminiCountTokensPayload:
    system_instruction, chat_messages = _transform_system_message(
        supports_system_message=get_supports_system_message(model=model, custom_llm_provider="gemini"),
        messages=cast(  # cast-ok: chat-shaped message dicts, copied so the converters cannot touch the caller's request
            "list[AllMessageValues]", _json_array_copy(messages)
        ),
    )
    return GeminiCountTokensPayload(
        contents=tuple(
            _gemini_convert_messages_with_history(messages=chat_messages, model=model, custom_llm_provider="gemini")
        ),
        system_instruction=system_instruction,
        tools=_gemini_tools_like_chat(model=model, tool_params=tool_params),
    )


def _build_anthropic_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload | InvalidCountTokensRequest:
    request: Final = _json_object_copy(
        MappingProxyType(
            {key: value for key, value in (("model", model), ("system", system), ("tools", tools)) if value}
        )
    )
    openai_request, _ = _ANTHROPIC_ADAPTER.translate_completion_input_params_with_tool_mapping(
        {  # mutable-ok: the adapter pops model and messages out of the dict it is given
            **request,
            "messages": sanitize_replayed_anthropic_messages(_json_array_copy(messages)),
        },
        custom_llm_provider="gemini",
    )
    if openai_request is None:
        return InvalidCountTokensRequest(message="Anthropic request could not be translated for Gemini")
    return _payload_like_chat(
        model=model,
        messages=openai_request["messages"],
        tool_params=_tool_params(
            tools=openai_request.get("tools") or (),
            web_search_options=openai_request.get("web_search_options"),
        ),
    )


def _build_openai_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    return _payload_like_chat(
        model=model,
        messages=(
            (MappingProxyType({"role": "system", "content": system}), *messages) if system is not None else messages
        ),
        tool_params=_openai_tool_params(tools or ()),
    )


def _native_system_instruction(system: object | None) -> SystemInstructions | None:
    if not isinstance(system, str):
        return cast("SystemInstructions | None", system)  # cast-ok: the native route passes Gemini's systemInstruction
    part: Final[PartType] = {"text": system}
    instruction: Final[SystemInstructions] = {"parts": [part]}  # mutable-ok: parts is Required[list]
    return instruction


def native_count_tokens_payload(
    model: str,
    contents: Sequence[object],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    return GeminiCountTokensPayload(
        contents=cast("tuple[ContentType, ...]", tuple(contents)),  # cast-ok: Gemini-native contents pass through
        system_instruction=_native_system_instruction(system),
        tools=_native_tools(model=model, tools=tools),
    )


def build_count_tokens_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload | InvalidCountTokensRequest:
    try:
        if _has_anthropic_shape(system=system, tools=tools, messages=messages):
            return _build_anthropic_payload(model=model, messages=messages, system=system, tools=tools)
        return _build_openai_payload(model=model, messages=messages, system=system, tools=tools)
    except Exception as e:  # noqa: BLE001  # translation is pure; any failure is untranslatable input
        return InvalidCountTokensRequest(message=f"Invalid token count request: {e!r}")
