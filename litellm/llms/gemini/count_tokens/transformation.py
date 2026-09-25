import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypedDict, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,  # pyright: ignore[reportPrivateUsage]  # shared chat-path converter
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # shared chat-path system splitter
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools
from litellm.types.utils import AllMessageValues


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: tuple[ContentType, ...]
    system_instruction: SystemInstructions | None
    tools: tuple[Tools, ...] | None


@dataclass(frozen=True, slots=True)
class InvalidCountTokensRequest:
    message: str


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
        "mcp_tool_use",
        "mcp_tool_result",
        "container_upload",
    }
)

_ANTHROPIC_TOOL_TYPE_NAMES: Final = frozenset(
    {"web_search", "web_fetch", "code_execution", "computer", "text_editor", "bash", "mcp_toolset"}
)
_ANTHROPIC_DATED_TOOL_TYPE_RE: Final = re.compile(
    r"^(web_search|web_fetch|code_execution|computer|text_editor|bash|mcp_toolset)_\d{8}$"
)

_SERVER_SIDE_PART_TYPES: Final = frozenset(
    {
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "code_execution_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "container_upload",
        "redacted_thinking",
    }
)

_ANTHROPIC_HOSTED_TOOL_TYPES: Final = (
    ("code_execution", "codeExecution"),
    ("web_fetch", "urlContext"),
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


def _is_web_search_tool(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type")
    return isinstance(tool_type, str) and tool_type.startswith("web_search")


def _is_gemini_tool_shape(tool: Mapping[str, object]) -> bool:
    return any(key in tool for key in _GEMINI_TOOL_KEYS)


def _hosted_tool_type(tool: Mapping[str, object]) -> str | None:
    tool_type: Final = tool.get("type")
    if not isinstance(tool_type, str):
        return None
    for prefix, gemini_name in _ANTHROPIC_HOSTED_TOOL_TYPES:
        if tool_type.startswith(prefix):
            return gemini_name
    return None


def _normalize_openai_tool(tool: Mapping[str, object]) -> Mapping[str, object]:
    if "function" in tool:
        return tool
    if "input_schema" in tool:
        function: Final = {  # mutable-ok: _map_function mutates nested tool dicts
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("input_schema"),
        }
        return {  # mutable-ok: _map_function mutates nested tool dicts
            "type": "function",
            "function": function,
        }
    if tool.get("type") == "function" and "name" in tool:
        flat_function: Final = {  # mutable-ok: _map_function mutates nested tool dicts
            key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool
        }
        return {  # mutable-ok: _map_function mutates nested tool dicts
            "type": "function",
            "function": flat_function,
        }
    return tool


def _apply_mixed_tool_drop_rule(merged: Sequence[Tools]) -> tuple[Tools, ...] | None:
    if not merged:
        return None
    tools_list: Final = list(merged)  # mutable-ok: shared drop rule rewrites the list
    optional_params: Final = {"tools": tools_list}  # mutable-ok: shared drop rule takes a dict
    VertexGeminiConfig._drop_search_tools_mixed_with_functions(optional_params)  # pyright: ignore[reportPrivateUsage]  # shared chat-path drop rule
    kept: Final = optional_params["tools"]
    return tuple(kept) if kept else None


def _map_to_gemini_tools(
    openai_tools: Sequence[Mapping[str, object]],
    web_search_options: object | None,
) -> tuple[Tools, ...] | None:
    mapped_functions: Final = tuple(
        VertexGeminiConfig()._map_function(  # pyright: ignore[reportPrivateUsage]  # shared chat-path tool mapper
            value=[dict(tool) for tool in openai_tools],  # mutable-ok: _map_function mutates plain tool dicts
            optional_params={},  # mutable-ok: _map_function writes toolConfig into it
        )
        if openai_tools
        else ()
    )
    mapped_search: Final = (
        (VertexGeminiConfig()._map_web_search_options({}),)  # pyright: ignore[reportPrivateUsage]  # shared chat-path web-search mapper  # mutable-ok: signature takes a dict
        if web_search_options is not None
        else ()
    )
    return _apply_mixed_tool_drop_rule(mapped_functions + mapped_search)


def normalize_count_tokens_tools(
    tools: Sequence[Mapping[str, object]] | None,
) -> tuple[Tools, ...] | None:
    if not tools:
        return None
    gemini_shaped: Final = tuple(tool for tool in tools if _is_gemini_tool_shape(tool))
    rest: Final = tuple(tool for tool in tools if not _is_gemini_tool_shape(tool))
    mapped: Final = _map_to_gemini_tools(
        openai_tools=tuple(_normalize_openai_tool(tool) for tool in rest if not _is_web_search_tool(tool)),
        web_search_options=MappingProxyType({}) if any(_is_web_search_tool(tool) for tool in rest) else None,
    )
    passthrough: Final = tuple(
        cast(Tools, tool)  # cast-ok: Gemini wire-shape tools pass through
        for tool in gemini_shaped
    )
    return _apply_mixed_tool_drop_rule(passthrough + tuple(mapped or ()))


def _merge_signature_into_parts(parts: Sequence[object]) -> Sequence[object]:
    signature_by_text: Final = MappingProxyType(
        {
            part.get("text"): part.get("thoughtSignature")
            for part in parts
            if isinstance(part, Mapping)
            and part.get("thoughtSignature") is not None
            and part.get("thought") is not True
        }
    )
    if not signature_by_text:
        return parts
    thought_texts: Final = frozenset(
        part.get("text") for part in parts if isinstance(part, Mapping) and part.get("thought") is True
    )
    return tuple(
        (
            {**part, "thoughtSignature": signature_by_text[part.get("text")]}  # mutable-ok: parts serialize to JSON
            if isinstance(part, Mapping)
            and part.get("thought") is True
            and signature_by_text.get(part.get("text")) is not None
            else part
        )
        for part in parts
        if not (
            isinstance(part, Mapping)
            and part.get("thoughtSignature") is not None
            and part.get("thought") is not True
            and part.get("text") in thought_texts
        )
    )


def _merge_duplicate_thought_signature_parts(contents: Sequence[ContentType]) -> tuple[ContentType, ...]:
    return tuple(
        cast(  # cast-ok: same ContentType shape with deduped parts
            ContentType,
            {  # mutable-ok: rebuilt content for the request body
                **content,
                "parts": _merge_signature_into_parts(content["parts"]),
            },
        )
        if isinstance(content.get("parts"), list)
        else content
        for content in contents
    )


def _textify_server_side_blocks(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    def _textify(content: object) -> object:
        if not isinstance(content, list):
            return content
        return [  # mutable-ok: rebuilt message content for the adapter
            (
                {  # mutable-ok: textified block must serialize to JSON
                    "type": "text",
                    "text": json.dumps(dict(block), ensure_ascii=False),  # mutable-ok: plain dict for the dump
                }
                if isinstance(block, Mapping) and block.get("type") in _SERVER_SIDE_PART_TYPES
                else block
            )
            for block in content
        ]

    return tuple(
        (
            {**message, "content": _textify(message.get("content"))}  # mutable-ok: rebuilt message for the adapter
            if isinstance(message.get("content"), list)
            else message
        )
        for message in messages
    )


def _payload_from_openai_parts(
    model: str,
    messages: Sequence[object],
    tools: Sequence[Mapping[str, object]],
    web_search_options: object | None,
) -> GeminiCountTokensPayload:
    system_instruction, remaining_messages = _transform_system_message(
        supports_system_message=litellm.supports_system_messages(model=model, custom_llm_provider="gemini"),
        messages=cast(  # cast-ok: chat-shaped message dicts accepted by the helper
            "list[AllMessageValues]",
            list(messages),  # mutable-ok: helper contract takes a list
        ),
    )
    contents: Final = _merge_duplicate_thought_signature_parts(
        _gemini_convert_messages_with_history(
            messages=remaining_messages,
            model=model,
            custom_llm_provider="gemini",
        )
    )
    return GeminiCountTokensPayload(
        contents=contents,
        system_instruction=system_instruction,
        tools=_map_to_gemini_tools(openai_tools=tools, web_search_options=web_search_options),
    )


_ANTHROPIC_REQUEST: Final = TypeAdapter(AnthropicMessagesRequest)


def _build_anthropic_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload | InvalidCountTokensRequest:
    hosted_tools: Final = tuple(tool for tool in tools or () if _hosted_tool_type(tool) is not None)
    adapter_tools: Final = tuple(tool for tool in tools or () if _hosted_tool_type(tool) is None)
    raw_request: Final = {  # mutable-ok: request dict for the anthropic adapter
        "model": model,
        "messages": list(_textify_server_side_blocks(messages)),  # mutable-ok: adapter contract takes a list
        **({"system": system} if system else {}),  # mutable-ok: optional adapter field
        **({"tools": list(adapter_tools)} if adapter_tools else {}),  # mutable-ok: optional adapter field
    }
    try:
        _ANTHROPIC_REQUEST.validate_python(raw_request)
    except ValidationError as e:
        return InvalidCountTokensRequest(message=str(e))
    anthropic_request: Final = cast(  # cast-ok: validated above; pydantic returns lazy Iterable validators
        AnthropicMessagesRequest,
        raw_request,
    )
    openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
        anthropic_request, custom_llm_provider="gemini"
    )
    openai_tools: Final = openai_request.get("tools")
    payload: Final = _payload_from_openai_parts(
        model=model,
        messages=openai_request["messages"],
        tools=tuple(dict(tool) for tool in openai_tools)  # mutable-ok: plain dicts for the normalizer
        if openai_tools
        else (),
        web_search_options=openai_request.get("web_search_options"),
    )
    if not hosted_tools:
        return payload
    hosted_tool_names: Final = frozenset(_hosted_tool_type(tool) for tool in hosted_tools)
    hosted_tool_dicts: Final = tuple(
        cast(Tools, {gemini_name: {}})  # cast-ok: hosted tool wire shape  # mutable-ok: hosted tool wire shape
        for gemini_name in hosted_tool_names
    )
    merged_tools: Final = _apply_mixed_tool_drop_rule(tuple(payload.tools or ()) + hosted_tool_dicts)
    return GeminiCountTokensPayload(
        contents=payload.contents,
        system_instruction=payload.system_instruction,
        tools=merged_tools,
    )


def _build_openai_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    openai_messages: Final = (
        [{"role": "system", "content": system}, *messages]  # mutable-ok: converter contract takes a list
        if system is not None
        else list(messages)  # mutable-ok: converter contract takes a list
    )
    return _payload_from_openai_parts(
        model=model,
        messages=openai_messages,
        tools=tuple(_normalize_openai_tool(tool) for tool in tools or () if not _is_web_search_tool(tool)),
        web_search_options=MappingProxyType({}) if any(_is_web_search_tool(tool) for tool in tools or ()) else None,
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


_INLINE_DATA_BASE64_RE: Final = re.compile(r"[A-Za-z0-9+/=]{16,}")


def _elide_data_key(obj: Mapping[str, object]) -> Mapping[str, object]:
    return {  # mutable-ok: object_hook contract returns a dict per JSON node
        key: (
            "<binary>"
            if key == "data" and isinstance(value, str) and _INLINE_DATA_BASE64_RE.fullmatch(value)
            else value
        )
        for key, value in obj.items()
    }


def _serialize_part(part: object) -> str:
    return json.dumps(json.loads(json.dumps(part, default=str), object_hook=_elide_data_key), default=str)


def _part_to_text(part: object) -> str:
    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
        return part["text"]
    return _serialize_part(part)


def _content_parts(content: Mapping[str, object]) -> tuple[object, ...]:
    parts: Final = content.get("parts")
    if isinstance(parts, list):
        return tuple(parts)
    return (content,)


class _LocalCountMessage(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]


def _local_count_message(role: str, content: str) -> _LocalCountMessage:
    message: Final[_LocalCountMessage] = {"role": role, "content": content}
    return message


def gemini_contents_as_chat_messages(contents: object) -> tuple[Mapping[str, object], ...] | None:
    if contents is None:
        return None
    if isinstance(contents, list):
        messages: Final = tuple(
            _local_count_message(
                role="assistant" if content.get("role") == "model" else "user",
                content="\n".join(_part_to_text(part) for part in _content_parts(content)),
            )
            for content in contents
            if isinstance(content, Mapping)
        )
        counted: Final = tuple(message for message in messages if message["content"])
        if counted:
            return counted
    fallback: Final[tuple[Mapping[str, object], ...]] = (
        _local_count_message(role="user", content=_serialize_part(contents)),
    )
    return fallback
