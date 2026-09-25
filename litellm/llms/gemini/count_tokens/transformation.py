"""Translate a token-count request into a Gemini countTokens payload
(contents + systemInstruction + tools).

Callers send Anthropic Messages shapes (/v1/messages/count_tokens) or
already-OpenAI shapes (/v1/responses/input_tokens, /utils/token_counter).
OpenAI input skips the Anthropic adapter, which only reads Anthropic fields
and would drop tool_calls, tool messages, and web search tool declarations.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,  # pyright: ignore[reportPrivateUsage]  # shared helper already used by gemini/chat, context_caching, and vertex_and_google_ai_studio_gemini
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # same helper the Gemini chat transformation uses to split system prompts
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools
from litellm.types.utils import AllMessageValues


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: list[ContentType]
    system_instruction: SystemInstructions | None
    tools: list[Tools] | None


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

# Anthropic hosted tools are date-versioned (web_search_20250305). OpenAI
# names like web_search_preview or computer_use must not match, so prefixes
# are excluded unless they carry a date suffix.
_ANTHROPIC_TOOL_TYPE_NAMES: Final = frozenset(
    {"web_search", "web_fetch", "code_execution", "computer", "text_editor", "bash", "mcp_toolset"}
)
_ANTHROPIC_TOOL_TYPE_RE: Final = re.compile(
    r"^(web_search|web_fetch|code_execution|computer|text_editor|bash|mcp_toolset)_\d{8}$"
)

# Server-side Anthropic content blocks the anthropic->openai adapter drops, so
# they would silently count as ~1 token each. They are flattened to text
# (closest token mass to the serialized block Anthropic itself bills).
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

# Anthropic hosted tools with a native Gemini equivalent: mapping preserves
# roughly the hosted-tool token overhead instead of collapsing them to a
# name-only function declaration.
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
    # A top-level system given as a list of blocks only exists in the Anthropic API
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
                tool_type in _ANTHROPIC_TOOL_TYPE_NAMES or _ANTHROPIC_TOOL_TYPE_RE.match(tool_type)
            ):
                return True
    return False


def _is_web_search_tool(tool: Mapping[str, object]) -> bool:
    tool_type = tool.get("type")
    return isinstance(tool_type, str) and tool_type.startswith("web_search")


def _is_gemini_tool_shape(tool: Mapping[str, object]) -> bool:
    return any(key in tool for key in _GEMINI_TOOL_KEYS)


def _hosted_tool_type(tool: Mapping[str, object]) -> str | None:
    tool_type = tool.get("type")
    if not isinstance(tool_type, str):
        return None
    for prefix, gemini_name in _ANTHROPIC_HOSTED_TOOL_TYPES:
        if tool_type.startswith(prefix):
            return gemini_name
    return None


def _normalize_openai_tool(tool: Mapping[str, object]) -> dict[str, object]:
    if "function" in tool:
        return dict(tool)  # mutable-ok: _map_function takes plain tool dicts
    if "input_schema" in tool:
        return {  # mutable-ok: normalized tool dict for _map_function (anthropic function shape reaching the openai path)
            "type": "function",
            "function": {  # mutable-ok: normalized tool dict for _map_function
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("input_schema"),
            },
        }
    if tool.get("type") == "function" and "name" in tool:
        return {  # mutable-ok: normalized tool dict for _map_function (responses-api flat shape)
            "type": "function",
            "function": {  # mutable-ok: normalized tool dict for _map_function
                key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool
            },
        }
    return dict(tool)  # mutable-ok: _map_function takes plain tool dicts


def _apply_mixed_tool_drop_rule(merged: Sequence[Tools]) -> list[Tools] | None:
    if not merged:
        return None
    optional_params: Final = {  # mutable-ok: shared Vertex drop-rule mutates the tools list in place
        "tools": list(merged)  # mutable-ok: shared Vertex drop-rule mutates the tools list in place
    }
    VertexGeminiConfig._drop_search_tools_mixed_with_functions(optional_params)  # pyright: ignore[reportPrivateUsage]  # same drop rule the Vertex chat path applies before counting
    kept: Final = optional_params["tools"]
    return kept or None


def _map_to_gemini_tools(
    openai_tools: Sequence[Mapping[str, object]],
    web_search_options: object | None,
) -> list[Tools] | None:
    merged: Final = (
        VertexGeminiConfig()._map_function(  # pyright: ignore[reportPrivateUsage]  # same tool mapper the Gemini chat path uses
            value=[dict(tool) for tool in openai_tools],  # mutable-ok: _map_function takes plain tool dicts
            optional_params={},  # mutable-ok: _map_function signature takes a dict
        )
        if openai_tools
        else []  # mutable-ok: merged with the mapped tools list below
    ) + (
        [VertexGeminiConfig()._map_web_search_options({})]  # pyright: ignore[reportPrivateUsage]  # same web-search mapper the Vertex chat path uses  # mutable-ok: merged tools list for the drop-rule
        if web_search_options is not None
        else []  # mutable-ok: merged with the mapped tools list
    )
    return _apply_mixed_tool_drop_rule(merged)


def normalize_count_tokens_tools(
    tools: Sequence[Mapping[str, object]] | None,
) -> list[Tools] | None:
    """Tools arriving alongside native Gemini contents may be in Gemini,
    OpenAI, Responses-API, or Anthropic shape. Gemini-shaped entries pass
    through; the rest are normalized so mixed-shape callers do not 400."""
    if not tools:
        return None
    gemini_shaped: Final = tuple(tool for tool in tools if _is_gemini_tool_shape(tool))
    rest: Final = tuple(tool for tool in tools if not _is_gemini_tool_shape(tool))
    mapped: Final = _map_to_gemini_tools(
        openai_tools=tuple(_normalize_openai_tool(tool) for tool in rest if not _is_web_search_tool(tool)),
        web_search_options={}  # mutable-ok: truthy marker for _map_web_search_options
        if any(_is_web_search_tool(tool) for tool in rest)
        else None,
    )
    merged: Final = (
        [
            cast(  # cast-ok: plain dicts for the Tools wire shape
                Tools,
                dict(tool),  # mutable-ok: plain dicts for the Tools wire shape
            )
            for tool in gemini_shaped
        ]
        + list(mapped or [])  # mutable-ok: concat with mapped tools
    )
    return _apply_mixed_tool_drop_rule(merged)


def _dedupe_thought_signature_parts(contents: Sequence[ContentType]) -> list[ContentType]:
    """The openai->gemini converter emits both a {thought: true, text} part
    and a duplicate {thoughtSignature, text} part for one signed thinking
    block; the signature belongs as a field on the thought part, not a second
    part, or every reasoning turn is counted twice."""

    def _merge(parts: Sequence[object]) -> Sequence[object]:
        sig_by_text: Final = {  # mutable-ok: text->signature lookup
            part.get("text"): part.get("thoughtSignature")
            for part in parts
            if isinstance(part, Mapping)
            and part.get("thoughtSignature") is not None
            and part.get("thought") is not True
        }
        if not sig_by_text:
            return parts
        thought_texts: Final = frozenset(
            part.get("text") for part in parts if isinstance(part, Mapping) and part.get("thought") is True
        )
        return tuple(
            (
                {  # mutable-ok: rebuilt part with merged signature
                    **dict(part),  # mutable-ok: rebuilt part with merged signature
                    "thoughtSignature": sig_by_text[part.get("text")],
                }
                if isinstance(part, Mapping)
                and part.get("thought") is True
                and sig_by_text.get(part.get("text")) is not None
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

    return [  # mutable-ok: rebuilt contents list
        cast(  # cast-ok: same ContentType shape with deduped parts
            ContentType,
            {  # mutable-ok: rebuilt content with deduped parts
                **dict(content),  # mutable-ok: rebuilt content with deduped parts
                "parts": _merge(content.get("parts", [])),  # mutable-ok: default parts list for the merge
            },
        )
        if isinstance(content.get("parts"), list)
        else content
        for content in contents
    ]


def _textify_server_side_blocks(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Flatten server-side Anthropic content blocks to text so the
    anthropic->openai adapter (which drops them) still counts their mass."""

    def _textify(content: object) -> object:
        if not isinstance(content, list):
            return content
        return [  # mutable-ok: rebuilt content list
            (
                {  # mutable-ok: textified block for counting
                    "type": "text",
                    "text": json.dumps(dict(block), ensure_ascii=False),  # mutable-ok: plain dict copy for the dump
                }
                if isinstance(block, Mapping) and block.get("type") in _SERVER_SIDE_PART_TYPES
                else block
            )
            for block in content
        ]

    return tuple(
        (
            {**dict(message), "content": _textify(message.get("content"))}  # mutable-ok: rebuilt message
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
    contents: Final = _dedupe_thought_signature_parts(
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
    raw_request: Final = {  # mutable-ok: transient request dict for the anthropic adapter
        "model": model,
        "messages": list(  # mutable-ok: adapter contract takes a list of messages
            _textify_server_side_blocks(messages)
        ),
        **({"system": system} if system else {}),  # mutable-ok: transient request dict for the anthropic adapter
        **(
            {"tools": list(adapter_tools)}
            if adapter_tools
            else {}  # mutable-ok: transient request dict for the anthropic adapter
        ),
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
    merged_tools: Final = _apply_mixed_tool_drop_rule(
        list(payload.tools or [])  # mutable-ok: merged tools for the drop-rule
        + [  # mutable-ok: merged tools list for the drop-rule
            cast(Tools, {gemini_name: {}})  # cast-ok: hosted tool wire shape  # mutable-ok: hosted tool wire shape
            for gemini_name in {_hosted_tool_type(tool) for tool in hosted_tools}  # mutable-ok: set dedupe
        ]
    )
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
        [{"role": "system", "content": system}, *messages]  # mutable-ok: transient list for the message converter
        if system is not None
        else list(messages)  # mutable-ok: transient list for the message converter
    )
    return _payload_from_openai_parts(
        model=model,
        messages=openai_messages,
        tools=tuple(_normalize_openai_tool(tool) for tool in tools or () if not _is_web_search_tool(tool)),
        web_search_options={}  # mutable-ok: truthy marker for _map_web_search_options
        if any(_is_web_search_tool(tool) for tool in tools or ())
        else None,
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


# Matches real inlineData blobs; a short or non-base64 `data` field (tool args,
# function responses) stays text so the fallback count keeps its mass.
_BASE64_BLOB_RE: Final = re.compile(r"[A-Za-z0-9+/=]{16,}")


def _elide_data_key(obj: dict[str, object]) -> dict[str, object]:
    """json.loads object_hook that replaces base64 blobs (inlineData.data)
    so serialized parts stay a sane size for the local tokenizer."""
    return {  # mutable-ok: object_hook contract returns a rebuilt object per JSON node
        key: ("<binary>" if key == "data" and isinstance(value, str) and _BASE64_BLOB_RE.fullmatch(value) else value)
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


def gemini_contents_as_chat_messages(contents: object) -> tuple[Mapping[str, object], ...] | None:
    """Approximate gemini contents as chat messages for the local fallback
    tokenizer. Text parts count as text; other parts count as their JSON
    frame with base64 blobs elided."""
    if contents is None:
        return None
    if isinstance(contents, list):
        messages: Final = tuple(
            {  # mutable-ok: transient chat-shaped message for the local tokenizer
                "role": "assistant" if content.get("role") == "model" else "user",
                "content": "\n".join(_part_to_text(part) for part in _content_parts(content)),
            }
            for content in contents
            if isinstance(content, Mapping)
        )
        counted: Final = tuple(message for message in messages if message["content"])
        if counted:
            return counted
    fallback: Final[tuple[Mapping[str, object], ...]] = (
        {  # mutable-ok: transient chat-shaped message for the local tokenizer
            "role": "user",
            "content": _serialize_part(contents),
        },
    )
    return fallback
