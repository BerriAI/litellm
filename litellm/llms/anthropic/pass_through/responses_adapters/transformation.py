"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all format conversions for the direct v1/messages -> Responses API
path used for OpenAI and Azure models.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
    encrypted_reasoning_signature,
    responses_reasoning_items_from_thinking_blocks,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.llms.anthropic.common_utils import system_without_claude_code_billing_header
from litellm.llms.anthropic.pass_through.messages.utils import (
    refusal_stop_details,
    responses_output_refusal_text,
)
from litellm.llms.anthropic.pass_through.utils import (
    REASONING_EFFORT_OFF,
    content_blocks,
    is_between_tool_calls,
    is_reasoning_auto_summary_enabled,
    json_copy_or_original,
    leaves_an_object_open,
    openai_strict_json_schema,
    prompt_cache_key_from_user_id,
    requested_reasoning_summary,
)
from litellm.llms.base_llm.base_utils import as_json_object
from litellm.types.llms.anthropic import (
    AllAnthropicPassThroughMessageValues,
    AllAnthropicToolsValues,
    AnthropicFinishReason,
    AnthropicMcpServerTool,
    AnthropicMcpServerToolConfiguration,
    AnthropicMcpToolConfig,
    AnthropicMcpToolset,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicResponseContentBlockRedactedThinking,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIResponse,
)

REASONING_SUMMARY_PART_SEPARATOR: Final = "\n\n"
RESPONSES_INCLUDE_ENCRYPTED_REASONING: Final = "reasoning.encrypted_content"
_DECODED_JSON: Final = TypeAdapter(object)
_MCP_TOOLSET_TYPE: Final = "mcp_toolset"
_MCP_TOOLSET: Final = TypeAdapter(AnthropicMcpToolset)
_MCP_SERVER: Final = TypeAdapter(AnthropicMcpServerTool)
_WEB_SEARCH_FILTER_KEYS: Final = ("allowed_domains", "blocked_domains")
_STRUCTURED_OUTPUT_NAME: Final = "structured_output"
_OBJECT_MAP: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class _Translation:
    fields: Mapping[str, object]
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ResponsesTools:
    tools: tuple[Mapping[str, object], ...] = ()
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AllTools:
    pass


@dataclass(frozen=True, slots=True)
class _OnlyTools:
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NoEquivalent:
    field: str


_McpAccess: TypeAlias = _AllTools | _OnlyTools | _NoEquivalent


def _present_fields(source: Mapping[str, object], keys: Iterable[str]) -> Mapping[str, object]:
    return {key: source[key] for key in keys if source.get(key) is not None}


def _is_web_search_tool(tool: Mapping[str, object]) -> bool:
    tool_type: Final = tool.get("type", "")
    return (isinstance(tool_type, str) and tool_type.startswith("web_search")) or tool.get("name") == "web_search"


def _web_search_tool(tool: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: Responses tool payload
    """Domain filters only exist on Responses' ``web_search``, so a filtered search moves off ``web_search_preview``."""
    filters: Final = _present_fields(tool, _WEB_SEARCH_FILTER_KEYS)
    location: Final = _present_fields(tool, ("user_location",))
    if filters:
        return {"type": "web_search", "filters": filters, **location}
    return {"type": "web_search_preview", **location}


def _allowed_callers_unsupported(tool: Mapping[str, object]) -> tuple[str, ...]:
    callers: Final = tool.get("allowed_callers")
    if callers is None or callers == ["direct"]:
        return ()
    return (f"allowed_callers on tool {tool.get('name')!r}",)


def _max_tool_calls(
    tools: Sequence[Mapping[str, object]], responses_tools: Sequence[Mapping[str, object]]
) -> _Translation:
    """Responses caps every built-in tool call together, so ``max_uses`` maps only when web search is the one."""
    capped: Final = tuple(tool for tool in tools if _is_web_search_tool(tool) and tool.get("max_uses") is not None)
    if not capped:
        return _Translation(fields={})
    built_in: Final = tuple(tool for tool in responses_tools if tool.get("type") != "function")
    if len(built_in) == 1:
        return _Translation(fields={"max_tool_calls": capped[0].get("max_uses")})
    return _Translation(
        fields={},
        unsupported=tuple(f"max_uses on tool {tool.get('name')!r} next to another built-in tool" for tool in capped),
    )


def _configuration_access(configuration: AnthropicMcpServerToolConfiguration | None) -> _McpAccess:
    if configuration is None:
        return _AllTools()
    if configuration.get("enabled") is False:
        return _OnlyTools(())
    allowed_tools: Final = configuration.get("allowed_tools")
    return _AllTools() if allowed_tools is None else _OnlyTools(tuple(allowed_tools))


def _toolset_access(toolset: AnthropicMcpToolset) -> _McpAccess:
    default_config: Final[AnthropicMcpToolConfig] = toolset.get("default_config", {})
    configs: Final[Mapping[str, AnthropicMcpToolConfig]] = toolset.get("configs", {})
    if not default_config.get("enabled", True):
        return _OnlyTools(tuple(name for name, config in configs.items() if config.get("enabled") is True))
    disabled: Final = tuple(name for name, config in configs.items() if config.get("enabled") is False)
    if not disabled:
        return _AllTools()
    return _NoEquivalent(
        f"mcp_toolset {toolset['mcp_server_name']!r} disabling {', '.join(disabled)} while its other tools stay enabled"
    )


def _toolset_defer_loading_unsupported(toolset: AnthropicMcpToolset) -> tuple[str, ...]:
    default_config: Final[AnthropicMcpToolConfig] = toolset.get("default_config", {})
    configs: Final[Mapping[str, AnthropicMcpToolConfig]] = toolset.get("configs", {})
    deferred: Final = default_config.get("defer_loading") is True or any(
        config.get("defer_loading") is True for config in configs.values()
    )
    return (f"defer_loading on mcp_toolset {toolset['mcp_server_name']!r}",) if deferred else ()


def _mcp_tool(server: AnthropicMcpServerTool) -> Mapping[str, object]:
    """Anthropic's MCP connector calls tools without asking, so the Responses tool must never wait for an approval."""
    connection: Final[Mapping[str, object]] = {
        "type": "mcp",
        "server_label": server.get("name"),
        "server_url": server.get("url"),
        "require_approval": "never",
    }
    authorization: Final = server.get("authorization_token")
    return connection if authorization is None else {**connection, "authorization": authorization}


def _mcp_server_tools(
    server: AnthropicMcpServerTool, access: _McpAccess, unsupported: tuple[str, ...]
) -> _ResponsesTools:
    match access:
        case _AllTools():
            return _ResponsesTools(tools=(_mcp_tool(server),), unsupported=unsupported)
        case _OnlyTools(names=()):
            return _ResponsesTools(unsupported=unsupported)
        case _OnlyTools(names=names):
            return _ResponsesTools(
                tools=({**_mcp_tool(server), "allowed_tools": list(names)},), unsupported=unsupported
            )
        case _NoEquivalent(field=field):
            return _ResponsesTools(unsupported=(*unsupported, field))


def _is_mcp_toolset(tool: Mapping[str, object]) -> bool:
    return tool.get("type") == _MCP_TOOLSET_TYPE


def _requested_mcp_servers(request: AnthropicMessagesRequest) -> Sequence[object]:
    match request:
        case {"mcp_servers": list() as servers}:
            return servers
        case _:
            return ()


def _server_name(server: object) -> str | None:
    fields: Final = _object_map(server)
    name: Final = None if fields is None else fields.get("name")
    return name if isinstance(name, str) else None


def _toolsets_naming_no_server(tools: Sequence[Mapping[str, object]], servers: Sequence[object]) -> tuple[str, ...]:
    names: Final = frozenset(_server_name(server) for server in servers)
    return tuple(
        f"mcp_toolset {tool.get('mcp_server_name')!r} without a matching mcp_servers entry"
        for tool in tools
        if _is_mcp_toolset(tool) and tool.get("mcp_server_name") not in names
    )


def _mcp_server_translation(requested: object, tools: Sequence[Mapping[str, object]]) -> _ResponsesTools:
    """One Responses ``mcp`` tool per server, its access read from the server's ``mcp_toolset`` entry in ``tools``,
    else from the deprecated ``tool_configuration``."""
    try:
        server: Final = _MCP_SERVER.validate_python(requested)
    except ValidationError:
        return _ResponsesTools(unsupported=("a malformed mcp_servers entry",))
    name: Final = server["name"]
    toolset: Final = next(
        (tool for tool in tools if _is_mcp_toolset(tool) and tool.get("mcp_server_name") == name),
        None,
    )
    if toolset is None:
        return _mcp_server_tools(server, _configuration_access(server.get("tool_configuration")), ())
    try:
        parsed: Final = _MCP_TOOLSET.validate_python(toolset)
    except ValidationError:
        return _mcp_server_tools(server, _NoEquivalent(f"the malformed mcp_toolset for {name!r}"), ())
    return _mcp_server_tools(server, _toolset_access(parsed), _toolset_defer_loading_unsupported(parsed))


def _parallel_tool_calls(tool_choice: Mapping[str, object] | None) -> _Translation:
    if (
        tool_choice is None
        or tool_choice.get("type") == "none"
        or tool_choice.get("disable_parallel_tool_use") is not True
    ):
        return _Translation(fields={})
    return _Translation(fields={"parallel_tool_calls": False})


def _service_tier(speed: object) -> _Translation:
    match speed:
        case None | "standard":
            return _Translation(fields={})
        case "fast":
            return _Translation(fields={"service_tier": "priority"})
        case _:
            return _Translation(fields={}, unsupported=(f"speed {speed!r}",))


_FIELDS_WITHOUT_EQUIVALENT: Final = ("top_k", "inference_geo", "container")


def _no_equivalent(request: Mapping[str, object], field: str) -> _Translation:
    return _Translation(fields={}, unsupported=() if request.get(field) is None else (field,))


def _object_map(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT_MAP.validate_python(value)
    except ValidationError:
        return None


def _requested_output_format(request: Mapping[str, object]) -> Mapping[str, object] | None:
    output_format: Final = _object_map(request.get("output_format"))
    if output_format is not None:
        return output_format
    output_config: Final = _object_map(request.get("output_config"))
    return None if output_config is None else _object_map(output_config.get("format"))


def _field_translations(request: Mapping[str, object]) -> tuple[_Translation, ...]:
    return (
        _parallel_tool_calls(_object_map(request.get("tool_choice"))),
        _service_tier(request.get("speed")),
        _text_format(_requested_output_format(request)),
        *(_no_equivalent(request, field) for field in _FIELDS_WITHOUT_EQUIVALENT),
    )


def _text_format(output_format: Mapping[str, object] | None) -> _Translation:
    """A json_schema format goes up strict, as Anthropic structured outputs guarantee the schema. Making every
    property required keeps strict output valid against the caller's schema. An explicit ``strict: false``, a schema
    that is not JSON, or one with a map or free-form object, which OpenAI's strict mode rejects, goes up as is."""
    if output_format is None or output_format.get("type") != "json_schema":
        return _Translation(fields={})
    schema: Final = output_format.get("schema")
    if not schema:
        return _Translation(fields={})
    strict_schema: Final = None if output_format.get("strict") is False else as_json_object(schema)
    if strict_schema is None or leaves_an_object_open(strict_schema):
        loose: Final = {"type": "json_schema", "name": _STRUCTURED_OUTPUT_NAME, "schema": schema, "strict": False}
        return _Translation(fields={"text": {"format": loose}})
    strict: Final = {
        "type": "json_schema",
        "name": _STRUCTURED_OUTPUT_NAME,
        "schema": openai_strict_json_schema(strict_schema),
        "strict": True,
    }
    return _Translation(fields={"text": {"format": strict}})


def _unsupported_params_error(model: str, unsupported: Sequence[str]) -> litellm.UnsupportedParamsError:
    pronoun: Final = "it" if len(unsupported) == 1 else "them"
    return litellm.UnsupportedParamsError(
        message=(
            f"{model} is served through the OpenAI Responses API, which has no equivalent for "
            f"{'; '.join(unsupported)}. To drop {pronoun}, set `litellm.drop_params=True`, or `drop_params: true` "
            "in the model's `litellm_params` or under `litellm_settings` on the proxy"
        ),
        model=model,
    )


def _output_item_mapping(item: object) -> Mapping[str, object]:
    if isinstance(item, BaseModel):
        return item.model_dump()
    return as_str_mapping(item) or MappingProxyType({})


def _tool_input(arguments: object) -> object:
    try:
        return _DECODED_JSON.validate_python(json.loads(str(arguments or "{}")))
    except json.JSONDecodeError:
        return {}


def _text_block_text(block: object) -> str:
    match as_str_mapping(block):
        case {"type": "text", "text": str(text)}:
            return text
        case _:
            return ""


def _mcp_tool_result_text(result: Mapping[str, object]) -> str:
    content: Final = result.get("content")
    return content if isinstance(content, str) else "".join(map(_text_block_text, iter_items(content or ())))


class LiteLLMAnthropicToResponsesAPIAdapter:
    """
    Converts Anthropic /v1/messages requests to OpenAI Responses API format and
    converts Responses API responses back to Anthropic format.
    """

    @staticmethod
    def translate_responses_api_usage_to_anthropic_usage(
        raw_usage: ResponseAPIUsage | None,
    ) -> AnthropicUsage:
        """Map Responses API usage onto Anthropic usage, where ``input_tokens``
        excludes the cache-read and cache-write tokens reported alongside it.
        """
        if raw_usage is None:
            return AnthropicUsage(input_tokens=0, output_tokens=0)

        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.responses.utils import ResponseAPILoggingUtils

        chat_usage = ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(raw_usage)
        return LiteLLMAnthropicMessagesAdapter.translate_openai_usage_to_anthropic_usage(chat_usage)

    # ------------------------------------------------------------------ #
    # Request translation: Anthropic -> Responses API                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_anthropic_image_source_to_url(source: object) -> str | None:
        """Convert Anthropic image source to a URL string."""
        if not isinstance(source, dict):
            return None
        source_type: Final = source.get("type")
        if source_type == "base64":
            media_type: Final = source.get("media_type", "image/jpeg")
            data: Final = source.get("data", "")
            return f"data:{media_type};base64,{data}" if data else None
        elif source_type == "url":
            return source.get("url")
        return None

    @staticmethod
    def _input_image_part(url: str) -> dict[str, object]:  # mutable-ok: API message payload
        return {"type": "input_image", "image_url": url, "detail": "auto"}

    @staticmethod
    def _translate_anthropic_document_block_to_file_part(
        block: Mapping[str, object],
    ) -> dict[str, str] | None:  # mutable-ok: API message payload
        """Convert an Anthropic document block to a Responses input_file part."""
        raw_source: Final = block.get("source")
        if not isinstance(raw_source, Mapping):
            return None
        source: Final = cast(Mapping[str, object], raw_source)  # cast-ok: untrusted client payload
        source_type: Final = source.get("type")
        if source_type == "base64":
            data: Final = source.get("data")
            if not isinstance(data, str) or not data:
                return None
            raw_media_type: Final = source.get("media_type")
            media_type: Final = (
                raw_media_type if isinstance(raw_media_type, str) and raw_media_type else "application/pdf"
            )
            raw_title: Final = block.get("title")
            filename: Final = raw_title if isinstance(raw_title, str) and raw_title else "document.pdf"
            return {
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:{media_type};base64,{data}",
            }
        if source_type == "url":
            url: Final = source.get("url")
            if not isinstance(url, str) or not url:
                return None
            return {"type": "input_file", "file_url": url}
        return None

    @classmethod
    def _tool_result_image_parts(
        cls, blocks: Sequence[Mapping[str, object]]
    ) -> tuple[dict[str, object], ...]:  # mutable-ok: json parts
        """Each image block as an input_image part, keeping a breakpoint the image carries."""
        sources: Final = tuple(
            (cls._translate_anthropic_image_source_to_url(block.get("source")), block.get("prompt_cache_breakpoint"))
            for block in blocks
            if block.get("type") == "image"
        )
        return tuple(
            with_prompt_cache_breakpoint(cls._input_image_part(url), breakpoint) for url, breakpoint in sources if url
        )

    @classmethod
    def _tool_result_images_by_breakpoint(
        cls, block: Mapping[str, object]
    ) -> tuple[object, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:  # mutable-ok: json parts
        """The breakpoint covering a tool_result, its images at or before that breakpoint, and its images after it.

        A mark on the tool_result covers all its images, and a mark on a block inside it covers the images up to that
        block, the images Anthropic caches for the mark. The result's text and documents join into one output, which
        the mark covers whole.
        """
        inner: Final = content_blocks(block.get("content"))
        own_breakpoint: Final = block.get("prompt_cache_breakpoint")
        if own_breakpoint is not None:
            return own_breakpoint, cls._tool_result_image_parts(inner), ()
        last_marked: Final = next(
            (
                position
                for position in reversed(range(len(inner)))
                if inner[position].get("prompt_cache_breakpoint") is not None
            ),
            None,
        )
        if last_marked is None:
            return None, (), cls._tool_result_image_parts(inner)
        return (
            inner[last_marked].get("prompt_cache_breakpoint"),
            cls._tool_result_image_parts(inner[: last_marked + 1]),
            cls._tool_result_image_parts(inner[last_marked + 1 :]),
        )

    @staticmethod
    def _tool_result_images_message(
        image_parts: Sequence[Mapping[str, object]],
        prompt_cache_breakpoint: object,
    ) -> dict[str, object]:  # mutable-ok: API message payload
        """Tool result images as one user message, the tool_result's breakpoint on the last image.

        OpenAI caches only up to a breakpoint, so a marked result's images go right after its output and ahead of any
        later output, the images Anthropic caches for that mark.
        """
        *earlier_images, last_image = image_parts
        return {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
                *earlier_images,
                with_prompt_cache_breakpoint(last_image, prompt_cache_breakpoint),
            ],
        }

    @staticmethod
    def _tool_result_output_value(
        output_text: str,
        file_parts: tuple[dict[str, str], ...],  # mutable-ok: json content parts
        prompt_cache_breakpoint: object,
    ) -> str | list[dict[str, str]]:  # mutable-ok: API message payload
        """Plain string output, or a part list when document file parts or a cache breakpoint are present.

        OpenAI reads a breakpoint only on a part, so a marked result ends on a part that carries it.
        """
        text_parts: Final = [{"type": "input_text", "text": output_text}] if output_text else []
        parts: Final = [*text_parts, *file_parts]
        if not parts or (not file_parts and prompt_cache_breakpoint is None):
            return output_text
        return [*parts[:-1], with_prompt_cache_breakpoint(parts[-1], prompt_cache_breakpoint)]

    @staticmethod
    def _translate_midturn_system_content_to_responses(
        content: str | Iterable[object],
    ) -> list[dict[str, object]]:  # mutable-ok: API message payload
        """Convert in-sequence system content to Responses input-text parts."""
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}] if content else []
        if not isinstance(content, list):
            return []
        return [
            with_prompt_cache_breakpoint({"type": "input_text", "text": text}, block.get("prompt_cache_breakpoint"))
            for raw_block in content
            if (block := as_str_mapping(raw_block)) is not None
            and block.get("type") == "text"
            and (text := block.get("text"))
        ]

    @staticmethod
    def _summary_part_text(part: object) -> str:
        if isinstance(part, Mapping):
            mapping: Final = cast(Mapping[str, Any], part)  # cast-ok: summary parts are untyped provider json
            return str(mapping.get("text") or "")
        return str(getattr(part, "text", None) or "")

    @classmethod
    def thinking_block_from_reasoning_item(
        cls,
        summary: Iterable[object],
        encrypted_content: object,
    ) -> dict[str, object] | None:  # mutable-ok: API message payload
        """The one Anthropic block for a Responses reasoning item.

        The item's encrypted reasoning rides the block's opaque field (`signature`, or
        `data` when there is no summary text) so the client echoes it back and the next
        turn replays the very item OpenAI produced; without it the signature stays empty,
        since only Anthropic can sign a thinking block.
        """
        text: Final = REASONING_SUMMARY_PART_SEPARATOR.join(
            part_text for part in summary if (part_text := cls._summary_part_text(part))
        )
        if not isinstance(encrypted_content, str) or not encrypted_content:
            if not text:
                return None
            return AnthropicResponseContentBlockThinking(type="thinking", thinking=text, signature="").model_dump()
        signature: Final = encrypted_reasoning_signature(encrypted_content)
        if not text:
            return AnthropicResponseContentBlockRedactedThinking(type="redacted_thinking", data=signature).model_dump()
        return AnthropicResponseContentBlockThinking(type="thinking", thinking=text, signature=signature).model_dump()

    @staticmethod
    def _assistant_block_group_key(indexed_block: tuple[int, Mapping[str, object]]) -> str:
        """Group a run of thinking blocks; keep every other block alone."""
        index, block = indexed_block
        match block.get("type"):
            case "thinking" | "redacted_thinking":
                return "thinking"
            case _:
                return f"block:{index}"

    @classmethod
    def _assistant_blocks_to_input_items(
        cls, blocks: Sequence[Mapping[str, object]]
    ) -> tuple[dict[str, object], ...]:  # mutable-ok: API message payload
        mcp_results: Final = MappingProxyType(
            {block.get("tool_use_id"): block for block in blocks if block.get("type") == "mcp_tool_result"}
        )
        return tuple(
            chain.from_iterable(
                cls._assistant_group_to_input_items(tuple(block for _, block in group), mcp_results)
                for _, group in groupby(enumerate(blocks), key=cls._assistant_block_group_key)
            )
        )

    @classmethod
    def _assistant_group_to_input_items(
        cls, group: tuple[Mapping[str, object], ...], mcp_results: Mapping[object, Mapping[str, object]]
    ) -> tuple[dict[str, object], ...]:  # mutable-ok: API message payload
        first: Final = group[0]
        btype: Final = first.get("type")
        if btype in ("thinking", "redacted_thinking"):
            replayed: Final = responses_reasoning_items_from_thinking_blocks(group)
            return tuple(dict(item) for item in replayed)
        if btype == "tool_use":
            return (
                {
                    "type": "function_call",
                    "call_id": first.get("id", ""),
                    "name": first.get("name", ""),
                    "arguments": json.dumps(first.get("input", {})),
                },
            )
        if btype == "mcp_tool_use":
            return (cls._mcp_call_item(first, mcp_results.get(first.get("id"))),)
        return ()

    @staticmethod
    def _mcp_call_item(
        tool_use: Mapping[str, object], result: Mapping[str, object] | None
    ) -> dict[str, object]:  # mutable-ok: API message payload
        call: Final = {
            "type": "mcp_call",
            "id": tool_use.get("id", ""),
            "server_label": tool_use.get("server_name", ""),
            "name": tool_use.get("name", ""),
            "arguments": json.dumps(tool_use.get("input", {})),
        }
        if result is None:
            return call
        text: Final = _mcp_tool_result_text(result)
        return {**call, "error": text} if result.get("is_error") else {**call, "output": text}

    @staticmethod
    def mcp_blocks_from_call(item: object) -> tuple[dict[str, object], dict[str, object]]:  # mutable-ok: API payload
        call: Final = _output_item_mapping(item)
        call_id: Final = str(call.get("id") or "")
        error: Final = call.get("error")
        return (
            {
                "type": "mcp_tool_use",
                "id": call_id,
                "name": str(call.get("name") or ""),
                "server_name": str(call.get("server_label") or ""),
                "input": _tool_input(call.get("arguments")),
            },
            {
                "type": "mcp_tool_result",
                "tool_use_id": call_id,
                "is_error": error is not None,
                "content": [{"type": "text", "text": str(call.get("output") or "") if error is None else str(error)}],
            },
        )

    def translate_messages_to_responses_input(
        self,
        messages: list[AllAnthropicPassThroughMessageValues],
    ) -> list[dict[str, object]]:
        """
        Convert Anthropic messages list to Responses API `input` items.

        Mapping:
          system text        -> message(role=system, input_text)
          user text          -> message(role=user, input_text)
          user image         -> message(role=user, input_image)
          user document      -> message(role=user, input_file)
          user tool_result   -> function_call_output
          assistant text     -> message(role=assistant, output_text)
          assistant thinking -> reasoning
          assistant tool_use -> function_call
          assistant mcp_tool_use + mcp_tool_result -> mcp_call
        """
        input_items: Final[list[dict[str, object]]] = []

        for m in messages:
            if m["role"] == "system":
                system_parts = self._translate_midturn_system_content_to_responses(m.get("content"))
                if system_parts:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "system",
                            "content": system_parts,
                        }
                    )
                continue

            role = m["role"]
            content = m.get("content")

            if role == "user":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    user_parts: list[Mapping[str, object]] = []
                    tool_image_parts: list[dict[str, object]] = []  # mutable-ok: json content parts
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            user_parts.append(
                                with_prompt_cache_breakpoint(
                                    {"type": "input_text", "text": block.get("text", "")},
                                    block.get("prompt_cache_breakpoint"),
                                )
                            )
                        elif btype == "image":
                            url = self._translate_anthropic_image_source_to_url(cast(dict, block.get("source", {})))
                            if url:
                                user_parts.append(
                                    with_prompt_cache_breakpoint(
                                        self._input_image_part(url), block.get("prompt_cache_breakpoint")
                                    )
                                )
                        elif btype == "document":
                            file_part = self._translate_anthropic_document_block_to_file_part(block)
                            if file_part:
                                user_parts.append(
                                    with_prompt_cache_breakpoint(file_part, block.get("prompt_cache_breakpoint"))
                                )
                        elif btype == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            inner = block.get("content")
                            tool_breakpoint, covered_image_parts, later_image_parts = (
                                self._tool_result_images_by_breakpoint(
                                    cast(Mapping[str, object], block)  # cast-ok: TypedDict upcast
                                )
                            )
                            document_candidates = (
                                tuple(
                                    self._translate_anthropic_document_block_to_file_part(c)
                                    for c in inner
                                    if isinstance(c, dict) and c.get("type") == "document"
                                )
                                if isinstance(inner, list)
                                else ()
                            )
                            tool_file_parts = tuple(part for part in document_candidates if part is not None)
                            if inner is None:
                                output_text = ""
                            elif isinstance(inner, str):
                                output_text = inner
                            elif isinstance(inner, list):
                                parts = [
                                    c.get("text", "") for c in inner if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                output_text = "\n".join(parts)
                                if covered_image_parts or later_image_parts:
                                    output_text = (
                                        f"{output_text}\n{TOOL_RESULT_IMAGE_PLACEHOLDER}"
                                        if output_text
                                        else TOOL_RESULT_IMAGE_PLACEHOLDER
                                    )
                            else:
                                output_text = str(inner)
                            images_inside_breakpoint = (
                                (*tool_image_parts, *covered_image_parts) if tool_breakpoint is not None else ()
                            )
                            # tool_result is a top-level item, not inside the message
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_use_id,
                                    "output": self._tool_result_output_value(
                                        output_text,
                                        tool_file_parts,
                                        None if images_inside_breakpoint else tool_breakpoint,
                                    ),
                                }
                            )
                            if images_inside_breakpoint:
                                input_items.append(
                                    self._tool_result_images_message(images_inside_breakpoint, tool_breakpoint)
                                )
                                tool_image_parts.clear()
                            tool_image_parts.extend(later_image_parts)
                    if tool_image_parts:
                        input_items.append(self._tool_result_images_message(tool_image_parts, None))
                    if user_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": user_parts,
                            }
                        )

            elif role == "assistant":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    blocks = tuple(block for block in content if isinstance(block, dict))
                    input_items.extend(self._assistant_blocks_to_input_items(blocks))
                    asst_parts: list[dict[str, object]] = [  # mutable-ok: API message payload
                        {"type": "output_text", "text": block.get("text", "")}
                        for block in blocks
                        if block.get("type") == "text"
                    ]
                    if asst_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": asst_parts,
                            }
                        )

        return input_items

    @staticmethod
    def _translate_tool_to_responses_api(tool_dict: Mapping[str, object]) -> dict[str, object]:
        if _is_web_search_tool(tool_dict):
            return _web_search_tool(tool_dict)
        tool_name: Final = tool_dict.get("name", "")
        # Responses turns strict mode on when `strict` is omitted, silently rewriting
        # `required` to every property. Anthropic tools are non-strict unless asked.
        func_tool: Final[dict[str, object]] = {
            "type": "function",
            "name": tool_name,
            "strict": bool(tool_dict.get("strict")),
        }
        if "description" in tool_dict:
            func_tool["description"] = tool_dict["description"]
        if "input_schema" in tool_dict:
            func_tool["parameters"] = json_copy_or_original(tool_dict["input_schema"])
        return with_prompt_cache_breakpoint(func_tool, tool_dict.get("prompt_cache_breakpoint"))

    def translate_tools_to_responses_api(
        self,
        tools: list[AllAnthropicToolsValues],
    ) -> list[dict[str, object]]:
        """Convert Anthropic tool definitions to Responses API tools. An ``mcp_toolset`` entry configures its MCP
        server's tool, so it never becomes a tool of its own."""
        return [self._translate_tool_to_responses_api(tool) for tool in tools if not _is_mcp_toolset(tool)]

    def _translate_tools(self, anthropic_request: AnthropicMessagesRequest) -> _Translation:
        tools: Final = cast(list[AllAnthropicToolsValues], anthropic_request.get("tools") or [])
        requested_servers: Final = _requested_mcp_servers(anthropic_request)
        mcp_servers: Final = tuple(_mcp_server_translation(server, tools) for server in requested_servers)
        responses_tools: Final = (
            *self.translate_tools_to_responses_api(tools),
            *chain.from_iterable(server.tools for server in mcp_servers),
        )
        max_tool_calls: Final = _max_tool_calls(tools, responses_tools)
        return _Translation(
            fields={**({"tools": list(responses_tools)} if responses_tools else {}), **max_tool_calls.fields},
            unsupported=(
                *chain.from_iterable(_allowed_callers_unsupported(tool) for tool in tools),
                *chain.from_iterable(server.unsupported for server in mcp_servers),
                *_toolsets_naming_no_server(tools, requested_servers),
                *max_tool_calls.unsupported,
            ),
        )

    @staticmethod
    def translate_tool_choice_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
    ) -> str | dict[str, object]:
        """Convert Anthropic tool_choice to Responses API tool_choice."""
        tc_type: Final = tool_choice.get("type")
        if tc_type == "any":
            return "required"
        elif tc_type == "tool":
            return {"type": "function", "name": tool_choice.get("name", "")}
        elif tc_type == "none":
            return "none"
        return "auto"

    @staticmethod
    def translate_context_management_to_responses_api(
        context_management: dict[str, object],
    ) -> list[dict[str, object]] | None:
        """
        Convert Anthropic context_management dict to OpenAI Responses API array format.

        Anthropic format: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
        OpenAI format:    [{"type": "compaction", "compact_threshold": 150000}]
        """
        if not isinstance(context_management, dict):
            return None

        edits: Final = context_management.get("edits", [])
        if not isinstance(edits, list):
            return None

        result: Final[list[dict[str, object]]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                entry: dict[str, object] = {"type": "compaction"}
                trigger = edit.get("trigger")
                if isinstance(trigger, dict) and trigger.get("value") is not None:
                    entry["compact_threshold"] = int(trigger["value"])
                result.append(entry)

        return result if result else None

    @staticmethod
    def _requested_effort(
        thinking: Mapping[str, object],
        configured_effort: object,
        *,
        between_tool_calls: bool,
    ) -> object:
        match thinking.get("type"):
            case "adaptive":
                return configured_effort or "medium"
            case "enabled":
                raw_budget: Final = thinking.get("budget_tokens", 0)
                budget_tokens: Final = int(raw_budget) if isinstance(raw_budget, (int, float)) else 0
                return reasoning_effort_from_thinking_budget(budget_tokens)
            case "disabled":
                return REASONING_EFFORT_OFF
            case "between_tools":
                return (configured_effort or "medium") if between_tool_calls else REASONING_EFFORT_OFF
            case _:
                return configured_effort or None

    @staticmethod
    def _reasoning_summary(thinking: Mapping[str, object]) -> str | None:
        requested: Final = requested_reasoning_summary(thinking)
        if requested:
            return requested
        if thinking.get("display") is not None:
            return None
        return "detailed" if is_reasoning_auto_summary_enabled() else None

    @classmethod
    def translate_thinking_to_reasoning(
        cls,
        thinking: object,
        output_config: object = None,
        *,
        between_tool_calls: bool = False,
    ) -> dict[str, object] | None:
        """
        Convert Anthropic thinking and ``output_config.effort`` to the Responses API reasoning param.

        ``between_tools`` thinking reasons only once the model has tool results to read, so a fresh turn turns
        reasoning off like ``disabled`` does.
        """
        thinking_params: Final[Mapping[str, object]] = as_str_mapping(thinking) or {}
        output_params: Final = as_str_mapping(output_config)
        effort: Final = cls._requested_effort(
            thinking_params,
            output_params.get("effort") if output_params is not None else None,
            between_tool_calls=between_tool_calls,
        )
        if effort is None:
            return None
        if effort == REASONING_EFFORT_OFF:
            return {"effort": effort}
        summary: Final = cls._reasoning_summary(thinking_params)
        return {"effort": effort} if summary is None else {"effort": effort, "summary": summary}

    @classmethod
    def system_blocks_to_instructions(cls, system: object) -> str:
        return "\n".join(
            text
            for block in content_blocks(system)
            if block.get("type") == "text" and isinstance(text := block.get("text"), str) and text
        )

    def translate_request(
        self,
        anthropic_request: AnthropicMessagesRequest,
        include_encrypted_reasoning: bool = True,
        *,
        drop_params: bool = True,
        preserve_system_instructions: bool = True,
    ) -> dict[str, Any]:
        """
        Translate a full Anthropic /v1/messages request dict to
        litellm.responses() / litellm.aresponses() kwargs.

        ``include_encrypted_reasoning`` asks the provider for ``reasoning.encrypted_content``
        on every call, so a reasoning model's items can be replayed intact next turn even
        when the client sent no ``thinking`` block; pass False for a provider whose
        Responses API rejects ``include``.
        """
        model: Final[str] = anthropic_request["model"]
        messages_list: Final = cast(
            list[AllAnthropicPassThroughMessageValues],
            anthropic_request["messages"],
        )

        input_items: Final = self.translate_messages_to_responses_input(messages_list)
        system: Final = system_without_claude_code_billing_header(anthropic_request.get("system"))
        developer_parts: Final = (
            self._translate_midturn_system_content_to_responses(system)
            if isinstance(system, (str, list))
            and (
                not preserve_system_instructions
                or (
                    isinstance(system, list)
                    and any(
                        isinstance(block, dict) and block.get("prompt_cache_breakpoint") is not None for block in system
                    )
                )
            )
            else ()
        )
        if developer_parts:
            input_items.insert(
                0,
                {
                    "type": "message",
                    "role": "developer",
                    "content": developer_parts,
                },
            )

        responses_kwargs: Final[dict[str, object]] = {
            "model": model,
            "input": input_items,
        }
        if include_encrypted_reasoning:
            responses_kwargs["include"] = [RESPONSES_INCLUDE_ENCRYPTED_REASONING]

        if system and not developer_parts:
            if isinstance(system, str):
                responses_kwargs["instructions"] = system
            elif isinstance(system, list):
                responses_kwargs["instructions"] = self.system_blocks_to_instructions(system)

        # max_tokens -> max_output_tokens
        max_tokens: Final = anthropic_request.get("max_tokens")
        if max_tokens:
            responses_kwargs["max_output_tokens"] = max_tokens

        # temperature / top_p passed through
        if "temperature" in anthropic_request:
            responses_kwargs["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            responses_kwargs["top_p"] = anthropic_request["top_p"]

        tool_choice: Final = anthropic_request.get("tool_choice")
        if tool_choice:
            responses_kwargs["tool_choice"] = self.translate_tool_choice_to_responses_api(
                cast(AnthropicMessagesToolChoice, tool_choice)
            )

        translations: Final = (self._translate_tools(anthropic_request), *_field_translations(anthropic_request))
        unsupported: Final = tuple(chain.from_iterable(translation.unsupported for translation in translations))
        if unsupported and not drop_params:
            raise _unsupported_params_error(model, unsupported)
        responses_kwargs.update(chain.from_iterable(translation.fields.items() for translation in translations))

        reasoning: Final = self.translate_thinking_to_reasoning(
            anthropic_request.get("thinking"),
            output_config=anthropic_request.get("output_config"),
            between_tool_calls=is_between_tool_calls(anthropic_request),
        )
        if reasoning:
            responses_kwargs["reasoning"] = reasoning

        # context_management: Anthropic dict -> OpenAI array
        context_management: Final = anthropic_request.get("context_management")
        if isinstance(context_management, dict):
            openai_cm: Final = self.translate_context_management_to_responses_api(context_management)
            if openai_cm is not None:
                responses_kwargs["context_management"] = openai_cm

        # metadata user_id -> user and prompt_cache_key
        metadata: Final = anthropic_request.get("metadata")
        if isinstance(metadata, dict) and "user_id" in metadata:
            responses_kwargs["user"] = str(metadata["user_id"])[:64]
            prompt_cache_key: Final = prompt_cache_key_from_user_id(metadata["user_id"])
            if prompt_cache_key is not None:
                responses_kwargs["prompt_cache_key"] = prompt_cache_key

        return responses_kwargs

    # ------------------------------------------------------------------ #
    # Response translation: Responses API -> Anthropic                    #
    # ------------------------------------------------------------------ #

    def translate_response(
        self,
        response: ResponsesAPIResponse,
    ) -> AnthropicMessagesResponse:
        """
        Translate an OpenAI ResponsesAPIResponse to AnthropicMessagesResponse.
        """
        content: Final[list[dict[str, object]]] = []
        stop_reason: AnthropicFinishReason = "end_turn"
        refusal_text: Final = responses_output_refusal_text(
            cast(Iterable[object], response.output)  # cast-ok: output items re-validated per item
        )

        for item in map(_output_item_mapping, response.output):
            item_type = item.get("type")
            if item_type == "message":
                for part in filter(None, map(as_str_mapping, iter_items(item.get("content") or ()))):
                    part_type = part.get("type")
                    if part_type == "output_text":
                        content.append(
                            AnthropicResponseContentBlockText(
                                type="text", text=str(part.get("text") or "")
                            ).model_dump()
                        )
                    elif part_type == "refusal":
                        content.append(
                            AnthropicResponseContentBlockText(
                                type="text", text=str(part.get("refusal") or "")
                            ).model_dump()
                        )
            elif item_type == "reasoning":
                reasoning_block = self.thinking_block_from_reasoning_item(
                    iter_items(item.get("summary") or item.get("content") or ()),
                    item.get("encrypted_content"),
                )
                if reasoning_block is not None:
                    content.append(reasoning_block)
            elif item_type == "function_call":
                content.append(
                    AnthropicResponseContentBlockToolUse.model_validate(
                        {
                            "type": "tool_use",
                            "id": item.get("call_id") or item.get("id") or "",
                            "name": item.get("name") or "",
                            "input": _tool_input(item.get("arguments")),
                        }
                    ).model_dump(exclude_none=True)
                )
                stop_reason = "tool_use"
            elif item_type == "mcp_call":
                content.extend(self.mcp_blocks_from_call(item))
        if response.status == "incomplete":
            stop_reason = "max_tokens"
        elif refusal_text is not None:
            stop_reason = "refusal"

        anthropic_usage: Final = self.translate_responses_api_usage_to_anthropic_usage(response.usage)

        return AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,
            content=content,
            stop_reason=stop_reason,
            stop_details=(refusal_stop_details(refusal_text) if stop_reason == "refusal" else None),
        )
