import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.prompt_templates.image_handling import async_inline_remote_media
from litellm.llms.anthropic.common_utils import sanitize_replayed_anthropic_messages
from litellm.llms.anthropic.pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.common_utils import get_supports_system_message
from litellm.llms.vertex_ai.gemini.transformation import (
    _ai_studio_inlines,  # pyright: ignore[reportPrivateUsage]  # shared chat-path media inlining rule
    _openai_messages_may_need_sync_gcs_metadata_fetch,  # pyright: ignore[reportPrivateUsage]  # shared chat-path check
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # shared chat-path system splitter
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools
from litellm.types.utils import AllMessageValues, CountTokensMessageFormat


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: tuple[ContentType, ...]
    system_instruction: SystemInstructions | None
    tools: tuple[Tools, ...] | None


_ANTHROPIC_ADAPTER: Final = LiteLLMAnthropicMessagesAdapter()


def _dict_copy(value: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: adapter pops keys
    return copy.deepcopy(dict(value))


def _list_copy(value: Sequence[Mapping[str, object]]) -> list[object]:  # mutable-ok: the chat converters take lists
    return [_dict_copy(item) for item in value]  # mutable-ok: the chat converters take plain lists


def _tool_params(tools: Sequence[object], web_search_options: object | None) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: value
            for key, value in (("tools", list(tools) or None), ("web_search_options", web_search_options))
            if value is not None
        }
    )


def _gemini_tools_like_chat(model: str, tool_params: Mapping[str, object]) -> tuple[Tools, ...] | None:
    if not tool_params:
        return None
    optional_params: Final = GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params=_dict_copy(tool_params),
        optional_params={},  # mutable-ok: the chat mapper writes the Gemini tools into it
        model=model,
        drop_params=False,
    )
    tools: Final = cast(  # cast-ok: the chat mapper stores Gemini Tools under "tools"
        "list[Tools] | None", optional_params.get("tools")
    )
    return tuple(tools) if tools else None


def _contents_like_chat(
    model: str, messages: list[AllMessageValues]
) -> tuple[SystemInstructions | None, tuple[ContentType, ...]]:
    system_instruction, chat_messages = _transform_system_message(
        supports_system_message=get_supports_system_message(model=model, custom_llm_provider="gemini"),
        messages=messages,
    )
    contents: Final = GoogleAIStudioGeminiConfig()._transform_messages(  # pyright: ignore[reportPrivateUsage]  # shared chat-path converter
        messages=chat_messages, model=model
    )
    return system_instruction, tuple(contents)


async def _payload_like_chat(
    model: str,
    messages: Sequence[object],
    tool_params: Mapping[str, object],
) -> GeminiCountTokensPayload:
    inlined: Final = await async_inline_remote_media(
        cast("list[AllMessageValues]", list(messages)),  # cast-ok: chat-shaped dicts the caller already copied
        should_inline=_ai_studio_inlines,
    )
    system_instruction, contents = (
        await asyncify(_contents_like_chat)(model, inlined)
        if _openai_messages_may_need_sync_gcs_metadata_fetch(inlined)
        else _contents_like_chat(model, inlined)
    )
    return GeminiCountTokensPayload(
        contents=contents,
        system_instruction=system_instruction,
        tools=_gemini_tools_like_chat(model=model, tool_params=tool_params),
    )


async def _anthropic_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    request: Final = cast(  # cast-ok: the same unvalidated dict the /v1/messages adapter builds its request from
        AnthropicMessagesRequest,
        _dict_copy(
            MappingProxyType(
                {key: value for key, value in (("model", model), ("system", system), ("tools", tools)) if value}
            )
        )
        | {"messages": sanitize_replayed_anthropic_messages(_list_copy(messages))},
    )
    openai_request, _ = _ANTHROPIC_ADAPTER.translate_anthropic_to_openai(
        anthropic_message_request=request, custom_llm_provider="gemini"
    )
    return await _payload_like_chat(
        model=model,
        messages=openai_request["messages"],
        tool_params=_tool_params(
            tools=openai_request.get("tools") or (),
            web_search_options=openai_request.get("web_search_options"),
        ),
    )


async def _openai_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> GeminiCountTokensPayload:
    return await _payload_like_chat(
        model=model,
        messages=_list_copy(
            (MappingProxyType({"role": "system", "content": system}), *messages) if system else messages
        ),
        tool_params=_tool_params(tools=tools or (), web_search_options=None),
    )


async def build_count_tokens_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
    message_format: CountTokensMessageFormat,
) -> GeminiCountTokensPayload:
    if message_format == "anthropic":
        return await _anthropic_payload(model=model, messages=messages, system=system, tools=tools)
    return await _openai_payload(model=model, messages=messages, system=system, tools=tools)
