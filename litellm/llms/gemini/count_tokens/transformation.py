import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.prompt_templates.common_utils import add_system_prompt_to_messages
from litellm.litellm_core_utils.prompt_templates.factory import map_system_message_pt
from litellm.litellm_core_utils.prompt_templates.image_handling import async_inline_remote_media
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.llms.anthropic.common_utils import sanitize_replayed_anthropic_messages
from litellm.llms.anthropic.pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.anthropic.pass_through.context_management.editors.compact import (
    apply_client_compaction_block_history,
)
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.common_utils import get_supports_system_message
from litellm.llms.vertex_ai.gemini.transformation import (
    _ai_studio_inlines,  # pyright: ignore[reportPrivateUsage]  # shared chat-path media inlining rule
    _transform_system_message,  # pyright: ignore[reportPrivateUsage]  # shared chat-path system splitter
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.vertex_ai import ContentType, SystemInstructions, Tools
from litellm.types.utils import AllMessageValues, CountTokensMessageFormat
from litellm.utils import validate_and_fix_openai_messages


@dataclass(frozen=True, slots=True)
class GeminiCountTokensPayload:
    contents: tuple[ContentType, ...]
    system_instruction: SystemInstructions | None
    tools: tuple[Tools, ...] | None


_ANTHROPIC_ADAPTER: Final = LiteLLMAnthropicMessagesAdapter()
_GEMINI_CHAT_CONFIG: Final = GoogleAIStudioGeminiConfig()


class DeploymentPromptSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_system_prompt: str | None = None
    supports_system_message: bool | None = None


_NO_DEPLOYMENT_PROMPT_SETTINGS: Final = DeploymentPromptSettings()


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
    optional_params: Final = _GEMINI_CHAT_CONFIG.map_openai_params(
        non_default_params=_dict_copy(tool_params),
        optional_params={},  # mutable-ok: the chat mapper writes the Gemini tools into it
        model=model,
        drop_params=False,
    )
    tools: Final = cast(  # cast-ok: the chat mapper stores Gemini Tools under "tools"
        "list[Tools] | None", optional_params.get("tools")
    )
    return tuple(tools) if tools else None


def _messages_like_completion(
    messages: list[AllMessageValues], settings: DeploymentPromptSettings
) -> list[AllMessageValues]:
    validated: Final = validate_and_fix_openai_messages(messages=messages)
    prompted: Final = (
        add_system_prompt_to_messages(
            messages=validated, system_prompt=settings.litellm_system_prompt, merge_with_first_system=True
        )
        if settings.litellm_system_prompt
        else validated
    )
    system_roles: Final = _GEMINI_CHAT_CONFIG.translate_developer_role_to_system_role(messages=prompted)
    return map_system_message_pt(messages=system_roles) if settings.supports_system_message is False else system_roles


@dataclass(frozen=True, slots=True)
class _ChatRequest:
    messages: tuple[AllMessageValues, ...]
    tool_params: Mapping[str, object]


def _chat_request_from_anthropic(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
    settings: DeploymentPromptSettings,
) -> _ChatRequest:
    sanitized: Final = sanitize_replayed_anthropic_messages(
        cast("list[dict[str, object]]", _list_copy(messages))  # cast-ok: _list_copy returns plain dicts
    )
    compacted: Final = apply_client_compaction_block_history(
        messages=sanitized,
        system=cast("str | list[dict[str, object]] | None", system),  # cast-ok: the raw Anthropic system field
    )
    request: Final = cast(  # cast-ok: the same unvalidated dict the /v1/messages adapter builds its request from
        AnthropicMessagesRequest,
        _dict_copy(
            MappingProxyType(
                {
                    key: value
                    for key, value in (
                        ("model", model),
                        ("system", system if compacted is None else compacted.system),
                        ("tools", tools),
                    )
                    if value
                }
            )
        )
        | {"messages": sanitized if compacted is None else compacted.messages},
    )
    openai_request, _ = _ANTHROPIC_ADAPTER.translate_anthropic_to_openai(
        anthropic_message_request=request, custom_llm_provider="gemini"
    )
    return _ChatRequest(
        messages=tuple(_messages_like_completion(openai_request["messages"], settings)),
        tool_params=_tool_params(
            tools=openai_request.get("tools") or (),
            web_search_options=openai_request.get("web_search_options"),
        ),
    )


def _chat_request_from_openai(
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
    settings: DeploymentPromptSettings,
) -> _ChatRequest:
    chat_messages: Final = _list_copy(
        (MappingProxyType({"role": "system", "content": system}), *messages) if system else messages
    )
    return _ChatRequest(
        messages=tuple(
            _messages_like_completion(
                cast("list[AllMessageValues]", chat_messages),  # cast-ok: chat-shaped dicts, copied above
                settings,
            )
        ),
        tool_params=_tool_params(tools=tools or (), web_search_options=None),
    )


def _payload_like_chat(
    model: str, messages: list[AllMessageValues], tool_params: Mapping[str, object]
) -> GeminiCountTokensPayload:
    system_instruction, chat_messages = _transform_system_message(
        supports_system_message=get_supports_system_message(model=model, custom_llm_provider="gemini"),
        messages=messages,
    )
    contents: Final = _GEMINI_CHAT_CONFIG._transform_messages(  # pyright: ignore[reportPrivateUsage]  # shared chat-path converter
        messages=chat_messages, model=model
    )
    return GeminiCountTokensPayload(
        contents=tuple(contents),
        system_instruction=system_instruction,
        tools=_gemini_tools_like_chat(model=model, tool_params=tool_params),
    )


async def build_count_tokens_payload(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    tools: Sequence[Mapping[str, object]] | None,
    message_format: CountTokensMessageFormat,
    settings: DeploymentPromptSettings = _NO_DEPLOYMENT_PROMPT_SETTINGS,
) -> GeminiCountTokensPayload:
    chat_request: Final = await (
        offload_token_count(_chat_request_from_anthropic)(
            model=model, messages=messages, system=system, tools=tools, settings=settings
        )
        if message_format == "anthropic"
        else offload_token_count(_chat_request_from_openai)(
            messages=messages, system=system, tools=tools, settings=settings
        )
    )
    inlined: Final = await async_inline_remote_media(list(chat_request.messages), should_inline=_ai_studio_inlines)
    return await offload_token_count(_payload_like_chat)(
        model=model, messages=inlined, tool_params=chat_request.tool_params
    )
