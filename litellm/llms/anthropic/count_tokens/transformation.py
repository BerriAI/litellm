"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import (
    Final,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # caller chat messages go to chat inference translation as-is
)

from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    anthropic_system_blocks,
    split_leading_system_run,
)
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers
from litellm.llms.anthropic.wif import resolve_anthropic_base
from litellm.types.llms.openai import AllMessageValues, ChatCompletionImageObject, ValidUserMessageContentTypes
from litellm.types.utils import TokenCountMessagesFormat, TokenCountResponse

_COUNT_REQUEST: Final = TypeAdapter(dict[str, JsonValue])
_SYSTEM_BLOCKS: Final = TypeAdapter(list[JsonValue])
_IMAGE_BLOCK: Final = TypeAdapter(ChatCompletionImageObject)
COUNT_TOKEN_OPTION_NAMES: Final = ("thinking", "tool_choice", "output_config")
_PLAIN_MESSAGE_KEYS: Final = frozenset({"role", "content"})


def _count_image(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return block
    from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_image_obj

    image_block: Final = _IMAGE_BLOCK.validate_python(block)
    image_url: Final = image_block["image_url"]
    source: Final = convert_to_anthropic_image_obj(
        openai_image_url=image_url if isinstance(image_url, str) else image_url["url"],
        format=image_url.get("format") if isinstance(image_url, dict) else None,
    )
    image: Final = _COUNT_REQUEST.validate_python({"type": "image", "source": source})
    return {**{key: value for key, value in block.items() if key not in {"type", "image_url"}}, **image}


def _count_block(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return _count_image(block)
    content: Final = block.get("content")
    if not isinstance(content, list):
        return block
    return {**block, "content": [_count_image(part) for part in content]}


def _count_content(content: JsonValue) -> JsonValue:
    return [_count_block(block) for block in content] if isinstance(content, list) else content


def _lift_leading_system(
    messages: Sequence[Mapping[str, JsonValue]], system: JsonValue
) -> tuple[tuple[Mapping[str, JsonValue], ...], JsonValue]:
    """Move the leading run of system-role messages into the top-level ``system`` parameter.

    count_tokens only takes the initial system prompt there and answers 400 on ``role: "system"``
    at the head of ``messages``; the chat path sends the same run as ``system``. A caller's own
    ``system`` keeps its place ahead of the lifted blocks, and a ``system`` that is neither text
    nor a block list is left as sent, messages included, for the provider to judge.
    """
    leading, conversation = split_leading_system_run(messages)
    if not leading or not (system is None or isinstance(system, (str, list))):
        return tuple(messages), system
    lifted: Final = _SYSTEM_BLOCKS.validate_python(list(anthropic_system_blocks(leading)))
    if isinstance(system, list):
        return conversation, [*system, *lifted]
    if isinstance(system, str) and system:
        return conversation, [{"type": "text", "text": system}, *lifted]
    return conversation, lifted or system


@dataclass(frozen=True, slots=True)
class AnthropicCountFields:
    messages: tuple[Mapping[str, JsonValue], ...]
    system: JsonValue
    tools: tuple[Mapping[str, JsonValue], ...] | None
    options: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ChatCountTranslationError:
    message: str

    def response(self, request_model: str, model_used: str, tokenizer_type: str) -> TokenCountResponse:
        return TokenCountResponse(
            total_tokens=0,
            request_model=request_model,
            model_used=model_used,
            tokenizer_type=tokenizer_type,
            error=True,
            error_message=self.message,
            status_code=400,
            rejected_before_provider_call=True,
        )


@with_config(ConfigDict(extra="ignore"))
class _AnthropicChatRequest(TypedDict):
    messages: ReadOnly[tuple[Mapping[str, JsonValue], ...]]
    system: NotRequired[ReadOnly[JsonValue]]
    tools: NotRequired[ReadOnly[tuple[Mapping[str, JsonValue], ...]]]
    thinking: NotRequired[ReadOnly[JsonValue]]
    tool_choice: NotRequired[ReadOnly[JsonValue]]
    output_config: NotRequired[ReadOnly[JsonValue]]
    output_format: NotRequired[ReadOnly[JsonValue]]


_ANTHROPIC_CHAT_REQUEST: Final = TypeAdapter(_AnthropicChatRequest)
_ChatParams: TypeAlias = dict[str, object]
_ChatMessages: TypeAlias = list[AllMessageValues]


class _ChatRequestTranslator(Protocol):
    def map_openai_params(
        self, non_default_params: _ChatParams, optional_params: _ChatParams, model: str, drop_params: bool
    ) -> _ChatParams: ...

    def transform_request(
        self,
        model: str,
        messages: _ChatMessages,
        optional_params: _ChatParams,
        litellm_params: _ChatParams,
        headers: _ChatParams,
    ) -> _ChatParams: ...


def _needs_chat_translation(message: Mapping[str, object]) -> bool:
    return message.get("role") == "tool" or not _PLAIN_MESSAGE_KEYS.issuperset(message.keys())


def _is_chat_tool(tool: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils.prompt_templates.factory import is_flat_responses_tool

    return is_flat_responses_tool(tool) or (tool.get("type") == "function" and "function" in tool)


def _has_non_chat_user_block(message: Mapping[str, object]) -> bool:
    content: Final = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return False
    blocks: Final = _SYSTEM_BLOCKS.validate_python(content)
    return any(isinstance(block, dict) and block.get("type") not in ValidUserMessageContentTypes for block in blocks)


def _gets_placeholder_name(tool: Mapping[str, object]) -> bool:
    from litellm.llms.anthropic.pass_through.adapters.transformation import is_provider_native_tool_dict

    name: Final = tool.get("name")
    return not (_is_chat_tool(tool) or is_provider_native_tool_dict(tool) or (isinstance(name, str) and name.strip()))


def _untranslatable_input(
    messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]
) -> ChatCountTranslationError | None:
    invalid_index: Final = next(
        (index for index, message in enumerate(messages) if _has_non_chat_user_block(message)), None
    )
    if invalid_index is not None:
        return ChatCountTranslationError(
            message=f"Invalid user message at index {invalid_index}. "
            "Please ensure all user messages are valid OpenAI chat completion messages."
        )
    unnamed_tool: Final = next(filter(_gets_placeholder_name, tools), None)
    if unnamed_tool is not None:
        return ChatCountTranslationError(
            message=f"Invalid request for token counting: tool type {unnamed_tool.get('type')!r} "
            "has no Chat Completions equivalent"
        )
    return None


def _system_messages(system: object) -> tuple[Mapping[str, object], ...]:
    message: Final[Mapping[str, object]] = {"role": "system", "content": system}
    return (message,) if isinstance(system, (str, list)) and system else ()


def _count_options(request: _AnthropicChatRequest) -> Mapping[str, JsonValue]:
    """Structured output reaches chat inference as the beta ``output_format``, which count_tokens
    refuses; its GA ``output_config.format`` form counts the same schema the same way."""
    output_config: Final = request.get("output_config")
    output_format: Final = request.get("output_format")
    counted_config: Final = {
        **(output_config if isinstance(output_config, dict) else {}),
        **({} if output_format is None else {"format": output_format}),
    }
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("thinking", request.get("thinking")),
                ("tool_choice", request.get("tool_choice")),
                ("output_config", counted_config or None),
            )
            if value is not None
        }
    )


def _translated_chat_request(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object,
    tools: Sequence[Mapping[str, object]],
    chat_params: Mapping[str, object],
    translator: _ChatRequestTranslator,
) -> _ChatParams:
    from litellm.litellm_core_utils.prompt_templates.factory import chat_completion_tools_for

    chat_tools: Final = [dict(chat_tool) for chat_tool in chain.from_iterable(map(chat_completion_tools_for, tools))]
    openai_params: Final = {name: value for name, value in chat_params.items() if name != "drop_params"}
    mapped_params: Final = {**openai_params, **({"tools": chat_tools} if chat_tools else {})}
    chat_messages: Final = cast(  # cast-ok: chat-shaped request messages, read the way chat inference reads them
        _ChatMessages, copy.deepcopy([*_system_messages(system), *messages])
    )
    return translator.transform_request(
        model=model,
        messages=chat_messages,
        optional_params=translator.map_openai_params(
            mapped_params, {}, model, drop_params=chat_params.get("drop_params") is True
        )
        if mapped_params
        else {},
        litellm_params={},
        headers={},
    )


def anthropic_fields_for_chat_tool_use(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object,
    tools: Sequence[Mapping[str, object]] | None,
    translator: _ChatRequestTranslator,
    chat_params: Mapping[str, object] = MappingProxyType({}),
) -> AnthropicCountFields | ChatCountTranslationError | None:
    counted_tools: Final = tools or ()
    if (
        not chat_params
        and not any(map(_is_chat_tool, counted_tools))
        and not any(map(_needs_chat_translation, messages))
    ):
        return None
    untranslatable: Final = _untranslatable_input(messages, counted_tools)
    if untranslatable is not None:
        return untranslatable
    from litellm.exceptions import BadRequestError
    from litellm.llms.anthropic.common_utils import AnthropicError

    try:
        request: Final = _translated_chat_request(model, messages, system, counted_tools, chat_params, translator)
    except (AnthropicError, BadRequestError) as e:
        return ChatCountTranslationError(message=str(e.message))
    except (ValueError, KeyError) as e:
        return ChatCountTranslationError(message=f"Invalid request for token counting: {e}")
    translated: Final = _ANTHROPIC_CHAT_REQUEST.validate_python(request)
    return AnthropicCountFields(
        messages=translated["messages"],
        system=translated.get("system"),
        tools=translated.get("tools"),
        options=_count_options(translated),
    )


async def anthropic_count_fields(
    model: str,
    messages: Sequence[Mapping[str, object]],
    system: object,
    tools: Sequence[Mapping[str, object]] | None,
    messages_format: TokenCountMessagesFormat,
    translator: _ChatRequestTranslator,
    chat_params: Mapping[str, object] = MappingProxyType({}),
) -> AnthropicCountFields | ChatCountTranslationError | None:
    if messages_format != "openai":
        return None
    return await asyncify(anthropic_fields_for_chat_tool_use)(model, messages, system, tools, translator, chat_params)


class AnthropicCountTokensConfig:
    """
    Configuration and transformation logic for Anthropic CountTokens API.

    Anthropic CountTokens API Specification:
    - Endpoint: POST https://api.anthropic.com/v1/messages/count_tokens
    - Beta header required: anthropic-beta: token-counting-2024-11-01
    - Response: {"input_tokens": <number>}
    """

    def get_anthropic_count_tokens_endpoint(self, api_base: str | None = None) -> str:
        """
        Get the Anthropic CountTokens API endpoint.

        Args:
            api_base: The deployment's api_base, which names the chat surface (a host, or a
                base already carrying ``/v1`` or ``/v1/messages``); the count-tokens path is
                appended to it, so it is never the full count-tokens URL. Unset or empty falls
                back to ``ANTHROPIC_API_BASE`` / ``ANTHROPIC_BASE_URL`` and then Anthropic's
                host, the same resolution chat and the federated exchange use

        Returns:
            The endpoint URL for the CountTokens API
        """
        return join_api_base(resolve_anthropic_base(api_base), "v1/messages/count_tokens")

    def transform_request_to_count_tokens(
        self,
        model: str,
        messages: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]] | None = None,
        system: JsonValue = None,
        optional_params: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:  # mutable-ok: provider transport requires JSON dictionaries
        """
        Transform request to Anthropic CountTokens format.

        Includes optional system and tools fields for accurate token counting; a leading run of
        system-role messages is counted through ``system``, the only place count_tokens accepts it.
        """
        options: Final[Mapping[str, JsonValue]] = optional_params or MappingProxyType({})
        counted_messages, counted_system = _lift_leading_system(messages, system)
        return _COUNT_REQUEST.validate_python(
            MappingProxyType(
                {
                    "model": model,
                    "messages": [
                        {**message, "content": _count_content(message["content"])} for message in counted_messages
                    ],
                    **MappingProxyType(
                        {
                            key: value
                            for key, value in (("system", counted_system), ("tools", tools))
                            if value is not None
                        }
                    ),
                    **MappingProxyType(
                        {key: value for key, value in options.items() if key in COUNT_TOKEN_OPTION_NAMES}
                    ),
                }
            )
        )

    def get_count_tokens_headers(self, auth_header: Mapping[str, str]) -> dict[str, str]:
        """The count-tokens headers around a resolved Anthropic auth header
        (``AnthropicModelInfo.get_auth_header``): x-api-key for a static key, an Authorization
        bearer for ``ANTHROPIC_AUTH_TOKEN`` and for sk-ant-oat tokens, whose mandatory oauth beta
        merges with the token-counting beta instead of replacing it."""
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **auth_header,
            "anthropic-beta": merge_anthropic_beta_headers(
                auth_header.get("anthropic-beta"), ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
            ),
        }

    def validate_request(
        self,
        model: str,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        system: JsonValue = None,
        tools: list[dict[str, JsonValue]] | None = None,
    ) -> None:
        """
        Validate the incoming count tokens request.

        Args:
            model: The model name
            messages: The messages to count tokens for

        Raises:
            ValueError: If the request is invalid
        """
        if not model:
            raise ValueError("model parameter is required")

        if not messages and not system and not tools:
            raise ValueError("messages parameter is required")

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValueError(f"Message {i} must have a 'role' field")

            if "content" not in message:
                raise ValueError(f"Message {i} must have a 'content' field")
