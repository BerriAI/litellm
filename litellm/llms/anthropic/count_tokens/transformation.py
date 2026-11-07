"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import (
    anthropic_system_blocks,
    split_leading_system_run,
)
from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers
from litellm.llms.anthropic.wif import resolve_anthropic_base
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.types.llms.openai import ChatCompletionImageObject
from litellm.types.utils import LlmProviders, TokenCountResponse

if TYPE_CHECKING:
    from litellm.types.router import GenericLiteLLMParams

_COUNT_REQUEST: Final = TypeAdapter(dict[str, JsonValue])
_SYSTEM_BLOCKS: Final = TypeAdapter(list[JsonValue])
_IMAGE_BLOCK: Final = TypeAdapter(ChatCompletionImageObject)
COUNT_TOKEN_OPTION_NAMES: Final = ("thinking", "tool_choice", "output_config", "context_management")


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
class ChatCountTranslationError:
    message: str
    carried_request_options: bool = False

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
            carried_request_options=self.carried_request_options,
        )


@with_config(ConfigDict(extra="ignore"))
class _CountedOptions(TypedDict):
    thinking: NotRequired[ReadOnly[JsonValue]]
    tool_choice: NotRequired[ReadOnly[JsonValue]]
    output_config: NotRequired[ReadOnly[JsonValue]]
    output_format: NotRequired[ReadOnly[JsonValue]]
    context_management: NotRequired[ReadOnly[JsonValue]]


@with_config(ConfigDict(extra="ignore"))
class _AnthropicChatRequest(_CountedOptions):
    messages: ReadOnly[tuple[Mapping[str, JsonValue], ...]]
    tools: NotRequired[ReadOnly[tuple[Mapping[str, JsonValue], ...]]]


_ANTHROPIC_CHAT_REQUEST: Final = TypeAdapter(_AnthropicChatRequest)
_INFERENCE_PARAMS: Final = TypeAdapter(dict[str, object])
_HEADERS: Final = TypeAdapter(dict[str, str])


def _count_options(request: _CountedOptions) -> Mapping[str, JsonValue]:
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
                ("context_management", request.get("context_management")),
            )
            if value is not None
        }
    )


@dataclass(frozen=True, slots=True)
class ClaudeCountOptions:
    options: Mapping[str, JsonValue]
    betas: tuple[str, ...]
    messages: tuple[Mapping[str, JsonValue], ...] | None
    tools: tuple[Mapping[str, JsonValue], ...] | None

    @property
    def carries_request_options(self) -> bool:
        return bool(self.options) or bool(self.betas)


NO_CLAUDE_COUNT_OPTIONS: Final = ClaudeCountOptions(options=MappingProxyType({}), betas=(), messages=None, tools=None)
_COUNTED_MESSAGES: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])
_MAX_TOKENS_ABOVE_ANY_BUDGET: Final = 2**31 - 1


_STATUSES_NOT_CAUSED_BY_THE_REQUEST: Final = frozenset({401, 403, 429})


def is_request_rejection(status_code: int) -> bool:
    return 400 <= status_code < 500 and status_code not in _STATUSES_NOT_CAUSED_BY_THE_REQUEST


def deployment_inference_model(deployment: Mapping[str, object] | None) -> str | None:
    deployment_params: Final = _INFERENCE_PARAMS.validate_python((deployment or {}).get("litellm_params") or {})
    deployment_model: Final = deployment_params.get("model")
    if not isinstance(deployment_model, str):
        return None
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    provider_hint: Final = deployment_params.get("custom_llm_provider")
    api_base: Final = deployment_params.get("api_base")
    inference_model, _, _, _ = get_llm_provider(
        model=deployment_model,
        custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None,
        api_base=api_base if isinstance(api_base, str) else None,
    )
    return inference_model


def _messages_transform(model: str, provider: LlmProviders) -> tuple[str, BaseAnthropicMessagesConfig] | None:
    from litellm.utils import ProviderConfigManager

    messages_config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(model, provider)
    return None if messages_config is None else (model, messages_config)


def _transform_litellm_params(deployment: Mapping[str, object] | None) -> "GenericLiteLLMParams":
    from litellm.types.router import GenericLiteLLMParams

    resolved_deployment: Final = deployment or {}
    deployment_params: Final = _INFERENCE_PARAMS.validate_python(resolved_deployment.get("litellm_params") or {})
    return GenericLiteLLMParams.model_validate(
        {
            "drop_params": deployment_params.get("drop_params"),
            "model_info": resolved_deployment.get("model_info"),
        }
    )


def _claude_count_options_for(
    messages_config: BaseAnthropicMessagesConfig,
    model: str,
    litellm_params: "GenericLiteLLMParams",
    provider: LlmProviders,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    sent: Mapping[str, JsonValue],
    anthropic_beta: str | None,
) -> ClaudeCountOptions:
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta

    leading_system, conversation = split_leading_system_run(messages)
    sent_conversation: Final = copy.deepcopy([dict(message) for message in conversation])
    sent_params: Final = {
        **copy.deepcopy(sent),
        **({"tools": copy.deepcopy([dict(tool) for tool in tools])} if tools else {}),
    }
    beta_headers: Final = messages_config.headers_with_anthropic_betas(
        {"anthropic-beta": anthropic_beta} if anthropic_beta else {}, sent_params, sent_conversation
    )
    sent_headers: Final = (
        _HEADERS.validate_python(
            update_headers_with_filtered_beta(
                headers=dict(beta_headers),
                provider=provider.value,
                keep=messages_config.model_specific_anthropic_betas(model, litellm_params),
            )
        )
        if messages_config.should_filter_anthropic_beta_headers()
        else dict(beta_headers)
    )
    request: Final = messages_config.transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the base config signature takes and returns bare dicts
        model=model,
        messages=sent_conversation,
        anthropic_messages_optional_request_params={**sent_params, "max_tokens": _MAX_TOKENS_ABOVE_ANY_BUDGET},
        litellm_params=litellm_params,
        headers=sent_headers,
    )
    transformed: Final = _ANTHROPIC_CHAT_REQUEST.validate_python(request)
    return ClaudeCountOptions(
        options=_count_options(transformed),
        betas=messages_config.sent_anthropic_betas(sent_headers, _INFERENCE_PARAMS.validate_python(request)),
        messages=(*_COUNTED_MESSAGES.validate_python(leading_system), *transformed["messages"]),
        tools=transformed.get("tools"),
    )


def claude_count_options(
    model: str,
    provider: LlmProviders,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    request_options: Mapping[str, JsonValue],
    anthropic_beta: str | None,
    deployment: Mapping[str, object] | None,
) -> ClaudeCountOptions | ChatCountTranslationError:
    sent: Final = {name: value for name, value in request_options.items() if value is not None}
    if not sent:
        return NO_CLAUDE_COUNT_OPTIONS
    inference_model: Final = deployment_inference_model(deployment)
    transform: Final = (
        None if inference_model is None else _messages_transform(inference_model, provider)
    ) or _messages_transform(model, provider)
    if transform is None:
        return NO_CLAUDE_COUNT_OPTIONS
    transform_model, messages_config = transform
    litellm_params: Final = _transform_litellm_params(deployment)
    from litellm.exceptions import BadRequestError
    from litellm.llms.anthropic.common_utils import AnthropicError

    try:
        return _claude_count_options_for(
            messages_config, transform_model, litellm_params, provider, messages, tools, sent, anthropic_beta
        )
    except (AnthropicError, BadRequestError) as e:
        return ChatCountTranslationError(message=str(e.message), carried_request_options=True)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return ChatCountTranslationError(
            message=f"Invalid request for token counting: {e}", carried_request_options=True
        )


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
        return resolve_anthropic_base(api_base) + "/v1/messages/count_tokens"

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

    def get_count_tokens_headers(self, auth_header: Mapping[str, str], betas: Sequence[str] = ()) -> dict[str, str]:
        """The count-tokens headers around a resolved Anthropic auth header
        (``AnthropicModelInfo.get_auth_header``): x-api-key for a static key, an Authorization
        bearer for ``ANTHROPIC_AUTH_TOKEN`` and for sk-ant-oat tokens, whose mandatory oauth beta
        merges with the token-counting beta instead of replacing it."""
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **auth_header,
            "anthropic-beta": merge_anthropic_beta_headers(
                auth_header.get("anthropic-beta"), (ANTHROPIC_TOKEN_COUNTING_BETA_VERSION, *betas)
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
