"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm._logging import CALLER_ERROR_LOG_EXTRA, verbose_logger
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import (
    context_management_betas,
    context_management_with_edits,
    without_unrendered_blocks,
)
from litellm.llms.anthropic.count_tokens.transformation import (
    ChatCountTranslationError,
    anthropic_count_fields,
    counted_tool_choice,
    messages_count_options,
)
from litellm.llms.anthropic.messages.convertible_blocks import deployment_drops_params
from litellm.llms.base_llm.base_utils import BaseTokenCounter, deployment_litellm_params
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.common_utils import (
    BEDROCK_INVOKE_CONTEXT_MANAGEMENT_EDIT_BETAS,
    BedrockError,
    bedrock_capability_model,
    bedrock_converse_supports_parallel_tool_use_config,
    get_bedrock_base_model,
    strip_bedrock_routing_prefix,
)
from litellm.llms.bedrock.count_tokens.handler import (
    BedrockCountTokensCredentialsError,
    BedrockCountTokensHandler,
    BedrockCountTokensRequestError,
    is_caller_refusal,
)
from litellm.llms.bedrock.count_tokens.mantle_handler import BedrockMantleCountTokensHandler
from litellm.llms.bedrock_mantle.common_utils import is_mantle_claude_model
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import (
    CountContextManagement,
    LiteLLMPydanticObjectBase,
    LlmProviders,
    TokenCountMessagesFormat,
    TokenCountResponse,
)

if TYPE_CHECKING:
    from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig

RUNTIME_TOKENIZER_TYPE: Final = "bedrock_api"
MANTLE_TOKENIZER_TYPE: Final = "bedrock_mantle_api"
_RUNTIME_FALLBACK_STATUSES: Final = frozenset({400, 404})
_MODEL_INFO: Final = TypeAdapter(dict[str, object])
_NO_OPTIONS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_COUNTED_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])


class _CountTokensReply(LiteLLMPydanticObjectBase):
    input_tokens: int


@dataclass(frozen=True, slots=True)
class CountedTokens:
    input_tokens: int
    original_response: Mapping[str, object]
    tokenizer_type: str


@dataclass(frozen=True, slots=True)
class CountTokensFailure:
    status_code: int
    message: str
    tokenizer_type: str
    rejected_before_provider_call: bool = False
    invalid_request: bool = False


CountTokensOutcome = CountedTokens | CountTokensFailure


def _deployment_route(litellm_params: Mapping[str, object]) -> tuple[str, LlmProviders] | None:
    deployment_model: Final = litellm_params.get("model")
    if not isinstance(deployment_model, str):
        return None
    provider_hint: Final = litellm_params.get("custom_llm_provider")
    model, provider, _, _ = litellm.get_llm_provider(
        model=deployment_model, custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None
    )
    return model, LlmProviders(provider)


def _deployment_chat_config(litellm_params: Mapping[str, object]) -> BaseConfig | None:
    route: Final = _deployment_route(litellm_params)
    return None if route is None else litellm.ProviderConfigManager.get_provider_chat_config(*route)


def _claude_messages_translator(litellm_params: Mapping[str, object]) -> "AnthropicMessagesConfig":
    from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )

    route: Final = _deployment_route(litellm_params)
    messages_config: Final = (
        None if route is None else litellm.ProviderConfigManager.get_provider_anthropic_messages_config(*route)
    )
    return (
        messages_config
        if isinstance(messages_config, AnthropicMessagesConfig)
        else AmazonAnthropicClaudeMessagesConfig()
    )


def _inference_capability_model(litellm_params: Mapping[str, object], counted_model: str) -> str:
    deployment_model: Final = litellm_params.get("model")
    return bedrock_capability_model(
        strip_bedrock_routing_prefix(deployment_model) if isinstance(deployment_model, str) else counted_model,
        litellm_params.get("base_model"),
    )


def _chat_params_inference_honors(
    deployment_config: BaseConfig | None, chat_params: Mapping[str, JsonValue], capability_model: str
) -> Mapping[str, JsonValue]:
    if not isinstance(deployment_config, litellm.AmazonConverseConfig):
        return chat_params
    if bedrock_converse_supports_parallel_tool_use_config(capability_model):
        return chat_params
    return MappingProxyType({key: value for key, value in chat_params.items() if key != "parallel_tool_calls"})


def _options_inference_sends(
    options: Mapping[str, JsonValue],
    deployment_config: BaseConfig | None,
    chat_params: Mapping[str, JsonValue],
    capability_model: str,
    model_info: Mapping[str, object] | None,
) -> Mapping[str, JsonValue]:
    match options.get("tool_choice"), deployment_config:
        case {"type": "tool", "name": name}, litellm.AmazonConverseConfig() as converse_config if (
            name == RESPONSE_FORMAT_TOOL_NAME
            and not converse_config.response_format_adds_tools(chat_params, capability_model, model_info)
        ):
            return MappingProxyType({key: value for key, value in options.items() if key != "tool_choice"})
        case _:
            return options


def _has_role(message: Mapping[str, object]) -> bool:
    return isinstance(message.get("role"), str)


def _deployment_model_info(deployment: Mapping[str, object] | None) -> Mapping[str, object] | None:
    model_info: Final = (deployment or {}).get("model_info")
    return _MODEL_INFO.validate_python(model_info) if isinstance(model_info, dict) else None


def _converse_messages_as_sent(
    converse_config: "litellm.AmazonConverseConfig",
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    chat_params: Mapping[str, JsonValue],
    model: str,
    model_info: Mapping[str, object] | None,
    litellm_params: Mapping[str, object],
) -> tuple[Mapping[str, object], ...] | ChatCountTranslationError:
    try:
        return converse_config.chat_messages_as_sent(messages, tools, chat_params, model, model_info, litellm_params)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return ChatCountTranslationError(message=f"Invalid request for token counting: {e}")


def _runtime_rejected_claude_model(outcome: CountTokensOutcome, resolved_model: str) -> bool:
    return (
        isinstance(outcome, CountTokensFailure)
        and not outcome.rejected_before_provider_call
        and outcome.status_code in _RUNTIME_FALLBACK_STATUSES
        and is_mantle_claude_model(resolved_model)
    )


def _mantle_retry_outcome(
    runtime: CountTokensOutcome, mantle: CountTokensOutcome, carried_invoke_fields: bool
) -> CountTokensOutcome:
    match runtime, mantle:
        case CountTokensFailure(status_code=400), CountTokensFailure(status_code=404) if carried_invoke_fields:
            return runtime
        case _:
            return mantle


def _request_data(
    model: str, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]] | None, system: object
) -> Mapping[str, object]:
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        **({"tools": [dict(tool) for tool in tools]} if tools else {}),
        **({"system": system} if system else {}),
    }


def _invoke_model_message(message: Mapping[str, object]) -> Mapping[str, object]:
    content: Final = message.get("content")
    return (
        {**message, "content": [{"type": "text", "text": content}]} if isinstance(content, str) and content else message
    )


def _invoke_supports_edit(edit: JsonValue) -> bool:
    edit_type: Final = edit.get("type") if isinstance(edit, dict) else None
    return isinstance(edit_type, str) and edit_type in BEDROCK_INVOKE_CONTEXT_MANAGEMENT_EDIT_BETAS


def _invoke_context_management(context_management: CountContextManagement | None) -> Mapping[str, JsonValue]:
    spec: Final = context_management_with_edits(None if context_management is None else context_management.spec)
    edits: Final = spec.get("edits") if isinstance(spec, dict) else None
    supported_edits: Final = tuple(filter(_invoke_supports_edit, edits if isinstance(edits, list) else ()))
    if not isinstance(spec, dict) or not supported_edits:
        return MappingProxyType({})
    supported_spec: Final[JsonValue] = {**spec, "edits": list(supported_edits)}
    return MappingProxyType(
        {
            "context_management": supported_spec,
            "anthropic_beta": list(context_management_betas(supported_spec)),
        }
    )


def _invoke_tool_choice(
    options: Mapping[str, JsonValue], tools: Sequence[Mapping[str, object]] | None, drops_params: bool
) -> Mapping[str, JsonValue] | ChatCountTranslationError:
    tool_choice: Final = options.get("tool_choice")
    match counted_tool_choice(tool_choice, drops_params):
        case ChatCountTranslationError() as error:
            return error
        case None:
            return _NO_OPTIONS
        case {"type": "auto" | "none"} as choice if not choice.get("disable_parallel_tool_use"):
            return _NO_OPTIONS
        case _ if drops_params and not tools:
            return _NO_OPTIONS
        case _:
            return MappingProxyType({"tool_choice": tool_choice})


class BedrockTokenCounter(BaseTokenCounter):
    """Counts on the configured Bedrock endpoint, with Mantle retry when Runtime rejects a Claude model"""

    def __init__(
        self,
        runtime_handler: BedrockCountTokensHandler | None = None,
        mantle_handler: BedrockMantleCountTokensHandler | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> None:
        self._runtime_handler: Final = runtime_handler or BedrockCountTokensHandler()
        self._mantle_handler: Final = mantle_handler or BedrockMantleCountTokensHandler()
        self._client: Final = client

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.BEDROCK.value

    async def _count_with(
        self,
        handler: BedrockCountTokensHandler | BedrockMantleCountTokensHandler,
        tokenizer_type: str,
        request_data: Mapping[str, object],
        litellm_params: dict[str, object],
        resolved_model: str,
    ) -> CountTokensOutcome:
        try:
            result: Final = await handler.handle_count_tokens_request(
                request_data=dict(request_data),
                litellm_params=litellm_params,
                resolved_model=resolved_model,
                client=self._client,
            )
            reply: Final = _CountTokensReply.model_validate(result)
        except (BedrockCountTokensRequestError, BedrockCountTokensCredentialsError) as e:
            return CountTokensFailure(
                status_code=e.status_code,
                message=e.message,
                tokenizer_type=tokenizer_type,
                rejected_before_provider_call=True,
                invalid_request=isinstance(e, BedrockCountTokensRequestError),
            )
        except BedrockError as e:
            verbose_logger.debug(
                "%s CountTokens API error: status=%s, message=%s", tokenizer_type, e.status_code, e.message
            )
            return CountTokensFailure(status_code=e.status_code, message=e.message, tokenizer_type=tokenizer_type)
        except Exception as e:
            verbose_logger.debug("Error calling %s CountTokens API: %s", tokenizer_type, e)
            return CountTokensFailure(status_code=500, message=str(e), tokenizer_type=tokenizer_type)
        return CountedTokens(input_tokens=reply.input_tokens, original_response=result, tokenizer_type=tokenizer_type)

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, object]] | None,
        contents: Sequence[Mapping[str, object]] | None,
        deployment: Mapping[str, object] | None = None,
        request_model: str = "",
        tools: Sequence[Mapping[str, object]] | None = None,
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
        count_options: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None
        rendered_messages: Final = without_unrendered_blocks(messages)

        use_mantle: Final = model_to_use.startswith("bedrock_mantle/")
        counted_model: Final = model_to_use.removeprefix("bedrock_mantle/")
        resolved_model: Final = get_bedrock_base_model(counted_model)
        primary_tokenizer: Final = MANTLE_TOKENIZER_TYPE if use_mantle else RUNTIME_TOKENIZER_TYPE
        litellm_params: Final = dict(deployment_litellm_params(deployment))
        claude: Final = is_mantle_claude_model(resolved_model)
        deployment_config: Final = _deployment_chat_config(litellm_params) if claude else None
        chat_config: Final = deployment_config if isinstance(deployment_config, AnthropicConfig) else None
        capability_model: Final = _inference_capability_model(litellm_params, counted_model)
        model_info: Final = _deployment_model_info(deployment)
        converse_config: Final = (
            deployment_config
            if messages_format == "openai"
            and isinstance(deployment_config, litellm.AmazonConverseConfig)
            and all(map(_has_role, rendered_messages))
            else None
        )
        counted_input: Final = (
            rendered_messages
            if converse_config is None
            else _converse_messages_as_sent(
                converse_config,
                rendered_messages,
                tools,
                chat_params,
                capability_model,
                model_info,
                litellm_params,
            )
        )
        if isinstance(counted_input, ChatCountTranslationError):
            return counted_input.response(request_model, model_to_use, primary_tokenizer)
        translated: Final = (
            await anthropic_count_fields(
                counted_model,
                counted_input,
                system,
                tools,
                messages_format,
                chat_config or AnthropicConfig(),
                _chat_params_inference_honors(deployment_config, chat_params, capability_model),
                messages_are_converse_turns=converse_config is not None,
                translate_contentless_turns=chat_config is not None,
            )
            if claude
            else None
        )
        if isinstance(translated, ChatCountTranslationError):
            return translated.response(request_model, model_to_use, primary_tokenizer)
        options: Final[Mapping[str, JsonValue] | ChatCountTranslationError] = (
            _options_inference_sends(translated.options, deployment_config, chat_params, capability_model, model_info)
            if translated is not None
            else await asyncify(messages_count_options)(
                _claude_messages_translator(litellm_params),
                counted_model,
                _COUNTED_MESSAGES.validate_python(rendered_messages),
                count_options,
                litellm_params,
            )
            if claude
            else _NO_OPTIONS
        )
        if isinstance(options, ChatCountTranslationError):
            return options.response(request_model, model_to_use, primary_tokenizer)
        native_context_management: Final = _invoke_context_management(context_management if claude else None)
        counted_messages: Final = counted_input if translated is None else translated.messages
        counted_tools: Final = tools if translated is None else translated.tools
        invoke_tool_choice: Final = _invoke_tool_choice(options, counted_tools, deployment_drops_params(deployment))
        if isinstance(invoke_tool_choice, ChatCountTranslationError):
            return invoke_tool_choice.response(request_model, model_to_use, primary_tokenizer)
        invoke_fields: Final = {**native_context_management, **invoke_tool_choice}
        runtime_request: Final = {
            **_request_data(
                model_to_use,
                tuple(map(_invoke_model_message, counted_messages)) if invoke_fields else counted_messages,
                counted_tools,
                system if translated is None else translated.system,
            ),
            **invoke_fields,
        }
        mantle_options: Final = MappingProxyType({key: value for key, value in options.items() if key != "tool_choice"})
        mantle_request: Final = {**runtime_request, **mantle_options}

        primary: Final = (
            await self._count_with(
                self._mantle_handler, primary_tokenizer, mantle_request, litellm_params, resolved_model
            )
            if use_mantle
            else await self._count_with(
                self._runtime_handler, primary_tokenizer, runtime_request, litellm_params, resolved_model
            )
        )
        outcome: Final = (
            _mantle_retry_outcome(
                primary,
                await self._count_with(
                    self._mantle_handler, MANTLE_TOKENIZER_TYPE, mantle_request, litellm_params, resolved_model
                ),
                carried_invoke_fields=bool(invoke_fields),
            )
            if not use_mantle and _runtime_rejected_claude_model(primary, resolved_model)
            else primary
        )
        match outcome:
            case CountedTokens():
                return TokenCountResponse(
                    total_tokens=outcome.input_tokens,
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type=outcome.tokenizer_type,
                    original_response=dict(outcome.original_response),
                )
            case CountTokensFailure():
                refused_for_caller: Final = is_caller_refusal(
                    outcome.status_code, outcome.message, outcome.invalid_request
                )
                verbose_logger.warning(
                    "%s CountTokens API error: status=%s, message=%s",
                    outcome.tokenizer_type,
                    outcome.status_code,
                    outcome.message,
                    extra=CALLER_ERROR_LOG_EXTRA if refused_for_caller else None,
                )
                return TokenCountResponse(
                    total_tokens=0,
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type=outcome.tokenizer_type,
                    error=True,
                    error_message=outcome.message,
                    status_code=outcome.status_code,
                    rejected_before_provider_call=outcome.rejected_before_provider_call,
                )
            case _:
                assert_never(outcome)
