"""
DeepSeek token counting through the count_tokens route of its Anthropic-compatible API.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountFields,
    ChatCountTranslationError,
    anthropic_count_fields,
    messages_count_options,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.deepseek.chat.transformation import deepseek_reasoning_params
from litellm.llms.deepseek.common_utils import DeepSeekModelInfo
from litellm.llms.deepseek.messages.transformation import (
    DeepSeekAnthropicMessagesConfig,
    sanitize_tools_for_deepseek,
)
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

DEEPSEEK_TOKENIZER_TYPE: Final = "deepseek_api"
_CHAT_REASONING_PARAMS: Final = ("thinking", "reasoning_effort")
_COUNT_ROUTE_MISSING: Final = frozenset({404, 405})
_TOOLS: Final = TypeAdapter(list[dict[str, JsonValue]])
_OPTIONS: Final = TypeAdapter(dict[str, JsonValue])

deepseek_count_tokens_handler: Final = AnthropicCountTokensHandler()


def _chat_reasoning_options(chat_params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    mapped: Final = deepseek_reasoning_params(chat_params.get("thinking"), chat_params.get("reasoning_effort"))
    effort: Final = mapped.get("reasoning_effort")
    return _OPTIONS.validate_python(
        {
            **({} if mapped.get("thinking") is None else {"thinking": mapped["thinking"]}),
            **({} if effort is None else {"output_config": {"effort": effort}}),
        }
    )


def _with_chat_reasoning(
    options: Mapping[str, JsonValue], reasoning: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    output_config: Final = options.get("output_config")
    reasoning_output_config: Final = reasoning.get("output_config")
    return _OPTIONS.validate_python(
        {
            **options,
            **reasoning,
            **(
                {"output_config": {**output_config, **reasoning_output_config}}
                if isinstance(output_config, dict) and isinstance(reasoning_output_config, dict)
                else {}
            ),
        }
    )


def _litellm_params(deployment: Mapping[str, object] | None) -> Mapping[str, object]:
    params: Final = None if deployment is None else as_str_mapping(deployment.get("litellm_params"))
    return MappingProxyType({}) if params is None else params


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


class DeepSeekTokenCounter(BaseTokenCounter):
    """Counts with the request DeepSeek bills: an Anthropic Messages body sent to the Anthropic-compatible
    count_tokens route, with thinking and effort mapped the way the deployment's inference maps them."""

    def __init__(self, handler: AnthropicCountTokensHandler = deepseek_count_tokens_handler) -> None:
        self._handler: Final = handler

    def should_use_token_counting_api(self, custom_llm_provider: str | None = None) -> bool:
        return custom_llm_provider == LlmProviders.DEEPSEEK.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, JsonValue]] | None,
        contents: Sequence[Mapping[str, object]] | None,
        deployment: Mapping[str, object] | None = None,
        request_model: str = "",
        tools: Sequence[Mapping[str, JsonValue]] | None = None,
        system: JsonValue = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
        count_options: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        litellm_params: Final = _litellm_params(deployment)
        match translated := await anthropic_count_fields(
            model_to_use,
            messages,
            system,
            tools,
            messages_format,
            AnthropicConfig(),
            {name: value for name, value in chat_params.items() if name not in _CHAT_REASONING_PARAMS},
        ):
            case ChatCountTranslationError():
                return translated.response(request_model, model_to_use, DEEPSEEK_TOKENIZER_TYPE)
            case AnthropicCountFields():
                return await self._count(
                    model_to_use=model_to_use,
                    messages=translated.messages,
                    litellm_params=litellm_params,
                    request_model=request_model,
                    tools=translated.tools,
                    system=translated.system,
                    count_options=_with_chat_reasoning(translated.options, _chat_reasoning_options(chat_params)),
                )
            case None:
                pass

        if messages_format == "openai":
            return await self._count(
                model_to_use=model_to_use,
                messages=messages,
                litellm_params=litellm_params,
                request_model=request_model,
                tools=tools,
                system=system,
                count_options=_chat_reasoning_options(chat_params),
            )

        match options := await asyncify(messages_count_options)(
            DeepSeekAnthropicMessagesConfig(), model_to_use, messages, count_options, litellm_params
        ):
            case ChatCountTranslationError():
                return options.response(request_model, model_to_use, DEEPSEEK_TOKENIZER_TYPE)
            case _:
                return await self._count(
                    model_to_use=model_to_use,
                    messages=messages,
                    litellm_params=litellm_params,
                    request_model=request_model,
                    tools=tools,
                    system=system,
                    count_options=options,
                )

    async def _count(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, JsonValue]],
        litellm_params: Mapping[str, object],
        request_model: str,
        tools: Sequence[Mapping[str, JsonValue]] | None,
        system: JsonValue,
        count_options: Mapping[str, JsonValue],
    ) -> TokenCountResponse | None:
        model_info: Final = DeepSeekModelInfo()
        try:
            result: Final = await self._handler.handle_count_tokens_request(
                model=model_to_use,
                messages=[dict(message) for message in messages],
                auth_header=model_info.validate_environment(
                    headers={},
                    model=model_to_use,
                    messages=(),
                    optional_params={},
                    litellm_params=litellm_params,
                    api_key=_optional_str(litellm_params.get("api_key")),
                ),
                api_base=model_info.get_api_base(_optional_str(litellm_params.get("api_base"))),
                tools=None if tools is None else _TOOLS.validate_python(sanitize_tools_for_deepseek(tools)),
                system=system,
                optional_params=count_options,
            )
        except AnthropicError as e:
            if e.status_code in _COUNT_ROUTE_MISSING:
                verbose_logger.debug(
                    "DeepSeek count_tokens route answered %s, so the count is estimated", e.status_code
                )
                return None
            verbose_logger.warning("DeepSeek CountTokens error: status=%s, message=%s", e.status_code, e.message)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type=DEEPSEEK_TOKENIZER_TYPE,
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        input_tokens: Final = result.get("input_tokens")
        return TokenCountResponse(
            total_tokens=input_tokens if isinstance(input_tokens, int) else 0,
            request_model=request_model,
            model_used=model_to_use,
            tokenizer_type=DEEPSEEK_TOKENIZER_TYPE,
            original_response=result,
        )
