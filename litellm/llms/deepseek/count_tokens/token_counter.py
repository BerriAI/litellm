"""
DeepSeek token counting through the count_tokens route of its Anthropic-compatible API.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountFields,
    ChatCountTranslationError,
    anthropic_count_fields,
    messages_count_options,
)
from litellm.llms.base_llm.base_utils import (
    BaseTokenCounter,
    deployment_litellm_params,
    deployment_timeout,
    optional_str,
)
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig, deepseek_reasoning_params
from litellm.llms.deepseek.common_utils import DeepSeekModelInfo
from litellm.llms.deepseek.messages.transformation import (
    DeepSeekAnthropicMessagesConfig,
    deepseek_anthropic_base,
    sanitize_tools_for_deepseek,
)
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

DEEPSEEK_TOKENIZER_TYPE: Final = "deepseek_api"
_CHAT_REASONING_PARAMS: Final = ("thinking", "reasoning_effort")
DEEPSEEK_COUNT_TIMEOUT_SECONDS: Final = 10.0
DEEPSEEK_API_HOST: Final = "api.deepseek.com"
_ESTIMATED_COUNT_STATUSES: Final = frozenset({404, 405, 408, 422, 429})
_TOOLS: Final = TypeAdapter(list[dict[str, JsonValue]])
_OPTIONS: Final = TypeAdapter(dict[str, JsonValue])

deepseek_count_tokens_handler: Final = AnthropicCountTokensHandler()


def _count_left_to_estimate(status_code: int) -> bool:
    return status_code in _ESTIMATED_COUNT_STATUSES or status_code >= 500


def _count_timeout(litellm_params: Mapping[str, object]) -> float | httpx.Timeout:
    match timeout := deployment_timeout(litellm_params):
        case None:
            return DEEPSEEK_COUNT_TIMEOUT_SECONDS
        case httpx.Timeout():
            return timeout
        case _:
            return min(timeout, DEEPSEEK_COUNT_TIMEOUT_SECONDS)


def _chat_request_inference_sends(
    tools: Sequence[Mapping[str, JsonValue]] | None, chat_params: Mapping[str, JsonValue]
) -> tuple[list[dict[str, JsonValue]] | None, Mapping[str, JsonValue]]:
    sent: Final = _OPTIONS.validate_python(
        DeepSeekChatConfig.drop_unsupported_tools({**chat_params, **({} if tools is None else {"tools": list(tools)})})
    )
    sent_tools: Final = sent.get("tools")
    return (
        None if sent_tools is None else _TOOLS.validate_python(sent_tools),
        MappingProxyType({name: value for name, value in sent.items() if name != "tools"}),
    )


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
        litellm_params: Final = deployment_litellm_params(deployment)
        api_base: Final = deepseek_anthropic_base(optional_str(litellm_params.get("api_base")))
        if not messages or httpx.URL(api_base).host != DEEPSEEK_API_HOST:
            return None

        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        chat_tools, sent_chat_params = (
            _chat_request_inference_sends(tools, chat_params) if messages_format == "openai" else (tools, chat_params)
        )
        match translated := await anthropic_count_fields(
            model_to_use,
            messages,
            system,
            chat_tools,
            messages_format,
            AnthropicConfig(),
            {name: value for name, value in sent_chat_params.items() if name not in _CHAT_REASONING_PARAMS},
        ):
            case ChatCountTranslationError():
                return translated.response(request_model, model_to_use, DEEPSEEK_TOKENIZER_TYPE)
            case AnthropicCountFields():
                return await self._count(
                    model_to_use=model_to_use,
                    messages=translated.messages,
                    litellm_params=litellm_params,
                    api_base=api_base,
                    request_model=request_model,
                    tools=translated.tools,
                    system=translated.system,
                    count_options=_with_chat_reasoning(translated.options, _chat_reasoning_options(sent_chat_params)),
                )
            case None:
                pass

        if messages_format == "openai":
            return await self._count(
                model_to_use=model_to_use,
                messages=messages,
                litellm_params=litellm_params,
                api_base=api_base,
                request_model=request_model,
                tools=chat_tools,
                system=system,
                count_options=_chat_reasoning_options(sent_chat_params),
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
                    api_base=api_base,
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
        api_base: str,
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
                    api_key=optional_str(litellm_params.get("api_key")),
                ),
                api_base=api_base,
                tools=None if tools is None else _TOOLS.validate_python(sanitize_tools_for_deepseek(tools)),
                system=system,
                optional_params=count_options,
                timeout=_count_timeout(litellm_params),
            )
        except AnthropicError as e:
            if _count_left_to_estimate(e.status_code):
                verbose_logger.warning(
                    "DeepSeek count_tokens route answered %s, so the count is estimated: %s", e.status_code, e.message
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
