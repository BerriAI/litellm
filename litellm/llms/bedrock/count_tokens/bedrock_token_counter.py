"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from pydantic import JsonValue
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.count_tokens.transformation import ChatCountTranslationError, anthropic_count_fields
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_base_model
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
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

RUNTIME_TOKENIZER_TYPE: Final = "bedrock_api"
MANTLE_TOKENIZER_TYPE: Final = "bedrock_mantle_api"
_RUNTIME_FALLBACK_STATUSES: Final = frozenset({400, 404})


class _CountTokensReply(LiteLLMPydanticObjectBase):
    input_tokens: int


@dataclass(frozen=True, slots=True)
class CountedTokens:
    input_tokens: int
    original_response: Mapping[str, JsonValue]
    tokenizer_type: str


@dataclass(frozen=True, slots=True)
class CountTokensFailure:
    status_code: int
    message: str
    tokenizer_type: str


CountTokensOutcome = CountedTokens | CountTokensFailure


def _runtime_rejected_claude_model(outcome: CountTokensOutcome, resolved_model: str) -> bool:
    return (
        isinstance(outcome, CountTokensFailure)
        and outcome.status_code in _RUNTIME_FALLBACK_STATUSES
        and is_mantle_claude_model(resolved_model)
    )


def _request_data(
    model: str, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]] | None, system: object
) -> Mapping[str, object]:
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        **({"tools": [dict(tool) for tool in tools]} if tools else {}),
        **({"system": system} if system else {}),
    }


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
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: Sequence[Mapping[str, object]] | None = None,
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        use_mantle: Final = model_to_use.startswith("bedrock_mantle/")
        counted_model: Final = model_to_use.removeprefix("bedrock_mantle/")
        resolved_model: Final = get_bedrock_base_model(counted_model)
        primary_tokenizer: Final = MANTLE_TOKENIZER_TYPE if use_mantle else RUNTIME_TOKENIZER_TYPE
        translated: Final = (
            await anthropic_count_fields(
                counted_model, messages, system, tools, messages_format, AnthropicConfig(), chat_params
            )
            if is_mantle_claude_model(resolved_model)
            else None
        )
        if isinstance(translated, ChatCountTranslationError):
            return translated.response(request_model, model_to_use, primary_tokenizer)
        litellm_params: Final = (deployment or {}).get("litellm_params", {})
        runtime_request: Final = (
            _request_data(model_to_use, messages, tools, system)
            if translated is None
            else _request_data(model_to_use, translated.messages, translated.tools, translated.system)
        )
        mantle_request: Final = runtime_request if translated is None else {**runtime_request, **translated.options}

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
            await self._count_with(
                self._mantle_handler, MANTLE_TOKENIZER_TYPE, mantle_request, litellm_params, resolved_model
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
                verbose_logger.warning(
                    "%s CountTokens API error: status=%s, message=%s",
                    outcome.tokenizer_type,
                    outcome.status_code,
                    outcome.message,
                )
                return TokenCountResponse(
                    total_tokens=0,
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type=outcome.tokenizer_type,
                    error=True,
                    error_message=outcome.message,
                    status_code=outcome.status_code,
                )
            case _:
                assert_never(outcome)
