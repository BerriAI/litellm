"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import JsonValue
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_base_model
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.bedrock.count_tokens.mantle_handler import BedrockMantleCountTokensHandler
from litellm.llms.bedrock_mantle.common_utils import is_mantle_claude_model
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import LiteLLMPydanticObjectBase, LlmProviders, TokenCountResponse

RUNTIME_TOKENIZER_TYPE: Final = "bedrock_api"
MANTLE_TOKENIZER_TYPE: Final = "bedrock_mantle_api"


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
        and outcome.status_code == 400
        and is_mantle_claude_model(resolved_model)
    )


class BedrockTokenCounter(BaseTokenCounter):
    """Token counter for AWS Bedrock: bedrock-runtime CountTokens first, and Anthropic's count_tokens
    on bedrock-mantle for the Claude models bedrock-runtime answers 400 for"""

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
        request_data: dict[str, object],
        litellm_params: dict[str, object],
        resolved_model: str,
    ) -> CountTokensOutcome:
        try:
            result: Final = await handler.handle_count_tokens_request(
                request_data=request_data,
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
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        litellm_params: Final = (deployment or {}).get("litellm_params", {})
        request_data: Final[dict[str, object]] = {
            "model": model_to_use,
            "messages": messages,
            **({"tools": tools} if tools else {}),
            **({"system": system} if system else {}),
        }
        resolved_model: Final = get_bedrock_base_model(model_to_use)

        runtime: Final = await self._count_with(
            self._runtime_handler, RUNTIME_TOKENIZER_TYPE, request_data, litellm_params, resolved_model
        )
        outcome: Final = (
            await self._count_with(
                self._mantle_handler, MANTLE_TOKENIZER_TYPE, request_data, litellm_params, resolved_model
            )
            if _runtime_rejected_claude_model(runtime, resolved_model)
            else runtime
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
