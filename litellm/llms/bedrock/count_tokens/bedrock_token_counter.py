"""
Bedrock Token Counter implementation using the CountTokens API.
"""

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from pydantic import JsonValue
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.count_tokens.transformation import (
    NO_CLAUDE_COUNT_OPTIONS,
    ChatCountTranslationError,
    ClaudeCountOptions,
    claude_count_options,
    deployment_inference_model,
    is_request_rejection,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.bedrock.common_utils import BedrockError, get_bedrock_base_model
from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.bedrock.count_tokens.mantle_handler import BedrockMantleCountTokensHandler
from litellm.llms.bedrock_mantle.common_utils import is_mantle_claude_model
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER
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


def _deployment_litellm_params(deployment: Mapping[str, object] | None) -> Mapping[str, object]:
    return as_str_mapping((deployment or {}).get("litellm_params")) or MappingProxyType({})


def _deployment_route(litellm_params: Mapping[str, object]) -> tuple[str, str] | None:
    deployment_model: Final = litellm_params.get("model")
    if not isinstance(deployment_model, str):
        return None
    provider_hint: Final = litellm_params.get("custom_llm_provider")
    model, provider, _, _ = litellm.get_llm_provider(
        model=deployment_model, custom_llm_provider=provider_hint if isinstance(provider_hint, str) else None
    )
    return model, provider


def _claude_options_provider(route: tuple[str, str] | None, use_mantle: bool) -> LlmProviders:
    if not use_mantle:
        return LlmProviders.BEDROCK
    if route is None:
        return LlmProviders.BEDROCK_MANTLE
    model, provider = route
    runs_on_invoke: Final = provider == LlmProviders.BEDROCK.value and not _runs_on_converse(model)
    return LlmProviders.BEDROCK if runs_on_invoke else LlmProviders.BEDROCK_MANTLE


def _runtime_rejected_claude_model(outcome: CountTokensOutcome, resolved_model: str) -> bool:
    return (
        isinstance(outcome, CountTokensFailure)
        and outcome.status_code in _RUNTIME_FALLBACK_STATUSES
        and is_mantle_claude_model(resolved_model)
    )


def _mantle_answered_for_runtime(fallback: CountTokensOutcome, forwarded_options: bool) -> bool:
    return not forwarded_options or isinstance(fallback, CountedTokens) or is_request_rejection(fallback.status_code)


def _invoke_fields(options: Mapping[str, JsonValue], betas: Sequence[str]) -> Mapping[str, object]:
    return {**options, **({"anthropic_beta": list(betas)} if betas else {})}


def _sets_effort(options: Mapping[str, JsonValue]) -> bool:
    output_config: Final = options.get("output_config")
    return isinstance(output_config, dict) and "effort" in output_config


def _runtime_count_betas(claude_options: ClaudeCountOptions) -> tuple[str, ...]:
    if not _sets_effort(claude_options.options) or ANTHROPIC_EFFORT_BETA_HEADER in claude_options.betas:
        return claude_options.betas
    return (*claude_options.betas, ANTHROPIC_EFFORT_BETA_HEADER)


def _runs_on_converse(inference_model: str) -> bool:
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    return BedrockModelInfo.get_bedrock_provider_config_for_messages_api(inference_model) is None


def _options_converse_sends(
    request_options: Mapping[str, JsonValue], inference_model: str | None
) -> Mapping[str, JsonValue]:
    tool_choice: Final = request_options.get("tool_choice")
    if (
        not isinstance(tool_choice, dict)
        or "disable_parallel_tool_use" not in tool_choice
        or inference_model is None
        or not _runs_on_converse(inference_model)
    ):
        return request_options
    return {
        **request_options,
        "tool_choice": {key: value for key, value in tool_choice.items() if key != "disable_parallel_tool_use"},
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

    def _count_on_mantle(
        self, request_data: Mapping[str, object], litellm_params: Mapping[str, object], resolved_model: str
    ) -> Awaitable[Mapping[str, JsonValue]]:
        return self._mantle_handler.handle_count_tokens_request(
            request_data=dict(request_data),
            litellm_params=dict(litellm_params),
            resolved_model=resolved_model,
            client=self._client,
        )

    async def _count_with(self, count: Awaitable[Mapping[str, JsonValue]], tokenizer_type: str) -> CountTokensOutcome:
        try:
            result: Final = await count
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
        request_options: Mapping[str, JsonValue] = MappingProxyType({}),
        anthropic_beta: str | None = None,
    ) -> TokenCountResponse | None:
        if not messages:
            return None

        use_mantle: Final = model_to_use.startswith("bedrock_mantle/")
        counted_model: Final = model_to_use.removeprefix("bedrock_mantle/")
        resolved_model: Final = get_bedrock_base_model(counted_model)
        primary_tokenizer: Final = MANTLE_TOKENIZER_TYPE if use_mantle else RUNTIME_TOKENIZER_TYPE
        litellm_params: Final = _deployment_litellm_params(deployment)
        route: Final = _deployment_route(litellm_params) if is_mantle_claude_model(resolved_model) else None
        claude_options: Final = (
            claude_count_options(
                counted_model,
                _claude_options_provider(route, use_mantle),
                messages,
                tools,
                _options_converse_sends(request_options, deployment_inference_model(deployment)),
                anthropic_beta,
                deployment,
            )
            if is_mantle_claude_model(resolved_model)
            else NO_CLAUDE_COUNT_OPTIONS
        )
        if isinstance(claude_options, ChatCountTranslationError):
            return claude_options.response(request_model, model_to_use, primary_tokenizer)
        invoke_fields: Final = _invoke_fields(claude_options.options, claude_options.betas)
        sent_tools: Final = tools if claude_options.messages is None else claude_options.tools
        counted_request: Final[Mapping[str, object]] = {
            "model": model_to_use,
            "messages": messages
            if claude_options.messages is None
            else [dict(message) for message in claude_options.messages],
            **({"tools": [dict(tool) for tool in sent_tools]} if sent_tools else {}),
            **({"system": system} if system else {}),
        }
        mantle_request: Final = {**counted_request, **invoke_fields}
        runtime_request: Final = {
            **counted_request,
            **_invoke_fields(claude_options.options, _runtime_count_betas(claude_options)),
        }

        primary: Final = (
            await self._count_with(
                self._count_on_mantle(mantle_request, litellm_params, resolved_model), primary_tokenizer
            )
            if use_mantle
            else await self._count_with(
                self._runtime_handler.handle_count_tokens_request(
                    request_data=runtime_request,
                    litellm_params=dict(litellm_params),
                    resolved_model=resolved_model,
                    client=self._client,
                    input_type="invokeModel" if invoke_fields else None,
                ),
                primary_tokenizer,
            )
        )
        fallback: Final = (
            await self._count_with(
                self._count_on_mantle(mantle_request, litellm_params, resolved_model), MANTLE_TOKENIZER_TYPE
            )
            if not use_mantle and _runtime_rejected_claude_model(primary, resolved_model)
            else None
        )
        outcome: Final = (
            fallback
            if fallback is not None
            and _mantle_answered_for_runtime(fallback, forwarded_options=claude_options.carries_request_options)
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
                    carried_request_options=claude_options.carries_request_options,
                )
            case _:
                assert_never(outcome)
