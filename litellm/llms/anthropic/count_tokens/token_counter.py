"""
Anthropic Token Counter implementation using the CountTokens API.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from pydantic import JsonValue

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountFields,
    ChatCountTranslationError,
    anthropic_count_fields,
    malformed_message_error,
    messages_count_options,
)
from litellm.llms.base_llm.base_utils import (
    BaseTokenCounter,
    deployment_litellm_params,
    deployment_timeout,
    optional_str,
)
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
anthropic_count_tokens_handler: Final = AnthropicCountTokensHandler()


class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Anthropic provider using the CountTokens API."""

    def __init__(self, client_oauth_token: str | None = None) -> None:
        self.client_oauth_token: Final = client_oauth_token

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.ANTHROPIC.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, JsonValue]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, JsonValue]] | None = None,
        system: JsonValue = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
        count_options: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        """
        Count tokens using Anthropic's CountTokens API.

        Args:
            model_to_use: The model identifier
            messages: The messages to count tokens for
            contents: Alternative content format (not used for Anthropic)
            deployment: Deployment configuration containing litellm_params
            request_model: The original request model name

        Returns:
            TokenCountResponse with token count, or None if counting fails
        """
        if not messages:
            return None

        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig

        litellm_params: Final = deployment_litellm_params(deployment)
        match translated := await anthropic_count_fields(
            model_to_use, messages, system, tools, messages_format, AnthropicConfig(), chat_params
        ):
            case ChatCountTranslationError():
                return translated.response(request_model, model_to_use, "anthropic_api")
            case AnthropicCountFields():
                return await self._count(
                    model_to_use=model_to_use,
                    messages=translated.messages,
                    litellm_params=litellm_params,
                    request_model=request_model,
                    tools=translated.tools,
                    system=translated.system,
                    count_options=translated.options,
                    context_management=context_management,
                )
            case None:
                pass

        if (malformed := malformed_message_error(messages)) is not None:
            return malformed.response(request_model, model_to_use, "anthropic_api")

        match options := await asyncify(messages_count_options)(
            AnthropicMessagesConfig(),
            model_to_use,
            messages,
            count_options,
            litellm_params,
        ):
            case ChatCountTranslationError():
                return options.response(request_model, model_to_use, "anthropic_api")
            case _:
                return await self._count(
                    model_to_use=model_to_use,
                    messages=messages,
                    litellm_params=litellm_params,
                    request_model=request_model,
                    tools=tools,
                    system=system,
                    count_options=options,
                    context_management=context_management,
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
        context_management: CountContextManagement | None,
    ) -> TokenCountResponse | None:
        from litellm.llms.anthropic.common_utils import AnthropicError, AnthropicModelInfo

        api_base: Final = optional_str(litellm_params.get("api_base"))

        try:
            configured_auth_header: Final = await AnthropicModelInfo.aget_auth_header(
                api_key=optional_str(litellm_params.get("api_key")) or litellm.anthropic_key or litellm.api_key,
                api_base=api_base,
                litellm_params=litellm_params,
                allow_workload_identity=True,
            )
            auth_header: Final = (
                configured_auth_header
                if configured_auth_header is not None or self.client_oauth_token is None
                else AnthropicModelInfo.get_auth_header(api_key=self.client_oauth_token, api_base=api_base)
            )
            if auth_header is None:
                return TokenCountResponse(
                    total_tokens=0,
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="anthropic_api",
                    error=True,
                    error_message=(
                        f"No Anthropic API key is set for model {request_model or model_to_use}, so tokens can't be "
                        "counted. Set api_key on the deployment or ANTHROPIC_API_KEY in the proxy environment."
                    ),
                    status_code=401,
                )

            result: Final = await anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=[dict(message) for message in messages],
                auth_header=auth_header,
                api_base=api_base,
                tools=None if tools is None else [dict(tool) for tool in tools],
                system=system,
                optional_params=count_options,
                context_management=context_management.spec if context_management is not None else None,
                drop_params=drops_unsupported_params(litellm_params.get("drop_params")),
                timeout=deployment_timeout(litellm_params),
            )

            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="anthropic_api",
                    original_response=result,
                )
        except (AnthropicError, AuthenticationError) as e:
            verbose_logger.warning("Anthropic CountTokens error: status=%s, message=%s", e.status_code, e.message)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="anthropic_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning("Error calling Anthropic CountTokens API: %s", e)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="anthropic_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None
