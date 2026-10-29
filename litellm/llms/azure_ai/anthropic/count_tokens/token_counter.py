"""
Azure AI Anthropic Token Counter implementation using the CountTokens API.
"""

import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from pydantic import JsonValue

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountFields,
    ChatCountTranslationError,
    anthropic_count_fields,
    messages_count_options,
)
from litellm.llms.azure_ai.anthropic.count_tokens.handler import (
    AzureAIAnthropicCountTokensHandler,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import CountContextManagement, LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
azure_ai_anthropic_count_tokens_handler: Final = AzureAIAnthropicCountTokensHandler()


def _missing_api_base_response(request_model: str, model_to_use: str) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=0,
        request_model=request_model,
        model_used=model_to_use,
        tokenizer_type="azure_ai_anthropic_api",
        error=True,
        error_message=(
            f"No Azure AI API base is set for model {request_model or model_to_use}, so tokens can't be counted. "
            "Set api_base on the deployment or AZURE_AI_API_BASE or AZURE_API_BASE in the proxy environment."
        ),
        status_code=500,
    )


class AzureAIAnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Azure AI Anthropic provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.AZURE_AI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, JsonValue]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, JsonValue]] | None = None,
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
        chat_params: Mapping[str, JsonValue] = MappingProxyType({}),
        count_options: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> TokenCountResponse | None:
        """
        Count tokens using Azure AI Anthropic's CountTokens API.

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

        from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig
        from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig

        match translated := await anthropic_count_fields(
            model_to_use, messages, system, tools, messages_format, AzureAnthropicConfig(), chat_params
        ):
            case ChatCountTranslationError():
                return translated.response(request_model, model_to_use, "azure_ai_anthropic_api")
            case AnthropicCountFields():
                return await self._count(
                    model_to_use=model_to_use,
                    messages=translated.messages,
                    deployment=deployment,
                    request_model=request_model,
                    tools=translated.tools,
                    system=translated.system,
                    count_options=translated.options,
                )
            case None:
                pass

        match options := await asyncify(messages_count_options)(
            AzureAnthropicMessagesConfig(),
            model_to_use,
            messages,
            count_options,
            (deployment or {}).get("litellm_params", {}),
        ):
            case ChatCountTranslationError():
                return options.response(request_model, model_to_use, "azure_ai_anthropic_api")
            case _:
                return await self._count(
                    model_to_use=model_to_use,
                    messages=messages,
                    deployment=deployment,
                    request_model=request_model,
                    tools=tools,
                    system=system,
                    count_options=options,
                )

    async def _count(
        self,
        model_to_use: str,
        messages: Sequence[Mapping[str, JsonValue]],
        deployment: Mapping[str, Any] | None,
        request_model: str,
        tools: Sequence[Mapping[str, JsonValue]] | None,
        system: object | None,
        count_options: Mapping[str, JsonValue],
    ) -> TokenCountResponse | None:
        from litellm.llms.anthropic.common_utils import AnthropicError

        litellm_params: Final = (deployment or {}).get("litellm_params", {})

        # Get Azure AI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("AZURE_AI_API_KEY")

        api_base: Final = (
            litellm_params.get("api_base")
            or os.getenv("AZURE_AI_API_BASE")
            or os.getenv("AZURE_API_BASE")
            or litellm.api_base
        )

        if not api_base:
            return _missing_api_base_response(request_model, model_to_use)

        try:
            result: Final = await azure_ai_anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=[dict(message) for message in messages],
                api_key=api_key,
                api_base=api_base,
                litellm_params=litellm_params,
                tools=None if tools is None else [dict(tool) for tool in tools],
                system=system,
                optional_params=count_options,
            )

            if result is not None:
                return TokenCountResponse.model_validate(
                    {
                        "total_tokens": result.get("input_tokens", 0),
                        "request_model": request_model,
                        "model_used": model_to_use,
                        "tokenizer_type": "azure_ai_anthropic_api",
                        "original_response": result,
                    }
                )
        except AnthropicError as e:
            verbose_logger.warning(
                "Azure AI Anthropic CountTokens API error: status=%s, message=%s", e.status_code, e.message
            )
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="azure_ai_anthropic_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning("Error calling Azure AI Anthropic CountTokens API: %s", e)
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="azure_ai_anthropic_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None
