"""
Azure AI Anthropic Token Counter implementation using the CountTokens API.
"""

import os
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.azure_ai.anthropic.count_tokens.handler import (
    AzureAIAnthropicCountTokensHandler,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
azure_ai_anthropic_count_tokens_handler: Final = AzureAIAnthropicCountTokensHandler()
_LITELLM_PARAMS_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_OPTIONAL_STRING_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


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
        deployment: Mapping[str, object] | None = None,
        request_model: str = "",
        tools: list[dict[str, JsonValue]] | None = None,
        system: object | None = None,
        messages_format: TokenCountMessagesFormat = "anthropic",
        output_format: JsonValue = None,
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
        from litellm.llms.anthropic.common_utils import AnthropicError

        if not messages:
            return None

        litellm_params_value: Final[object] = deployment.get("litellm_params", {}) if deployment is not None else {}
        litellm_params: Final[Mapping[str, object]] = _LITELLM_PARAMS_ADAPTER.validate_python(
            litellm_params_value, strict=True
        )

        # Get Azure AI API key from deployment config or environment
        configured_api_key: Final[object] = litellm_params.get("api_key")
        api_key: Final[str | None] = (
            _OPTIONAL_STRING_ADAPTER.validate_python(configured_api_key, strict=True)
            if configured_api_key
            else os.getenv("AZURE_AI_API_KEY")
        )

        configured_api_base: Final[object] = litellm_params.get("api_base")
        api_base: Final = (
            _OPTIONAL_STRING_ADAPTER.validate_python(configured_api_base, strict=True)
            if configured_api_base
            else os.getenv("AZURE_AI_API_BASE") or os.getenv("AZURE_API_BASE") or litellm.api_base
        )

        if not api_base:
            return _missing_api_base_response(request_model, model_to_use)

        try:
            result: Final = await azure_ai_anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                litellm_params=dict(litellm_params),
                tools=tools,
                system=system,
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
