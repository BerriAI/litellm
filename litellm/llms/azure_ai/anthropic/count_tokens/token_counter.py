"""
Azure AI Anthropic Token Counter implementation using the CountTokens API.
"""

import os
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.azure_ai.anthropic.count_tokens.handler import (
    AzureAIAnthropicCountTokensHandler,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
azure_ai_anthropic_count_tokens_handler = AzureAIAnthropicCountTokensHandler()


class AzureAIAnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Azure AI Anthropic provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.AZURE_AI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
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

        deployment = deployment or {}
        litellm_params = deployment.get("litellm_params", {})

        # Get Azure AI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("AZURE_AI_API_KEY")

        # Get API base from deployment config or environment
        api_base = litellm_params.get("api_base")
        if not api_base:
            api_base = os.getenv("AZURE_AI_API_BASE")

        if not api_key:
            verbose_logger.warning("No Azure AI API key found for token counting")
            return None

        if not api_base:
            verbose_logger.warning("No Azure AI API base found for token counting")
            return None

        try:
            result = await azure_ai_anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                litellm_params=litellm_params,
            )

            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="azure_ai_anthropic_api",
                    original_response=result,
                )
        except AnthropicError as e:
            verbose_logger.warning(
                f"Azure AI Anthropic CountTokens API error: status={e.status_code}, message={e.message}"
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
            verbose_logger.warning(
                f"Error calling Azure AI Anthropic CountTokens API: {e}"
            )
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
