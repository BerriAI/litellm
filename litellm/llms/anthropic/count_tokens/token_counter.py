"""
Anthropic Token Counter implementation using the CountTokens API.
"""

import os
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
anthropic_count_tokens_handler = AnthropicCountTokensHandler()


class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Anthropic provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.ANTHROPIC.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
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
        from litellm.llms.anthropic.common_utils import AnthropicError

        if not messages:
            return None

        deployment = deployment or {}
        litellm_params = deployment.get("litellm_params", {})

        # Get Anthropic API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            verbose_logger.warning("No Anthropic API key found for token counting")
            return None

        try:
            result = await anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=messages,
                api_key=api_key,
            )

            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="anthropic_api",
                    original_response=result,
                )
        except AnthropicError as e:
            verbose_logger.warning(
                f"Anthropic CountTokens API error: status={e.status_code}, message={e.message}"
            )
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
            verbose_logger.warning(f"Error calling Anthropic CountTokens API: {e}")
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
