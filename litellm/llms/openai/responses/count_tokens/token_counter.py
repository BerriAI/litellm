"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

import os
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.handler import (
    OpenAICountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)
from litellm.types.utils import LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
openai_count_tokens_handler = OpenAICountTokensHandler()


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        # Don't use Responses API if user opted out
        if litellm.use_chat_completions_url_for_anthropic_messages:
            return False
        return custom_llm_provider == LlmProviders.OPENAI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Optional[TokenCountResponse]:
        """
        Count tokens using OpenAI's Responses API /input_tokens endpoint.
        """
        if not messages:
            return None

        deployment = deployment or {}
        litellm_params = deployment.get("litellm_params", {})

        # Get OpenAI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            verbose_logger.warning("No OpenAI API key found for token counting")
            return None

        api_base = litellm_params.get("api_base")

        # Convert chat messages to Responses API input format
        input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
            messages
        )

        # Use system param if instructions not extracted from messages
        if instructions is None and system is not None:
            instructions = system if isinstance(system, str) else str(system)

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        try:
            result = await openai_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items if input_items is not None else [],
                api_key=api_key,
                api_base=api_base,
                tools=tools,
                instructions=instructions,
            )

            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="openai_api",
                    original_response=result,
                )
        except OpenAIError as e:
            verbose_logger.warning(
                f"OpenAI CountTokens API error: status={e.status_code}, message={e.message}"
            )
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="openai_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning(f"Error calling OpenAI CountTokens API: {e}")
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="openai_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None
