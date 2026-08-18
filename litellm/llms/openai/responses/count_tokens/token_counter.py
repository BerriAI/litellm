"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

import os
from typing import Any, Final

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
openai_count_tokens_handler: Final = OpenAICountTokensHandler()


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.OPENAI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, Any]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
    ) -> TokenCountResponse | None:
        """
        Count tokens using OpenAI's Responses API /input_tokens endpoint.
        """
        if not messages:
            return None

        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})

        # Get OpenAI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            verbose_logger.warning("No OpenAI API key found for token counting")
            return None

        api_base: Final = litellm_params.get("api_base")

        # Convert chat messages to Responses API input format
        input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

        # Use system param if instructions not extracted from messages
        if instructions is None and system is not None:
            instructions = system if isinstance(system, str) else str(system)

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        try:
            result: Final = await openai_count_tokens_handler.handle_count_tokens_request(
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
            verbose_logger.warning("OpenAI CountTokens API error: status=%s, message=%s", e.status_code, e.message)
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
            verbose_logger.warning("Error calling OpenAI CountTokens API: %s", e)
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
