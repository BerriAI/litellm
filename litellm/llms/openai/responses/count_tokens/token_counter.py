"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Final

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.handler import (
    OpenAICountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
    ResponsesCountRequestParts,
)
from litellm.types.utils import LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
openai_count_tokens_handler: Final = OpenAICountTokensHandler()


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.OPENAI.value

    @staticmethod
    def _responses_request_parts(
        model_to_use: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        system: object,
        messages_format: TokenCountMessagesFormat,
    ) -> ResponsesCountRequestParts:
        match messages_format:
            case "anthropic":
                return OpenAICountTokensConfig.anthropic_messages_to_responses_input(
                    model=model_to_use, messages=messages, system=system, tools=tools
                )
            case "openai":
                input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
                    [dict(message) for message in messages]
                )
                fallback_instructions: Final = (
                    None if system is None else system if isinstance(system, str) else str(system)
                )
                return OpenAICountTokensConfig.count_request_parts(
                    input_items, instructions if instructions is not None else fallback_instructions, tools
                )
            case _:
                assert_never(messages_format)

    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, object]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, object]] | None = None,
        system: object = None,
        messages_format: TokenCountMessagesFormat = "openai",
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

        input_items, instructions, counted_tools = self._responses_request_parts(
            model_to_use=model_to_use, messages=messages, tools=tools, system=system, messages_format=messages_format
        )

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        try:
            result: Final = await openai_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items,
                api_key=api_key,
                api_base=api_base,
                tools=counted_tools,
                instructions=instructions,
            )

            if result is not None:
                return TokenCountResponse.model_validate(
                    {
                        "total_tokens": result.get("input_tokens", 0),
                        "request_model": request_model,
                        "model_used": model_to_use,
                        "tokenizer_type": "openai_api",
                        "original_response": result,
                    }
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
