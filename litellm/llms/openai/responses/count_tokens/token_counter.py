"""
OpenAI Token Counter implementation using the Responses API /input_tokens endpoint.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import JsonValue
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.count_tokens.handler import (
    OpenAICountTokensHandler,
)
from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
    ResponsesCountRequestParts,
)
from litellm.llms.openai.workload_identity import (
    openai_static_api_key,
    openai_workload_identity_requested,
    resolve_openai_bearer_token_async,
)
from litellm.types.utils import LlmProviders, TokenCountMessagesFormat, TokenCountResponse

# Global handler instance - reuse across all token counting requests
openai_count_tokens_handler: Final = OpenAICountTokensHandler()


async def _token_counting_api_key(litellm_params: Mapping[str, object], api_base: str | None) -> str | None:
    deployment_api_key: Final = litellm_params.get("api_key")
    static_api_key: Final = (deployment_api_key if isinstance(deployment_api_key, str) else None) or os.getenv(
        "OPENAI_API_KEY"
    )
    if static_api_key:
        return static_api_key
    if not openai_workload_identity_requested(api_base=api_base, litellm_params=litellm_params):
        return None
    return await resolve_openai_bearer_token_async(
        static_api_key=openai_static_api_key(None), api_base=api_base, litellm_params=litellm_params
    )


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter implementation for OpenAI provider using the Responses API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.OPENAI.value

    @staticmethod
    def _system_instructions(system: object) -> str | None:
        if system is None or isinstance(system, str):
            return system
        return LiteLLMAnthropicToResponsesAPIAdapter.system_blocks_to_instructions(system) or None

    @staticmethod
    def _responses_request_parts(
        model_to_use: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        system: object,
        messages_format: TokenCountMessagesFormat,
        output_format: JsonValue,
    ) -> ResponsesCountRequestParts:
        match messages_format:
            case "anthropic":
                return OpenAICountTokensConfig.anthropic_messages_to_responses_input(
                    model=model_to_use, messages=messages, system=system, tools=tools, output_format=output_format
                )
            case "openai":
                input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
                    [dict(message) for message in messages]
                )
                fallback_instructions: Final = OpenAITokenCounter._system_instructions(system)
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
        output_format: JsonValue = None,
    ) -> TokenCountResponse | None:
        """
        Count tokens using OpenAI's Responses API /input_tokens endpoint.
        """
        if not messages:
            return None

        malformed_error: Final = (
            OpenAICountTokensConfig.malformed_anthropic_messages_error(messages)
            if messages_format == "anthropic"
            else None
        )
        if malformed_error is not None:
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="openai_api",
                error=True,
                error_message=malformed_error,
                status_code=400,
            )

        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})
        api_base: Final = litellm_params.get("api_base")

        input_items, instructions, counted_tools, text = self._responses_request_parts(
            model_to_use=model_to_use,
            messages=messages,
            tools=tools,
            system=system,
            messages_format=messages_format,
            output_format=output_format,
        )

        # If no input items were produced (e.g., system-only messages), fall back to local counting
        if not input_items:
            return None

        try:
            api_key: Final = await _token_counting_api_key(litellm_params, api_base)
            if not api_key:
                verbose_logger.warning("No OpenAI API key found for token counting")
                return None
            result: Final = await openai_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                input=input_items,
                api_key=api_key,
                api_base=api_base,
                tools=counted_tools,
                instructions=instructions,
                text=text,
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
