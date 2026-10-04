"""
Anthropic Token Counter implementation using the CountTokens API.
"""

from typing import Any, Final

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import LlmProviders, TokenCountResponse

# Global handler instance - reuse across all token counting requests
anthropic_count_tokens_handler: Final = AnthropicCountTokensHandler()


class AnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Anthropic provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        return custom_llm_provider == LlmProviders.ANTHROPIC.value

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
        from litellm.llms.anthropic.common_utils import AnthropicError, AnthropicModelInfo

        if not messages:
            return None

        deployment = deployment or {}
        litellm_params: Final = deployment.get("litellm_params", {})
        api_base: Final = litellm_params.get("api_base")

        try:
            auth_header: Final = await AnthropicModelInfo.aget_auth_header(
                api_key=litellm_params.get("api_key"),
                api_base=api_base,
                litellm_params=litellm_params,
                allow_workload_identity=True,
            )
            if auth_header is None:
                verbose_logger.warning("No Anthropic credential found for token counting")
                return None

            result: Final = await anthropic_count_tokens_handler.handle_count_tokens_request(
                model=model_to_use,
                messages=messages,
                auth_header=auth_header,
                api_base=api_base,
                tools=tools,
                system=system,
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
